import os, io, json, re
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from supabase import create_client, Client
import plotly.express as px

# -------- Config --------
st.set_page_config(page_title="Gestor Comercial da Clínica", layout="wide")
SUPABASE_URL = st.secrets["supabase_url"]
SUPABASE_KEY = st.secrets["supabase_anon_key"]
ADMIN_PASS = st.secrets.get("admin_pass", "")
OPENAI_KEY = st.secrets.get("openai_api_key")
OPENAI_MODEL = st.secrets.get("openai_model", "gpt-4o-mini")

@st.cache_resource
def get_sb() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)
sb = get_sb()

# -------- Auth simples --------
def auth_gate():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if not st.session_state.auth_ok:
        pw = st.text_input("Senha", type="password")
        if st.button("Entrar"):
            if pw == ADMIN_PASS and pw:
                st.session_state.auth_ok = True
            else:
                st.error("Senha inválida")
        st.stop()
auth_gate()

# -------- Helpers DB --------
def upsert_protocolo(nome, categoria):
    got = sb.table("protocolos").select("*").eq("nome", nome).limit(1).execute()
    if got.data:
        pid = got.data[0]["protocolo_id"]
        if categoria and got.data[0].get("categoria") != categoria:
            sb.table("protocolos").update({"categoria": categoria}).eq("protocolo_id", pid).execute()
        return pid
    created = sb.table("protocolos").insert({"nome": nome, "categoria": categoria}).execute()
    return created.data[0]["protocolo_id"]

def insert_paciente(nome, tel, cid, dnasc):
    res = sb.table("pacientes").insert({
        "nome": nome or None,
        "telefone": tel or None,
        "cidade_bairro": cid or None,
        "data_nascimento": dnasc or None
    }).execute()
    return res.data[0]["paciente_id"]

def get_or_create_paciente(nome, tel, cid, dnasc):
    q = sb.table("pacientes").select("paciente_id").eq("nome", nome).eq("telefone", tel).limit(1).execute()
    if q.data:
        return q.data[0]["paciente_id"]
    return insert_paciente(nome, tel, cid, dnasc)

def create_atendimento(pid, protocolo_nome, categoria, status, dinicio, dprev, dreal,
                       dose_ini, dose_fin, tcle, medica, origem, obs):
    prot_id = upsert_protocolo(protocolo_nome, categoria) if protocolo_nome else None
    res = sb.table("atendimentos").insert({
        "paciente_id": pid,
        "protocolo_id": prot_id,
        "status": status or None,
        "data_inicio": dinicio or None,
        "data_termino_prevista": dprev or None,
        "data_termino_real": dreal or None,
        "dose_inicial_prescrita": dose_ini or None,
        "dose_final_ajustada": dose_fin or None,
        "tcle_assinado": bool(tcle) if tcle is not None else None,
        "medica": medica or None,
        "origem": origem or None,
        "observacoes": obs or None
    }).execute()
    return res.data[0]["atendimento_id"]

def upsert_pagamento(aid, forma, valor, desconto, custo, parc_prev, parc_quit, dt_ult, situacao):
    def to_num(x):
        if x in ("", None): return None
        x = str(x).replace("R$","").replace(".","").replace(",",".").strip()
        try: return float(x)
        except: return None
    def to_int(x):
        if x in ("", None): return None
        try: return int(x)
        except: return None
    sb.table("pagamentos").insert({
        "atendimento_id": aid,
        "forma_pagamento": forma or None,
        "valor": to_num(valor),
        "desconto": to_num(desconto),
        "custo_estimado": to_num(custo),
        "parcelas_previstas": to_int(parc_prev),
        "parcelas_quitadas": to_int(parc_quit),
        "data_ultimo_pagamento": dt_ult or None,
        "situacao_financeira": situacao or None
    }).execute()

@st.cache_data(ttl=60)
def fetch_v_base(limit=5000):
    res = sb.table("v_base").select("*").limit(limit).execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

# -------- Modelagem para o agente --------
RETORNO_INATIVO_DIAS = 90

def rfm_scoring(df):
    if df.empty or "data_atendimento" not in df: return pd.DataFrame()
    d = df.dropna(subset=["data_atendimento"]).copy()
    d["data_atendimento"] = pd.to_datetime(d["data_atendimento"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    d12 = d[d["data_atendimento"] >= (today - pd.DateOffset(months=12))]
    grp = d12.groupby(["paciente_id","paciente_nome"], as_index=False).agg(
        ultima=("data_atendimento","max"),
        freq=("data_atendimento","count"),
        monet=("ticket_liquido","sum")
    )
    grp["recencia"] = (today - grp["ultima"]).dt.days
    grp["R"] = pd.cut(grp["recencia"], [-1,30,60,90,120,99999], labels=[5,4,3,2,1]).astype(int)
    grp["F"] = pd.cut(grp["freq"], [-1,1,2,3,5,9999], labels=[1,2,3,4,5]).astype(int)
    grp["M"] = pd.cut(grp["monet"], [-1,200,500,1000,2000,9999999], labels=[1,2,3,4,5]).astype(int)
    grp["risco_churn"] = np.where(grp["recencia"] > RETORNO_INATIVO_DIAS, "alto",
                           np.where(grp["recencia"] > 60, "médio", "baixo"))
    return grp

def retention_cohort(df):
    if df.empty or "data_atendimento" not in df or "paciente_id" not in df: return pd.DataFrame()
    d = df.dropna(subset=["data_atendimento","paciente_id"]).copy()
    d["data_atendimento"] = pd.to_datetime(d["data_atendimento"], errors="coerce")
    first = d.groupby("paciente_id")["data_atendimento"].min().rename("first_date")
    d = d.join(first, on="paciente_id")
    d["cohort"] = d["first_date"].dt.to_period("M").astype(str)
    d["period"] = (d["data_atendimento"].dt.to_period("M") - d["first_date"].dt.to_period("M")).apply(lambda x: x.n).astype(int)
    pv = d.pivot_table(index="cohort", columns="period", values="paciente_id", aggfunc=pd.Series.nunique)
    base = pv.iloc[:,0].replace(0, np.nan)
    return pv.divide(base, axis=0).fillna(0).round(3)

def protocol_performance(df):
    if df.empty or "protocolo" not in df: return pd.DataFrame()
    g = df.groupby("protocolo", as_index=False).agg(
        atendimentos=("atendimento_id","count"),
        receita=("ticket_liquido","sum"),
        ticket_medio=("ticket_liquido","mean")
    )
    return g.sort_values("receita", ascending=False)

def upsell_opportunities(df, rfm):
    if df.empty or rfm.empty: return pd.DataFrame()
    last = df.groupby(["paciente_id","paciente_nome"])["data_atendimento"].max().reset_index(name="ultima")
    last["ultima"] = pd.to_datetime(last["ultima"], errors="coerce")
    base = rfm.merge(last, on=["paciente_id","paciente_nome"], how="left")
    base["dias_ult"] = (pd.Timestamp.today().normalize() - base["ultima"]).dt.days
    cand = base[(base["F"] >= 2) & (base["M"].between(2,4)) & (base["dias_ult"].between(31,120))]
    top = protocol_performance(df).head(1)
    sugestao = top["protocolo"].iloc[0] if len(top) else None
    out = cand[["paciente_id","paciente_nome","risco_churn"]].copy()
    out["sugestao_proximo_protocolo"] = sugestao
    return out

# -------- Agente (planeja → executa → relatório) --------
def df_to_csv_text(df, max_rows=1000):
    if df is None or df.empty: return ""
    buf = io.StringIO(); df.head(max_rows).to_csv(buf, index=False); return buf.getvalue()

def agente_planejar(df, hint_question=None):
    base_stats = {
        "linhas": int(len(df)),
        "periodo_min": str(pd.to_datetime(df["data_atendimento"]).min().date()) if "data_atendimento" in df and not df.empty else None,
        "periodo_max": str(pd.to_datetime(df["data_atendimento"]).max().date()) if "data_atendimento" in df and not df.empty else None,
        "colunas": list(df.columns)
    }
    context = (
        "Você é um gestor comercial de clínica. Gere um plano de análise para aumentar retenção e venda de protocolos, "
        "com 5-10 ações no máximo, focando próximos 14 dias."
    )
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
    messages = [
        {"role":"system","content":context},
        {"role":"user","content":"Resumo dos dados:\n" + json.dumps(base_stats, ensure_ascii=False)}
    ]
    if hint_question:
        messages.append({"role":"user","content":"Pergunta guia (opcional): " + hint_question})
    resp = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=0.2, messages=messages, max_tokens=700
    )
    # aceitar texto livre em JSON-ish; se não vier JSON, seguimos mesmo assim
    txt = resp.choices[0].message.content.strip()
    # fallback simples: extrair blocos de 'tool:' e 'params:' se não for JSON
    plano = {"objetivos":[], "perguntas":[], "acoes":[], "metricas_chave":[]}
    try:
        plano = json.loads(txt)
    except:
        # plano textual; cria ações padrão
        plano["acoes"] = [
            {"tool":"rfm_scoring","params":{}},
            {"tool":"cohort_retencao","params":{}},
            {"tool":"protocol_performance","params":{}},
            {"tool":"pacientes_inativos","params":{"dias_inativos":90}},
            {"tool":"upsell_opportunities","params":{}}
        ]
    return plano

def agente_executar(df, plano):
    resultados = []
    for ac in plano.get("acoes", []):
        tool = str(ac.get("tool","")).lower()
        params = ac.get("params",{}) or {}
        saida = pd.DataFrame()
        if tool == "rfm_scoring":
            saida = rfm_scoring(df)
        elif tool == "cohort_retencao":
            saida = retention_cohort(df)
        elif tool == "protocol_performance":
            saida = protocol_performance(df)
        elif tool == "pacientes_inativos":
            dias = int(params.get("dias_inativos", 90))
            last = df.groupby(["paciente_id","paciente_nome"])["data_atendimento"].max().reset_index(name="ultima")
            last["ultima"] = pd.to_datetime(last["ultima"], errors="coerce")
            last["dias_sem_visita"] = (pd.Timestamp.today().normalize() - last["ultima"]).dt.days
            saida = last[last["dias_sem_visita"] > dias].sort_values("dias_sem_visita", ascending=False)
        elif tool == "upsell_opportunities":
            base_rfm = rfm_scoring(df)
            saida = upsell_opportunities(df, base_rfm)
        resultados.append({
            "tool": tool, "params": params,
            "tamanho": 0 if saida is None else (saida.shape[0] if isinstance(saida, pd.DataFrame) else 0),
            "amostra_csv": df_to_csv_text(saida, 200)
        })
    return resultados

def agente_relatorio(df, plano, resultados):
    evidencias = []
    for r in resultados:
        evidencias.append({
            "tool": r["tool"],
            "params": r["params"],
            "tamanho": r["tamanho"],
            "amostra_csv": r["amostra_csv"][:8000]
        })
    prompt = (
        "Com base nas evidências e no plano, escreva um Relatório Executivo em 5 seções: "
        "1) Resumo, 2) Principais Achados, 3) Oportunidades Prioritárias, 4) Metas para 14 dias, "
        "5) Próximas Ações com listas nominativas e scripts de abordagem quando possível. Seja específico e acionável."
    )
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=0.2, max_tokens=900,
        messages=[
            {"role":"system","content":"Você é um gestor comercial objetivo e orientado a dados."},
            {"role":"user","content":prompt + "\n\nPlano:\n" + json.dumps(plano, ensure_ascii=False) + "\n\nEvidências:\n" + json.dumps(evidencias, ensure_ascii=False)}
        ]
    )
    return resp.choices[0].message.content.strip()

# -------- UI --------
st.title("Gestor Comercial da Clínica")

with st.expander("Cadastrar novo atendimento", expanded=True):
    colA, colB, colC = st.columns(3)
    with colA:
        nome = st.text_input("Nome completo")
        telefone = st.text_input("Telefone / WhatsApp")
        cidade = st.text_input("Cidade / Bairro")
        data_nasc = st.date_input("Data de nascimento", value=None)
    with colB:
        protocolo_nome = st.text_input("Protocolo (ex: Semaglutida)")
        categoria = st.selectbox("Categoria", ["Emagrecimento", "Estética", "Outros"])
        status = st.selectbox("Status", ["Em curso","Concluído","Cancelado"])
        medica = st.text_input("Médica")
    with colC:
        origem = st.text_input("Origem (opcional)")
        tcle_assinado = st.checkbox("TCLE assinado?")
        dose_ini = st.text_input("Dose inicial prescrita", value="")
        dose_fin = st.text_input("Dose final ajustada", value="")
    colD, colE, colF = st.columns(3)
    with colD:
        data_inicio = st.date_input("Data de início", value=date.today())
    with colE:
        data_prev = st.date_input("Data prevista de término", value=None)
        data_real = st.date_input("Data de término real", value=None)
    with colF:
        forma = st.text_input("Forma de pagamento", value="Parcelado")
        valor = st.text_input("Valor (ex: 1500.00)")
        desconto = st.text_input("Desconto (ex: 0.00)")
        custo = st.text_input("Custo estimado (opcional)")
        parc_prev = st.text_input("Parcelas previstas", value="")
        parc_quit = st.text_input("Parcelas quitadas", value="")
        dt_ult = st.date_input("Data do último pagamento", value=None)
        situacao = st.text_input("Situação financeira", value="Em dia")
    obs = st.text_area("Observações", height=80)

    if st.button("Salvar atendimento"):
        if not nome:
            st.error("Informe o nome.")
        else:
            pid = get_or_create_paciente(nome, telefone, cidade, data_nasc)
            aid = create_atendimento(pid, protocolo_nome, categoria, status, data_inicio, data_prev, data_real, dose_ini, dose_fin, tcle_assinado, medica, origem, obs)
            upsert_pagamento(aid, forma, valor, desconto, custo, parc_prev, parc_quit, dt_ult, situacao)
            fetch_v_base.clear()
            st.success("Atendimento cadastrado.")

st.subheader("Atendimentos recentes")
df = fetch_v_base()
st.dataframe(df, use_container_width=True)

# KPIs
if not df.empty:
    def num(x): 
        try: return float(x)
        except: return 0.0
    df["ticket_liquido"] = df["ticket_liquido"].apply(num) if "ticket_liquido" in df else 0.0
    col1, col2, col3, col4 = st.columns(4)
    receita_total = df["ticket_liquido"].sum() if "ticket_liquido" in df else 0.0
    with col1: st.metric("Receita total", f"R$ {receita_total:,.2f}")
    with col2: st.metric("Atendimentos", len(df))
    with col3: st.metric("Pacientes únicos", df["paciente_id"].nunique() if "paciente_id" in df else 0)
    with col4:
        tm = receita_total/len(df) if len(df) else 0
        st.metric("Ticket médio", f"R$ {tm:,.2f}")

    if "protocolo" in df.columns:
        top = df.groupby("protocolo", as_index=False)["ticket_liquido"].sum().sort_values("ticket_liquido", ascending=False)
        fig = px.bar(top.head(10), x="protocolo", y="ticket_liquido")
        st.plotly_chart(fig, use_container_width=True)

# Agente
st.subheader("Agente Comercial")
hint = st.text_area("Pergunta opcional para guiar (deixe vazio para plano automático)")
if st.button("Rodar agente"):
    if df.empty:
        st.warning("Sem dados para análise.")
    else:
        with st.spinner("Planejando"):
            plano = agente_planejar(df, hint_question=hint)
        with st.spinner("Executando análises"):
            resultados = agente_executar(df, plano)
        with st.spinner("Gerando relatório executivo"):
            rel = agente_relatorio(df, plano, resultados)
        st.markdown("Relatório Executivo")
        st.write(rel)
        # Downloads das amostras
        for r in resultados:
            if r["amostra_csv"]:
                st.download_button(f"Baixar {r['tool']}.csv", r["amostra_csv"], file_name=f"{r['tool']}.csv", mime="text/csv")
