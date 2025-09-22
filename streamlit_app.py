import os, io, json, re, math
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from supabase import create_client, Client
import plotly.express as px

# =========================
# Config & Setup
# =========================
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

# =========================
# Auth simples
# =========================
def auth_gate():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if not st.session_state.auth_ok:
        st.title("Login")
        pw = st.text_input("Senha", type="password")
        if st.button("Entrar"):
            if pw == ADMIN_PASS and pw:
                st.session_state.auth_ok = True
            else:
                st.error("Senha inválida")
        st.stop()
auth_gate()

# =========================
# Helpers de formato/validação
# =========================
def iso(d): return d.isoformat() if d else None

def to_num(x):
    if x in ("", None): return None
    x = str(x).replace("R$","").replace(".","").replace(",",".").strip()
    try: return float(x)
    except: return None

def to_int(x):
    if x in ("", None): return None
    try: return int(x)
    except: return None

def nonempty(s): return s if (s and str(s).strip()) else None

# =========================
# Acesso ao DB
# =========================
def insert_paciente(nome, tel, cid, dnasc):
    res = sb.table("pacientes").insert({
        "nome": nonempty(nome),
        "telefone": nonempty(tel),
        "cidade_bairro": nonempty(cid),
        "data_nascimento": iso(dnasc)
    }).execute()
    return res.data[0]["paciente_id"]

def update_paciente(pid, nome, tel, cid, dnasc):
    sb.table("pacientes").update({
        "nome": nonempty(nome),
        "telefone": nonempty(tel),
        "cidade_bairro": nonempty(cid),
        "data_nascimento": iso(dnasc)
    }).eq("paciente_id", pid).execute()

@st.cache_data(ttl=60)
def fetch_pacientes(limit=2000):
    res = sb.table("pacientes").select("*").order("created_at", desc=True).limit(limit).execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

def upsert_protocolo(nome, categoria):
    if not nome: return None
    got = sb.table("protocolos").select("*").eq("nome", nome).limit(1).execute()
    if got.data:
        pid = got.data[0]["protocolo_id"]
        if categoria and got.data[0].get("categoria") != categoria:
            sb.table("protocolos").update({"categoria": categoria}).eq("protocolo_id", pid).execute()
        return pid
    created = sb.table("protocolos").insert({"nome": nome, "categoria": categoria}).execute()
    return created.data[0]["protocolo_id"]

def create_atendimento(pid, protocolo_nome, categoria, status, dinicio, dprev, dreal,
                       dose_ini, dose_fin, tcle, medica, origem, obs):
    prot_id = upsert_protocolo(protocolo_nome, categoria) if protocolo_nome else None
    res = sb.table("atendimentos").insert({
        "paciente_id": pid,
        "protocolo_id": prot_id,
        "status": nonempty(status),
        "data_inicio": iso(dinicio),
        "data_termino_prevista": iso(dprev),
        "data_termino_real": iso(dreal),
        "dose_inicial_prescrita": nonempty(dose_ini),
        "dose_final_ajustada": nonempty(dose_fin),
        "tcle_assinado": bool(tcle) if tcle is not None else None,
        "medica": nonempty(medica),
        "origem": nonempty(origem),
        "observacoes": nonempty(obs)
    }).execute()
    return res.data[0]["atendimento_id"]

def upsert_pagamento(aid, forma, valor, desconto, custo, parc_prev, parc_quit, dt_ult, situacao):
    sb.table("pagamentos").insert({
        "atendimento_id": aid,
        "forma_pagamento": nonempty(forma),
        "valor": to_num(valor),
        "desconto": to_num(desconto),
        "custo_estimado": to_num(custo),
        "parcelas_previstas": to_int(parc_prev),
        "parcelas_quitadas": to_int(parc_quit),
        "data_ultimo_pagamento": iso(dt_ult),
        "situacao_financeira": nonempty(situacao)
    }).execute()

@st.cache_data(ttl=60)
def fetch_v_base(limit=10000):
    res = sb.table("v_base").select("*").limit(limit).execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

def fetch_atendimentos_paciente(pid, limit=100):
    res = (sb.table("v_base").select("*").eq("paciente_id", pid)
           .order("data_atendimento", desc=True).limit(limit).execute())
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

# =========================
# Modelagem (Análises)
# =========================
RETORNO_INATIVO_DIAS = 90

def rfm_scoring(df):
    cols_out = ["paciente_id","paciente_nome","ultima","freq","monet","recencia","R","F","M","risco_churn"]
    if df.empty or "data_atendimento" not in df:
        return pd.DataFrame(columns=cols_out)

    d = df.copy()
    d["data_atendimento"] = pd.to_datetime(d["data_atendimento"], errors="coerce")
    d["ticket_liquido"] = pd.to_numeric(d.get("ticket_liquido", 0), errors="coerce").fillna(0)
    d = d.dropna(subset=["data_atendimento"])
    if d.empty:
        return pd.DataFrame(columns=cols_out)

    today = pd.Timestamp.today().normalize()
    d12 = d[d["data_atendimento"] >= (today - pd.DateOffset(months=12))]

    grp = d12.groupby(["paciente_id","paciente_nome"], as_index=False).agg(
        ultima=("data_atendimento","max"),
        freq=("data_atendimento","count"),
        monet=("ticket_liquido","sum")
    )
    grp["recencia"] = (today - grp["ultima"]).dt.days

    grp["R"] = pd.cut(grp["recencia"], [-1,30,60,90,120,99999], labels=[5,4,3,2,1]).astype(int)
    grp["F"] = pd.cut(grp["freq"],     [-1,1,2,3,5,9999],        labels=[1,2,3,4,5]).astype(int)
    grp["M"] = pd.cut(grp["monet"],    [-1,200,500,1000,2000,9999999], labels=[1,2,3,4,5]).astype(int)

    grp["risco_churn"] = np.where(grp["recencia"] > RETORNO_INATIVO_DIAS, "alto",
                           np.where(grp["recencia"] > 60, "médio", "baixo"))
    return grp

def retention_cohort(df):
    if df.empty or "data_atendimento" not in df or "paciente_id" not in df:
        return pd.DataFrame()
    d = df.dropna(subset=["data_atendimento","paciente_id"]).copy()
    d["data_atendimento"] = pd.to_datetime(d["data_atendimento"], errors="coerce")
    first = d.groupby("paciente_id")["data_atendimento"].min().rename("first_date")
    d = d.join(first, on="paciente_id")
    d["cohort"] = d["first_date"].dt.to_period("M").astype(str)
    d["period"] = (d["data_atendimento"].dt.to_period("M") - d["first_date"].dt.to_period("M")).apply(lambda x: x.n).astype(int)
    pv = d.pivot_table(index="cohort", columns="period", values="paciente_id", aggfunc=pd.Series.nunique)
    base = pv.iloc[:, 0].replace(0, np.nan)
    return pv.divide(base, axis=0).fillna(0).round(3)

def protocol_performance(df):
    if df.empty or "protocolo" not in df:
        return pd.DataFrame()
    d = df.copy()
    d["ticket_liquido"] = pd.to_numeric(d.get("ticket_liquido", 0), errors="coerce").fillna(0)
    g = d.groupby("protocolo", as_index=False).agg(
        atendimentos=("atendimento_id","count"),
        receita=("ticket_liquido","sum"),
        ticket_medio=("ticket_liquido","mean")
    )
    return g.sort_values("receita", ascending=False)

def upsell_opportunities(df, rfm):
    cols = ["paciente_id","paciente_nome","risco_churn","sugestao_proximo_protocolo"]
    if df.empty or rfm.empty: return pd.DataFrame(columns=cols)

    d = df.copy()
    d["data_atendimento"] = pd.to_datetime(d["data_atendimento"], errors="coerce")
    d["ticket_liquido"] = pd.to_numeric(d.get("ticket_liquido", 0), errors="coerce").fillna(0)
    d = d.dropna(subset=["data_atendimento"])
    if d.empty: return pd.DataFrame(columns=cols)

    last = (d.groupby(["paciente_id","paciente_nome"])["data_atendimento"]
              .max().reset_index().rename(columns={"data_atendimento":"ultima"}))

    base = rfm.merge(last, on=["paciente_id","paciente_nome"], how="left")
    if "ultima" not in base.columns: base["ultima"] = pd.NaT
    base["dias_ult"] = (pd.Timestamp.today().normalize() - base["ultima"]).dt.days
    base["dias_ult"] = base["dias_ult"].fillna(9999)

    # candidatos: F>=2, M entre 2 e 4, 31–120 dias desde a última
    cand = base[(base["F"] >= 2) & (base["M"].between(2,4)) & (base["dias_ult"].between(31,120))]

    perf = protocol_performance(d)
    sugestao = perf["protocolo"].iloc[0] if not perf.empty else None

    out = cand[["paciente_id","paciente_nome","risco_churn"]].copy()
    out["sugestao_proximo_protocolo"] = sugestao
    return out.reindex(columns=cols)

# =========================
# Agente Comercial
# =========================
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
    txt = resp.choices[0].message.content.strip()
    plano = {"objetivos":[], "perguntas":[], "acoes":[], "metricas_chave":[]}
    try:
        plano = json.loads(txt)
    except:
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
        try:
            if tool == "rfm_scoring":
                saida = rfm_scoring(df)
            elif tool == "cohort_retencao":
                saida = retention_cohort(df)
            elif tool == "protocol_performance":
                saida = protocol_performance(df)
            elif tool == "pacientes_inativos":
                dias = int(params.get("dias_inativos", 90))
                tmp = df.copy()
                tmp["data_atendimento"] = pd.to_datetime(tmp["data_atendimento"], errors="coerce")
                last = (tmp.dropna(subset=["data_atendimento"])
                          .groupby(["paciente_id","paciente_nome"])["data_atendimento"]
                          .max().reset_index(name="ultima"))
                last["dias_sem_visita"] = (pd.Timestamp.today().normalize() - last["ultima"]).dt.days
                saida = last[last["dias_sem_visita"] > dias].sort_values("dias_sem_visita", ascending=False)
            elif tool == "upsell_opportunities":
                base_rfm = rfm_scoring(df)
                saida = upsell_opportunities(df, base_rfm)
            else:
                saida = pd.DataFrame()
        except Exception as e:
            saida = pd.DataFrame()
            err_csv = f"erro,{str(e)}\n"
            resultados.append({"tool": tool, "params": params, "tamanho": 0, "amostra_csv": err_csv})
            continue

        resultados.append({
            "tool": tool,
            "params": params,
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

# =========================
# UI
# =========================
st.title("Gestor Comercial da Clínica")

aba1, aba2, aba3 = st.tabs(["Recepção - Pacientes", "Consultório - Atendimentos", "Gestão - Análises"])

# -------- Aba 1: Recepção (Cadastro/Busca/Edição de Pacientes)
with aba1:
    st.subheader("Cadastro de Pacientes")
    col1, col2 = st.columns(2)
    with col1:
        nome_new = st.text_input("Nome completo")
        tel_new = st.text_input("Telefone / WhatsApp")
    with col2:
        cid_new = st.text_input("Cidade / Bairro")
        dnasc_new = st.date_input("Data de nascimento", value=None)
    if st.button("Salvar paciente"):
        if not nome_new:
            st.error("Informe o nome.")
        else:
            insert_paciente(nome_new, tel_new, cid_new, dnasc_new)
            st.success("Paciente cadastrado!")
            fetch_pacientes.clear()

    st.markdown("### Lista de Pacientes")
    busca = st.text_input("Buscar por nome ou telefone", placeholder="Ex: Ana, 3199..., João")
    pacientes = fetch_pacientes()
    if not pacientes.empty:
        # busca dinâmica (nome/telefone)
        if busca:
            mask = pacientes["nome"].str.contains(busca, case=False, na=False) | pacientes["telefone"].str.contains(busca, case=False, na=False)
            pacientes = pacientes[mask]
        st.dataframe(pacientes[["nome","telefone","cidade_bairro","data_nascimento"]], use_container_width=True)

        # edição rápida (opcional)
        st.markdown("#### Editar paciente (opcional)")
        if not pacientes.empty:
            nomes = pacientes["nome"].tolist()
            edit_nome = st.selectbox("Selecione para editar", [""] + nomes)
            if edit_nome:
                row = pacientes[pacientes["nome"]==edit_nome].iloc[0]
                colA, colB = st.columns(2)
                with colA:
                    novo_nome = st.text_input("Nome", value=row["nome"])
                    novo_tel = st.text_input("Telefone", value=row["telefone"] or "")
                with colB:
                    novo_cid = st.text_input("Cidade/Bairro", value=row["cidade_bairro"] or "")
                    novo_dn = st.date_input("Data nascimento", value=pd.to_datetime(row["data_nascimento"]).date() if row["data_nascimento"] else None)
                if st.button("Salvar alterações"):
                    update_paciente(row["paciente_id"], novo_nome, novo_tel, novo_cid, novo_dn)
                    st.success("Paciente atualizado.")
                    fetch_pacientes.clear()

# -------- Aba 2: Consultório (Busca ultra-rápida + cadastro de atendimento)
with aba2:
    st.subheader("Cadastro de Atendimentos")
    pacientes = fetch_pacientes()
    if pacientes.empty:
        st.warning("Nenhum paciente cadastrado ainda.")
    else:
        # Busca digitável e lista reduzida no select
        busca2 = st.text_input("Buscar paciente por nome/telefone", placeholder="Digite para filtrar rapidamente…")
        filtrados = pacientes
        if busca2:
            mask = pacientes["nome"].str.contains(busca2, case=False, na=False) | pacientes["telefone"].str.contains(busca2, case=False, na=False)
            filtrados = pacientes[mask]
        # Ordena por nome para navegação mais fácil
        filtrados = filtrados.sort_values("nome")
        nomes_combo = filtrados["nome"].tolist()
        escolhido = st.selectbox("Selecione o paciente", nomes_combo, index=0 if len(nomes_combo)>0 else None)

        if escolhido:
            pid = filtrados.loc[filtrados["nome"]==escolhido,"paciente_id"].values[0]

            # Mostra últimos atendimentos do paciente para contexto
            hist = fetch_atendimentos_paciente(pid, limit=5)
            if not hist.empty:
                with st.expander("Últimos atendimentos do paciente (5 mais recentes)"):
                    st.dataframe(hist[["data_atendimento","protocolo","status","forma_pagamento","ticket_liquido","situacao_financeira"]], use_container_width=True)

            # Form de atendimento
            st.markdown("### Novo atendimento")
            col1, col2, col3 = st.columns(3)
            with col1:
                protocolo_nome = st.text_input("Protocolo (ex: Semaglutida)")
                categoria = st.selectbox("Categoria", ["Emagrecimento","Estética","Outros"])
                status = st.selectbox("Status", ["Em curso","Concluído","Cancelado"])
            with col2:
                medica = st.text_input("Médica")
                origem = st.text_input("Origem (opcional)")
                tcle_assinado = st.checkbox("TCLE assinado?")
            with col3:
                dose_ini = st.text_input("Dose inicial prescrita")
                dose_fin = st.text_input("Dose final ajustada")
            col4, col5, col6 = st.columns(3)
            with col4:
                data_inicio = st.date_input("Data de início", value=date.today())
            with col5:
                data_prev = st.date_input("Data prevista de término", value=None)
            with col6:
                data_real = st.date_input("Data de término real", value=None)
            col7, col8, col9 = st.columns(3)
            with col7:
                forma = st.text_input("Forma de pagamento", value="Parcelado")
                valor = st.text_input("Valor (ex: 1500.00)")
            with col8:
                desconto = st.text_input("Desconto (ex: 0.00)")
                custo = st.text_input("Custo estimado")
            with col9:
                parc_prev = st.text_input("Parcelas previstas")
                parc_quit = st.text_input("Parcelas quitadas")
                dt_ult = st.date_input("Data do último pagamento", value=None)
            obs = st.text_area("Observações")

            if st.button("Salvar atendimento"):
                aid = create_atendimento(pid, protocolo_nome, categoria, status, data_inicio, data_prev, data_real,
                                         dose_ini, dose_fin, tcle_assinado, medica, origem, obs)
                upsert_pagamento(aid, forma, valor, desconto, custo, parc_prev, parc_quit, dt_ult, situacao="Em dia")
                fetch_v_base.clear()
                st.success("Atendimento cadastrado!")

# -------- Aba 3: Gestão (KPIs, Dash e Agente)
with aba3:
    st.subheader("Visão consolidada")
    df = fetch_v_base()
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        # KPIs
        df["ticket_liquido"] = pd.to_numeric(df["ticket_liquido"], errors="coerce").fillna(0)
        col1, col2, col3, col4 = st.columns(4)
        receita_total = df["ticket_liquido"].sum()
        with col1: st.metric("Receita total", f"R$ {receita_total:,.2f}")
        with col2: st.metric("Atendimentos", len(df))
        with col3: st.metric("Pacientes únicos", df["paciente_id"].nunique())
        with col4:
            tm = receita_total/len(df) if len(df) else 0
            st.metric("Ticket médio", f"R$ {tm:,.2f}")

        # Protocolos
        if "protocolo" in df.columns:
            top = df.groupby("protocolo", as_index=False)["ticket_liquido"].sum().sort_values("ticket_liquido", ascending=False)
            st.markdown("### Top protocolos por receita")
            fig = px.bar(top.head(10), x="protocolo", y="ticket_liquido")
            st.plotly_chart(fig, use_container_width=True)

        # RFM e riscos
        st.markdown("### RFM e Risco de Churn")
        rfm = rfm_scoring(df)
        if not rfm.empty:
            st.dataframe(rfm.sort_values(["risco_churn","R","F","M"], ascending=[True, False, False, False]), use_container_width=True)
            risco = rfm["risco_churn"].value_counts().reset_index()
            risco.columns = ["risco","qtd"]
            fig2 = px.bar(risco, x="risco", y="qtd")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Ainda não há dados suficientes para RFM.")

        # Cohort
        st.markdown("### Cohort de retenção")
        coh = retention_cohort(df)
        if not coh.empty:
            st.dataframe((coh*100).round(1), use_container_width=True)
        else:
            st.info("Ainda não há dados suficientes para análise de cohort.")

        # Upsell
        st.markdown("### Oportunidades de Upsell")
        ups = upsell_opportunities(df, rfm) if not rfm.empty else pd.DataFrame()
        if not ups.empty:
            st.dataframe(ups, use_container_width=True)
        else:
            st.info("Sem oportunidades de upsell com as regras atuais.")

    # Agente Comercial
    st.markdown("## Agente Comercial")
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
            st.markdown("### Relatório Executivo")
            st.write(rel)
            # Downloads das evidências
            for r in resultados:
                if r["amostra_csv"]:
                    st.download_button(f"Baixar {r['tool']}.csv", r["amostra_csv"], file_name=f"{r['tool']}.csv", mime="text/csv")
