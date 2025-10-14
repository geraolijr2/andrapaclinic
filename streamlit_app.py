import os, io, json
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from supabase import create_client, Client
import plotly.express as px

# =====================================
# CONFIGURAÇÃO GERAL
# =====================================
st.set_page_config(page_title="AndrapaSmart – Controle de Protocolos", layout="wide")

SUPABASE_URL = st.secrets["supabase_url"]
SUPABASE_KEY = st.secrets["supabase_anon_key"]
ADMIN_PASS = st.secrets.get("admin_pass", "")
OPENAI_KEY = st.secrets.get("openai_api_key")
OPENAI_MODEL = st.secrets.get("openai_model", "gpt-4o-mini")

@st.cache_resource
def get_sb() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)
sb = get_sb()

# =====================================
# AUTENTICAÇÃO
# =====================================
def auth_gate():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if not st.session_state.auth_ok:
        st.title("Acesso Restrito")
        pw = st.text_input("Digite a senha de acesso", type="password")
        if st.button("Entrar"):
            if pw == ADMIN_PASS and pw:
                st.session_state.auth_ok = True
            else:
                st.error("Senha incorreta. Tente novamente.")
        st.stop()
auth_gate()

# =====================================
# SUPABASE – BASE PRINCIPAL
# =====================================
@st.cache_data(ttl=30)
def fetch_v_base(limit=10000):
    res = sb.table("v_base").select("*").limit(limit).execute()
    df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
    expected = [
        "atendimento_id", "paciente_id", "paciente_nome",
        "data_atendimento", "protocolo", "status",
        "ticket_liquido", "situacao_financeira"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = None
    return df[expected]

# =====================================
# ANÁLISES
# =====================================
RETORNO_INATIVO_DIAS = 90

def rfm_analise(df):
    if df.empty: return pd.DataFrame()
    d = df.copy()
    d["data_atendimento"] = pd.to_datetime(d["data_atendimento"], errors="coerce")
    d["ticket_liquido"] = pd.to_numeric(d.get("ticket_liquido", 0), errors="coerce").fillna(0)
    d = d.dropna(subset=["data_atendimento"])
    if d.empty: return pd.DataFrame()
    hoje = pd.Timestamp.today().normalize()
    ult_ano = d[d["data_atendimento"] >= (hoje - pd.DateOffset(months=12))]
    grp = ult_ano.groupby(["paciente_id","paciente_nome"], as_index=False).agg(
        ultima=("data_atendimento","max"),
        vezes=("data_atendimento","count"),
        total=("ticket_liquido","sum")
    )
    grp["dias_desde_ultima"] = (hoje - grp["ultima"]).dt.days
    grp["tempo_sem_retorno"] = np.select(
        [
            grp["dias_desde_ultima"] <= 30,
            grp["dias_desde_ultima"] <= 60,
            grp["dias_desde_ultima"] <= 90
        ],
        ["baixo", "médio", "alto"],
        default="muito alto"
    )
    grp = grp.rename(columns={
        "paciente_nome": "Paciente",
        "vezes": "Atendimentos",
        "total": "Total gasto (R$)",
        "dias_desde_ultima": "Dias desde o último atendimento",
        "tempo_sem_retorno": "Tempo sem retorno"
    })
    return grp

def desempenho_protocolos(df):
    if df.empty or "protocolo" not in df: return pd.DataFrame()
    d = df.copy()
    d["ticket_liquido"] = pd.to_numeric(d["ticket_liquido"], errors="coerce").fillna(0)
    g = d.groupby("protocolo", as_index=False).agg(
        Atendimentos=("atendimento_id","count"),
        Receita=("ticket_liquido","sum"),
        "Ticket médio (R$)"=("ticket_liquido","mean")
    ).sort_values("Receita", ascending=False)
    return g

def oportunidades_retorno(df, rfm_df):
    if df.empty or rfm_df.empty: return pd.DataFrame()
    d = rfm_df[rfm_df["Tempo sem retorno"].isin(["alto","muito alto"])].copy()
    d = d.sort_values("Dias desde o último atendimento", ascending=False)
    return d

# =====================================
# GPT - RELATÓRIO INTELIGENTE
# =====================================
def gerar_relatorio(df, resumo, oportunidades):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)

    resumo_txt = f"""
    Total de atendimentos: {resumo['total_atendimentos']}
    Total de pacientes: {resumo['pacientes']}
    Receita total: R$ {resumo['receita_total']:,.2f}
    Ticket médio: R$ {resumo['ticket_medio']:,.2f}
    """

    context = (
        "Você é um consultor comercial experiente. Analise os dados de uma clínica de estética e "
        "gere um relatório em linguagem simples, dividido em 5 seções: "
        "1) Resumo geral, 2) Resultados recentes, 3) Protocolos de destaque, "
        "4) Pacientes que precisam de contato, 5) Próximas ações recomendadas."
    )

    evidencias = oportunidades.head(15).to_dict(orient="records") if not oportunidades.empty else []

    messages = [
        {"role":"system", "content": context},
        {"role":"user", "content": f"Resumo:\n{resumo_txt}\n\nDados:\n{json.dumps(evidencias, ensure_ascii=False)}"}
    ]

    resp = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=0.3, max_tokens=900, messages=messages
    )
    return resp.choices[0].message.content.strip()

# =====================================
# INTERFACE PRINCIPAL
# =====================================
st.title("📊 Controle de Protocolos")
st.caption("Visualize os resultados e descubra oportunidades de crescimento.")

df = fetch_v_base()
if df.empty:
    st.info("Nenhum dado encontrado ainda.")
    st.stop()

df["ticket_liquido"] = pd.to_numeric(df["ticket_liquido"], errors="coerce").fillna(0)
df["data_atendimento"] = pd.to_datetime(df["data_atendimento"], errors="coerce")

# --- Indicadores gerais
col1, col2, col3, col4 = st.columns(4)
receita_total = df["ticket_liquido"].sum()
total_atendimentos = len(df)
pacientes_unicos = df["paciente_id"].nunique()
ticket_medio = receita_total / total_atendimentos if total_atendimentos else 0

col1.metric("💰 Receita total", f"R$ {receita_total:,.2f}")
col2.metric("📅 Atendimentos", total_atendimentos)
col3.metric("👥 Pacientes únicos", pacientes_unicos)
col4.metric("💵 Ticket médio", f"R$ {ticket_medio:,.2f}")

# --- Tabela principal
st.markdown("### 📋 Lista de protocolos realizados")
st.dataframe(df, use_container_width=True, height=350)

# --- Gráficos
st.markdown("### 📈 Protocolos com maior faturamento")
prot = desempenho_protocolos(df)
if not prot.empty:
    fig = px.bar(prot.head(10), x="protocolo", y="Receita", color="protocolo", title="Protocolos mais lucrativos")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Ainda não há dados suficientes para este gráfico.")

# --- Análise de pacientes e retorno
st.markdown("### 🔍 Pacientes e frequência de retorno")
rfm_df = rfm_analise(df)
if not rfm_df.empty:
    st.dataframe(rfm_df, use_container_width=True)
else:
    st.info("Ainda não há dados suficientes para análise de pacientes.")

# --- Oportunidades de reengajamento
st.markdown("### 📞 Pacientes que estão há muito tempo sem retornar")
op = oportunidades_retorno(df, rfm_df)
if not op.empty:
    st.dataframe(op[["Paciente", "Dias desde o último atendimento", "Total gasto (R$)"]], use_container_width=True)
else:
    st.info("Nenhum paciente inativo identificado até o momento.")

# --- Relatório Inteligente
st.markdown("## 🤖 Relatório Inteligente")
st.caption("Gere um resumo automático com sugestões de ação.")
if st.button("Gerar Relatório"):
    resumo = {
        "receita_total": receita_total,
        "total_atendimentos": total_atendimentos,
        "pacientes": pacientes_unicos,
        "ticket_medio": ticket_medio
    }
    with st.spinner("Gerando relatório com inteligência artificial..."):
        rel = gerar_relatorio(df, resumo, op)
    st.markdown("### 📋 Relatório Executivo")
    st.write(rel)
