import os, io, json
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from supabase import create_client, Client
import plotly.express as px
import uuid


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
# FORMULÁRIO INTELIGENTE – MODO MÉDICA
# =====================================

st.markdown("## 🩺 Registrar Protocolados")

# Carrega base para autocomplete
@st.cache_data(ttl=30)
def fetch_pacientes_base():
    res = sb.table("v_base").select("paciente_nome, telefone, cidade_bairro, medica, origem").execute()
    df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
    df = df.dropna(subset=["paciente_nome"]).drop_duplicates(subset=["paciente_nome"])
    return df

pacientes_df = fetch_pacientes_base()
nomes_existentes = sorted(pacientes_df["paciente_nome"].unique().tolist()) if not pacientes_df.empty else []

# Estado local para autocompletar campos
if "dados_paciente" not in st.session_state:
    st.session_state.dados_paciente = {}

def preencher_campos(nome):
    """Busca dados padrão do paciente selecionado"""
    dados = pacientes_df[pacientes_df["paciente_nome"] == nome]
    if not dados.empty:
        row = dados.iloc[0]
        st.session_state.dados_paciente = {
            "telefone": row.get("telefone", ""),
            "cidade_bairro": row.get("cidade_bairro", ""),
            "medica": row.get("medica", ""),
            "origem": row.get("origem", "")
        }
    else:
        st.session_state.dados_paciente = {}

with st.form("form_vbase_simplificado"):
    st.caption("Preencha os campos principais do atendimento atual.")

    col1, col2 = st.columns(2)

    # --- COLUNA 1: Dados principais ---
    with col1:
        paciente_nome = st.selectbox(
            "Nome do paciente",
            options=[""] + nomes_existentes,
            index=0,
            key="sel_paciente",
            help="Selecione um paciente existente ou digite um novo nome."
        )
        # Se selecionou um nome existente, preencher automaticamente
        if paciente_nome:
            preencher_campos(paciente_nome)

        telefone = st.text_input(
            "Telefone / WhatsApp",
            value=st.session_state.dados_paciente.get("telefone", ""),
            placeholder="Ex: (31) 99999-9999"
        )
        protocolo = st.text_input("Protocolo", placeholder="Ex: Semaglutida semanal")
        categoria = st.selectbox("Categoria", ["Emagrecimento", "Estética", "Outros"], index=0)
        medica = st.text_input(
            "Médica responsável",
            value=st.session_state.dados_paciente.get("medica", ""),
            placeholder="Ex: Dra. Mariana"
        )

    # --- COLUNA 2: Dados secundários ---
    with col2:
        data_atendimento = st.date_input("Data do atendimento", value=date.today())
        status = st.selectbox("Status", ["Em curso", "Concluído", "Cancelado"], index=0)
        tcle_assinado = st.checkbox("TCLE assinado?", value=True)
        origem = st.text_input(
            "Origem",
            value=st.session_state.dados_paciente.get("origem", ""),
            placeholder="Ex: Indicação, Instagram, Google"
        )
        cidade_bairro = st.text_input(
            "Cidade / Bairro",
            value=st.session_state.dados_paciente.get("cidade_bairro", ""),
            placeholder="Ex: Belo Horizonte / Lourdes"
        )

    # --- Detalhes do protocolo ---
    st.markdown("### 💊 Detalhes do protocolo")
    col3, col4 = st.columns(2)
    with col3:
        dose_inicial_prescrita = st.text_input("Dose inicial prescrita", placeholder="Ex: 0.25 mg")
        dose_final_ajustada = st.text_input("Dose final ajustada", placeholder="Ex: 1 mg")
    with col4:
        data_termino_prevista = st.date_input("Previsão de término", value=None)
        data_termino_real = st.date_input("Término real (se já finalizado)", value=None)

    observacoes = st.text_area("Observações", placeholder="Observações clínicas, efeitos relatados, evolução...")

    # --- Pagamento (expansível) ---
    with st.expander("💰 Dados de pagamento (opcional)", expanded=False):
        col5, col6, col7 = st.columns(3)
        with col5:
            forma_pagamento = st.selectbox("Forma de pagamento", ["Pix", "Cartão", "Dinheiro", "Outro"], index=0)
            valor = st.number_input("Valor total (R$)", min_value=0.0, step=10.0)
        with col6:
            desconto = st.number_input("Desconto (R$)", min_value=0.0, step=10.0)
            custo_estimado = st.number_input("Custo estimado (R$)", min_value=0.0, step=10.0)
        with col7:
            parcelas_previstas = st.number_input("Parcelas previstas", min_value=0, step=1)
            parcelas_quitadas = st.number_input("Parcelas quitadas", min_value=0, step=1)
            data_ultimo_pagamento = st.date_input("Último pagamento", value=None)
        situacao_financeira = st.selectbox("Situação financeira", ["Em dia", "Em aberto", "Atrasado"], index=0)

    enviar = st.form_submit_button("✅ Salvar atendimento")

if enviar:
    if not paciente_nome or not protocolo:
        st.error("Por favor, preencha pelo menos o nome do paciente e o protocolo.")
    else:
        try:
            ticket_liquido = (valor or 0) - (desconto or 0)
            sb.table("v_base").insert({
                "atendimento_id": str(uuid.uuid4()),
                "paciente_id": str(uuid.uuid4()),
                "paciente_nome": paciente_nome,
                "telefone": telefone,
                "cidade_bairro": cidade_bairro,
                "protocolo": protocolo,
                "categoria": categoria,
                "status": status,
                "data_atendimento": data_atendimento.isoformat() if data_atendimento else None,
                "data_termino_prevista": data_termino_prevista.isoformat() if data_termino_prevista else None,
                "data_termino_real": data_termino_real.isoformat() if data_termino_real else None,
                "dose_inicial_prescrita": dose_inicial_prescrita,
                "dose_final_ajustada": dose_final_ajustada,
                "tcle_assinado": bool(tcle_assinado),
                "medica": medica,
                "origem": origem,
                "observacoes": observacoes,
                "forma_pagamento": forma_pagamento,
                "ticket_liquido": float(ticket_liquido),
                "valor": float(valor or 0),
                "desconto": float(desconto or 0),
                "custo_estimado": float(custo_estimado or 0),
                "parcelas_previstas": int(parcelas_previstas or 0),
                "parcelas_quitadas": int(parcelas_quitadas or 0),
                "data_ultimo_pagamento": data_ultimo_pagamento.isoformat() if data_ultimo_pagamento else None,
                "situacao_financeira": situacao_financeira,
                "created_at": datetime.now().isoformat()
            }).execute()

            # Atualiza tudo automaticamente
            fetch_v_base.clear()
            fetch_pacientes_base.clear()
            st.success(f"✅ Atendimento de {paciente_nome} salvo com sucesso!")

            # Força recarregamento da página para atualizar métricas e gráficos
            st.experimental_rerun()

        except Exception as e:
            st.error(f"Erro ao salvar: {e}")


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
    if df.empty or "protocolo" not in df:
        return pd.DataFrame()
    d = df.copy()
    d["ticket_liquido"] = pd.to_numeric(d["ticket_liquido"], errors="coerce").fillna(0)
    g = d.groupby("protocolo", as_index=False).agg({
        "atendimento_id": "count",
        "ticket_liquido": ["sum", "mean"]
    })
    g.columns = ["Protocolo", "Atendimentos", "Receita", "Ticket médio (R$)"]
    return g.sort_values("Receita", ascending=False)


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
    fig = px.bar(
        prot.head(10),
        x="Protocolo",
        y="Receita",
        color="Protocolo",
        title="Protocolos mais lucrativos"
    )
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
