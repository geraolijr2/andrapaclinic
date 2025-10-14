import os, io, json, uuid
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

@st.cache_resource
def get_sb() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)
sb = get_sb()


# =====================================
# SUPABASE – FUNÇÕES DE ACESSO
# =====================================
@st.cache_data(ttl=30)
def fetch_pacientes_base():
    res = sb.table("pacientes").select("paciente_id, nome, telefone, cidade_bairro").execute()
    df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
    if not df.empty:
        df = df.dropna(subset=["nome"]).drop_duplicates(subset=["nome"])
    return df


@st.cache_data(ttl=30)
def fetch_protocolos_base():
    res = sb.table("protocolos").select("protocolo_id, nome, categoria").execute()
    df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
    if not df.empty:
        df = df.dropna(subset=["nome"]).drop_duplicates(subset=["nome"])
    return df


@st.cache_data(ttl=30)
def fetch_v_base(limit=10000):
    try:
        res = sb.table("v_base").select("*").limit(limit).execute()

        if not res.data or len(res.data) == 0:
            st.info("⚠️ Nenhum registro encontrado em v_base.")
            return pd.DataFrame()

        # Corrige UUIDs antes de criar o DataFrame
        rows = []
        for r in res.data:
            # Força conversão de UUIDs para texto
            for key in ["paciente_id", "atendimento_id"]:
                if key in r and r[key] is not None:
                    r[key] = str(r[key])
                elif key not in r:
                    # se o Supabase não trouxe o campo, cria vazio
                    r[key] = None
            rows.append(r)

        df = pd.DataFrame(rows)

        # Normaliza nomes de colunas
        df.columns = (
            df.columns
            .str.encode("ascii", "ignore")
            .str.decode("utf-8")
            .str.strip()
            .str.replace("\u00a0", "")
            .str.replace("\ufeff", "")
            .str.lower()
        )

        # Força presença dos campos esperados
        expected_cols = [
            "atendimento_id", "paciente_id", "paciente_nome", "protocolo",
            "data_atendimento", "ticket_liquido", "status", "situacao_financeira"
        ]
        for c in expected_cols:
            if c not in df.columns:
                df[c] = None



        # Garante tipos corretos
        if "data_atendimento" in df.columns:
            df["data_atendimento"] = pd.to_datetime(df["data_atendimento"], errors="coerce")
        if "ticket_liquido" in df.columns:
            df["ticket_liquido"] = pd.to_numeric(df["ticket_liquido"], errors="coerce").fillna(0)

        return df

    except Exception as e:
        st.error(f"❌ Erro ao carregar v_base: {e}")
        return pd.DataFrame()


# =====================================
# ANÁLISES
# =====================================
def rfm_analise(df):
    if df.empty:
        return pd.DataFrame()

    df.columns = df.columns.str.lower()
    cols_needed = {"paciente_id", "paciente_nome", "data_atendimento", "ticket_liquido"}
    if not cols_needed.issubset(df.columns):
        st.warning(f"⚠️ Faltam colunas em v_base: {cols_needed - set(df.columns)}")
        return pd.DataFrame()

    d = df.dropna(subset=["paciente_nome", "data_atendimento"]).copy()
    hoje = pd.Timestamp.today().normalize()
    ult_ano = d[d["data_atendimento"] >= (hoje - pd.DateOffset(months=12))]

    grp = ult_ano.groupby(["paciente_id", "paciente_nome"], as_index=False).agg(
        ultima=("data_atendimento", "max"),
        vezes=("data_atendimento", "count"),
        total=("ticket_liquido", "sum")
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
    if df.empty:
        return pd.DataFrame()

    df.columns = df.columns.str.lower()
    if "protocolo" not in df.columns or "ticket_liquido" not in df.columns:
        st.warning(f"⚠️ Colunas esperadas não encontradas em v_base.")
        st.write("Colunas atuais:", df.columns.tolist())
        return pd.DataFrame()

    d = df.copy()
    d["ticket_liquido"] = pd.to_numeric(d["ticket_liquido"], errors="coerce").fillna(0)

    try:
        g = d.groupby("protocolo", as_index=False).agg(
            atendimentos=("protocolo", "count"),
            receita=("ticket_liquido", "sum"),
            ticket_medio=("ticket_liquido", "mean")
        )
        g = g.rename(columns={
            "protocolo": "Protocolo",
            "atendimentos": "Atendimentos",
            "receita": "Receita",
            "ticket_medio": "Ticket médio (R$)"
        })
        g = g.sort_values("Receita", ascending=False)
        return g
    except Exception as e:
        st.error(f"Erro ao agrupar protocolos: {e}")
        return pd.DataFrame()


def oportunidades_retorno(df, rfm_df):
    if df.empty or rfm_df.empty:
        return pd.DataFrame()
    d = rfm_df[rfm_df["Tempo sem retorno"].isin(["alto", "muito alto"])].copy()
    d = d.sort_values("Dias desde o último atendimento", ascending=False)
    return d


# =====================================
# FORMULÁRIO INTELIGENTE – MODO MÉDICA
# =====================================
st.markdown("## 🩺 Registrar Protocolados")

pacientes_df = fetch_pacientes_base()
protocolos_df = fetch_protocolos_base()

nomes_pacientes = sorted(pacientes_df["nome"].unique().tolist()) if not pacientes_df.empty else []
nomes_protocolos = sorted(protocolos_df["nome"].unique().tolist()) if not protocolos_df.empty else []

if "form_data" not in st.session_state:
    st.session_state.form_data = {}

paciente_sel = st.selectbox(
    "Nome do paciente",
    options=[""] + nomes_pacientes,
    index=0,
    key="paciente_select"
)

if paciente_sel:
    dados = pacientes_df[pacientes_df["nome"] == paciente_sel]
    if not dados.empty:
        row = dados.iloc[0]
        st.session_state.form_data = {
            "paciente_nome": paciente_sel,
            "telefone": row.get("telefone", ""),
            "cidade_bairro": row.get("cidade_bairro", "")
        }
else:
    st.session_state.form_data = {}

with st.form("form_vbase_simplificado"):
    st.caption("Preencha os campos principais do atendimento atual.")
    col1, col2 = st.columns(2)

    with col1:
        paciente_nome = st.text_input(
            "Nome completo",
            value=st.session_state.form_data.get("paciente_nome", ""),
            placeholder="Ex: Ana Souza"
        )
        telefone = st.text_input(
            "Telefone / WhatsApp",
            value=st.session_state.form_data.get("telefone", ""),
            placeholder="Ex: (31) 99999-9999"
        )
        cidade_bairro = st.text_input(
            "Cidade / Bairro",
            value=st.session_state.form_data.get("cidade_bairro", ""),
            placeholder=""
        )
        protocolo = st.selectbox("Protocolo", options=[""] + nomes_protocolos, index=0)
        if protocolo == "":
            protocolo = st.text_input("Novo protocolo", placeholder="Ex: Semaglutida semanal")

        categoria = st.selectbox("Categoria", ["Emagrecimento", "Estética", "Outros"], index=0)
        medica = st.text_input("Médica responsável", placeholder="")

    with col2:
        data_atendimento = st.date_input("Data do atendimento", value=date.today())
        status = st.selectbox("Status", ["Em curso", "Concluído", "Cancelado"], index=0)
        tcle_assinado = st.checkbox("TCLE assinado?", value=True)
        origem = st.text_input("Origem", placeholder="Ex: Indicação, Instagram, Google")
        dose_inicial_prescrita = st.text_input("Dose inicial prescrita", placeholder="Ex: 0.25 mg")
        dose_final_ajustada = st.text_input("Dose final ajustada", placeholder="Ex: 1 mg")
        data_termino_prevista = st.date_input("Previsão de término", value=None)
        data_termino_real = st.date_input("Término real (se já finalizado)", value=None)

    observacoes = st.text_area("Observações", placeholder="Observações clínicas, evolução...")

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

# =====================================
# SALVAMENTO
# =====================================
if enviar:
    if not paciente_nome or not protocolo:
        st.error("Por favor, preencha pelo menos o nome do paciente e o protocolo.")
    else:
        try:
            paciente_q = sb.table("pacientes").select("paciente_id").ilike("nome", paciente_nome.strip()).execute()
            if paciente_q.data:
                paciente_id = paciente_q.data[0]["paciente_id"]
            else:
                novo = sb.table("pacientes").insert({
                    "paciente_id": str(uuid.uuid4()),
                    "nome": paciente_nome.strip(),
                    "telefone": telefone.strip() if telefone else None,
                    "cidade_bairro": cidade_bairro.strip() if cidade_bairro else None,
                    "created_at": datetime.now().isoformat()
                }).execute()
                paciente_id = novo.data[0]["paciente_id"]

            protocolo_q = sb.table("protocolos").select("protocolo_id").ilike("nome", protocolo.strip()).execute()
            if protocolo_q.data:
                protocolo_id = protocolo_q.data[0]["protocolo_id"]
            else:
                novo_p = sb.table("protocolos").insert({
                    "protocolo_id": str(uuid.uuid4()),
                    "nome": protocolo.strip(),
                    "categoria": categoria.strip(),
                    "ativo": True
                }).execute()
                protocolo_id = novo_p.data[0]["protocolo_id"]

            atendimento_id = str(uuid.uuid4())
            sb.table("atendimentos").insert({
                "atendimento_id": atendimento_id,
                "paciente_id": paciente_id,
                "protocolo_id": protocolo_id,
                "status": status,
                "data_inicio": data_atendimento.isoformat(),
                "data_termino_prevista": data_termino_prevista.isoformat() if data_termino_prevista else None,
                "data_termino_real": data_termino_real.isoformat() if data_termino_real else None,
                "dose_inicial_prescrita": dose_inicial_prescrita,
                "dose_final_ajustada": dose_final_ajustada,
                "tcle_assinado": bool(tcle_assinado),
                "medica": medica,
                "origem": origem,
                "observacoes": observacoes,
                "created_at": datetime.now().isoformat()
            }).execute()

            if valor or desconto or custo_estimado:
                sb.table("pagamentos").insert({
                    "pagamento_id": str(uuid.uuid4()),
                    "atendimento_id": atendimento_id,
                    "forma_pagamento": forma_pagamento,
                    "valor": float(valor or 0),
                    "desconto": float(desconto or 0),
                    "custo_estimado": float(custo_estimado or 0),
                    "parcelas_previstas": int(parcelas_previstas or 0),
                    "parcelas_quitadas": int(parcelas_quitadas or 0),
                    "data_ultimo_pagamento": data_ultimo_pagamento.isoformat() if data_ultimo_pagamento else None,
                    "situacao_financeira": situacao_financeira,
                    "created_at": datetime.now().isoformat()
                }).execute()

            # Atualiza dados e limpa apenas variáveis auxiliares
            fetch_v_base.clear()
            fetch_pacientes_base.clear()
            fetch_protocolos_base.clear()
            st.session_state.form_data = {}

            st.success(f"✅ Atendimento de {paciente_nome} salvo com sucesso!")
            st.rerun()

        except Exception as e:
            st.error(f"Erro ao salvar: {e}")



# =====================================
# DASHBOARD
# =====================================
st.title("📊 Controle de Protocolos")
st.caption("Visualize os resultados e descubra oportunidades de crescimento.")

df = fetch_v_base()
if df.empty:
    st.info("Nenhum dado encontrado ainda.")
    st.stop()

cols_visiveis = [c for c in df.columns if not c.endswith("_id") and c not in ["created_at"]]
df = df[cols_visiveis]

col1, col2, col3, col4 = st.columns(4)
receita_total = df["ticket_liquido"].sum()
total_atendimentos = len(df)
pacientes_unicos = df["paciente_nome"].nunique()
ticket_medio = receita_total / total_atendimentos if total_atendimentos else 0

col1.metric("💰 Receita total", f"R$ {receita_total:,.2f}")
col2.metric("📅 Atendimentos", total_atendimentos)
col3.metric("👥 Pacientes únicos", pacientes_unicos)
col4.metric("💵 Ticket médio", f"R$ {ticket_medio:,.2f}")

st.markdown("### 📋 Lista de protocolos realizados")
st.dataframe(df, use_container_width=True, height=350)

st.markdown("### 📈 Protocolos com maior faturamento")
prot = desempenho_protocolos(df)
if not prot.empty:
    fig = px.bar(prot.head(10), x="Protocolo", y="Receita", color="Protocolo",
                 title="Protocolos mais lucrativos")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Nenhum dado disponível para exibir o gráfico de faturamento por protocolo.")




