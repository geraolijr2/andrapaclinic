import streamlit as st
from supabase import create_client
from datetime import date

# -------- Config --------
st.set_page_config(page_title="Pré-cadastro - Clínica", layout="centered")

SUPABASE_URL = st.secrets["supabase_url"]
SUPABASE_KEY = st.secrets["supabase_anon_key"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------- UI --------
st.title("Pré-cadastro da Clínica")
st.write("Preencha seus dados para agilizar seu atendimento.")

with st.form("cadastro_form"):
    st.markdown("### Dados pessoais")
    nome = st.text_input("Nome completo")
    telefone = st.text_input("Telefone / WhatsApp")
    data_nasc = st.date_input("Data de nascimento", max_value=date.today())
    cidade = st.text_input("Cidade / Bairro")

    st.markdown("### Ficha de Anamnese")
    doencas = st.radio("Possui doenças crônicas?", ["Não", "Sim"], horizontal=True)
    medicamentos = st.text_area("Medicamentos em uso")
    alergias = st.text_area("Alergias conhecidas")
    objetivo = st.selectbox(
        "Qual é o seu principal objetivo na clínica?",
        ["Emagrecimento", "Estética", "Prevenção", "Outro"]
    )

    enviado = st.form_submit_button("Enviar pré-cadastro")

if enviado:
    if not nome or not telefone:
        st.error("Por favor, preencha pelo menos nome e telefone.")
        st.stop()

    # 1. Cria ou busca paciente
    res = sb.table("pacientes").select("paciente_id").eq("telefone", telefone).limit(1).execute()
    if res.data:
        paciente_id = res.data[0]["paciente_id"]
    else:
        newp = sb.table("pacientes").insert({
            "nome": nome,
            "telefone": telefone,
            "cidade_bairro": cidade,
            "data_nascimento": data_nasc.isoformat()
        }).execute()
        paciente_id = newp.data[0]["paciente_id"]

    # 2. Cria agendamento pendente (data = hoje só como placeholder)
    ag = sb.table("agendamentos").insert({
        "paciente_id": paciente_id,
        "data_hora": date.today().isoformat(),
        "status": "Pendente"
    }).execute()
    agendamento_id = ag.data[0]["agendamento_id"]

    # 3. Grava anamnese
    sb.table("anamneses").insert({
        "paciente_id": paciente_id,
        "agendamento_id": agendamento_id,
        "respostas": {
            "doencas_cronicas": doencas,
            "medicamentos": medicamentos,
            "alergias": alergias,
            "objetivo": objetivo
        }
    }).execute()

    st.success("✅ Pré-cadastro enviado com sucesso!")
    st.info("Obrigado por preencher seus dados. Quando chegar à clínica, basta informar seu nome na recepção.")
