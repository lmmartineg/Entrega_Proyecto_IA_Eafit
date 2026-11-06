"""Dashboard de calidad del aire del Valle de Aburrá (esqueleto de la Sesión 3).

Uso (desde la raíz del proyecto):
    streamlit run app.py

La Sesión 5 agrega el mapa, la serie de tiempo y la tarjeta de predicción.
"""
from __future__ import annotations

import json

import streamlit as st

from src import agent as A
from src import llm as L
from src.tools import Herramientas

st.set_page_config(page_title="Calidad del aire · Valle de Aburrá", layout="wide")
AVISO = ("Herramienta analítica con fines académicos; no es consejo médico. "
         "Datos: Repositorio de Datos SIATA (doi:10.83041/AUWZWT), procesados por este proyecto.")


@st.cache_resource
def cargar_herramientas() -> Herramientas:
    return Herramientas.desde_disco()


@st.cache_resource
def cargar_cliente():
    return L.ClienteLLM()


h = cargar_herramientas()
estaciones = sorted(h.estaciones(), key=lambda e: (not e["activa"], e["codigo"]))

with st.sidebar:
    st.header("Estación")
    etiquetas = {e["codigo"]: e["codigo"] + (f" · {e['municipio']}" if e["municipio"] else "")
                 + ("" if e["activa"] else " (inactiva)") for e in estaciones}
    estacion = st.selectbox("Estación del SIATA", list(etiquetas), format_func=etiquetas.get)
    st.caption(f"Datos disponibles hasta {h.fecha_max.date()}. No hay datos en tiempo real.")
    try:
        cliente = cargar_cliente()
        st.caption(f"LLM: {cliente.proveedor}"
                   + (f" (alterno: {cliente.proveedor_alt})" if cliente.proveedor_alt else ""))
    except L.ErrorLLM as e:
        cliente = None
        st.error(f"Configuración del LLM: {e}")

st.title("Calidad del aire en el Valle de Aburrá")
st.caption(AVISO)

col1, col2 = st.columns(2)
with col1:
    dato = h.get_pm25(estacion)
    st.metric(f"PM2.5 diario en {estacion} ({dato['fecha']})",
              f"{dato['pm25_promedio_diario_ug_m3']} µg/m³", help="Límite diario: 37 µg/m³ (Res. 2254)")
    st.write(f"Categoría ICA: **{dato['categoria_ica']}** ({dato['color_ica']})")
with col2:
    try:
        pred = h.predict_tomorrow(estacion)
        st.metric(f"Predicción para {pred['fecha_predicha']}", pred["clase_predicha"])
        st.caption(" · ".join(f"{k}: {v:.0%}" for k, v in pred["probabilidades"].items()))
    except Exception as e:  # el panel no debe caerse por la predicción
        st.warning(f"Sin predicción: {e}")

st.divider()
st.subheader("Pregúntale al asistente")
if "chat" not in st.session_state:
    st.session_state.chat = []

for turno in st.session_state.chat:
    with st.chat_message(turno["role"]):
        st.markdown(turno["content"])
        for p in turno.get("pasos", []):
            with st.expander(f"Herramienta: {p['herramienta']}"):
                st.code(json.dumps(p, ensure_ascii=False, indent=1, default=str), language="json")

pregunta = st.chat_input(f"Por ejemplo: ¿cómo está el aire hoy en {estacion}?")
if pregunta:
    with st.chat_message("user"):
        st.markdown(pregunta)
    historial = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat]
    with st.chat_message("assistant"):
        if cliente is None:
            r = A.RespuestaAgente("El LLM no está configurado; revisa el archivo .env (ver .env.example).")
        else:
            with st.spinner("Consultando..."):
                r = A.responder(pregunta, h, cliente, historial, estacion_ui=estacion)
        if r.error_llm:
            st.error(r.texto)
        else:
            st.markdown(r.texto)
        pasos = [{"herramienta": p.herramienta, "argumentos": p.argumentos,
                  "resultado": p.resultado, "error": p.error} for p in r.pasos]
        for p in pasos:
            with st.expander(f"Herramienta: {p['herramienta']}" + (" (error)" if p["error"] else "")):
                st.code(json.dumps(p, ensure_ascii=False, indent=1, default=str), language="json")
        if r.agoto_t_max:
            st.caption(f"Se alcanzó el límite de {A.T_MAX} iteraciones.")
    st.session_state.chat += [{"role": "user", "content": pregunta},
                              {"role": "assistant", "content": r.texto, "pasos": pasos}]
