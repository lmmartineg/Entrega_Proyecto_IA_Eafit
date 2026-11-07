"""Dashboard de calidad del aire del Valle de Aburrá.

Uso (desde la raíz del proyecto):
    streamlit run app.py

Contenido: selección de estación y fecha, mapa de estaciones con el PM2.5 del
día, serie de tiempo con el límite diario de 37 µg/m³, tarjeta de predicción
del día siguiente (con lo que pasó realmente, si ya hay dato) y chat con el agente.
"""
from __future__ import annotations

import json

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from src import agent as A
from src import config as cfg
from src import dashboard as D
from src import features as F
from src import ica
from src import llm as L
from src.tools import Herramientas, es_error_herramienta

st.set_page_config(page_title="Calidad del aire · Valle de Aburrá", layout="wide")
AVISO = ("Herramienta analítica con fines académicos; **no es consejo médico**. "
         "Datos: Repositorio de Datos SIATA (doi:10.83041/AUWZWT y doi:10.83041/XTI3FH), "
         "procesados por este proyecto. Pronóstico: modelo propio, no del SIATA.")
NARANJA = ica.COLORES_HEX["Naranja"]


@st.cache_resource
def cargar_herramientas() -> Herramientas:
    return Herramientas.desde_disco()


@st.cache_resource
def cargar_cliente():
    return L.ClienteLLM()


def chip(hex_: str, texto: str) -> str:
    """Cuadro de color + texto: el color nunca va solo."""
    return (f"<span style='display:inline-block;width:0.8em;height:0.8em;border-radius:2px;"
            f"background:{hex_};border:1px solid rgba(0,0,0,.35);margin-right:.4em;"
            f"vertical-align:-0.05em'></span>{texto}")


h = cargar_herramientas()
estaciones = sorted(h.estaciones(), key=lambda e: (not e["activa"], e["codigo"]))
etiquetas = {e["codigo"]: f"{e['codigo']} · {e.get('nombre') or ''}"
             + (f" ({e['municipio']})" if e.get("municipio") else "")
             + ("" if e["activa"] else " · inactiva") for e in estaciones}
fecha_min, fecha_max = h.diario["fecha"].min().date(), h.fecha_max.date()
if "estacion" not in st.session_state:
    st.session_state.estacion = "MED-ARAN" if "MED-ARAN" in etiquetas else next(iter(etiquetas))

# --- Barra lateral: contexto y estado -----------------------------------------
with st.sidebar:
    st.header("Acerca de")
    st.markdown(AVISO)
    st.caption(f"Datos disponibles: {fecha_min} a {fecha_max}. No hay datos en tiempo real.")
    try:
        cliente = cargar_cliente()
        st.caption(f"LLM: {cliente.proveedor}"
                   + (f" (alterno: {cliente.proveedor_alt})" if cliente.proveedor_alt else ""))
    except L.ErrorLLM as e:
        cliente = None
        st.error(f"Configuración del LLM: {e}")
    if h.buscador is None:
        st.warning("Sin índice de documentos: el asistente no podrá citar normativa. "
                   "Corre `python -m src.rag construir`.")

st.title("Calidad del aire en el Valle de Aburrá")

# --- Fila de filtros (aplica a todo lo de abajo) --------------------------------
f1, f2, f3 = st.columns([2.2, 1, 1])
estacion = f1.selectbox("Estación", list(etiquetas), format_func=etiquetas.get, key="estacion")
fecha = pd.Timestamp(f2.date_input("Fecha", value=fecha_max, min_value=fecha_min, max_value=fecha_max,
                                   format="YYYY-MM-DD"))
ventanas = {"Últimos 90 días": 90, "Último año": 365, "Todo el periodo": None}
ventana = f3.selectbox("Serie de tiempo", list(ventanas))

# --- Tarjetas: medido, predicción, lo que pasó ----------------------------------
t1, t2, t3 = st.columns(3)
with t1:
    st.markdown(f"**PM2.5 medido · {fecha.date()}**")
    try:
        d = h.get_pm25(estacion, fecha.date().isoformat())
        st.metric("Promedio diario", f"{d['pm25_promedio_diario_ug_m3']} µg/m³", label_visibility="collapsed")
        st.markdown(chip(ica.COLORES_HEX[d["color_ica"]], f"ICA: {d['categoria_ica']}"), unsafe_allow_html=True)
        if d["supera_limite_diario"]:
            st.markdown(f"Supera el límite diario de {cfg.LIMITE_PM25_24H:g} µg/m³ · "
                        f"nivel de **{d['nivel_episodio_res_2254']}** (Res. 2254, Tabla 4)")
        if "advertencia" in d:
            st.warning(d["advertencia"])
    except Exception as e:
        if not es_error_herramienta(e):
            raise
        st.info(f"Sin dato: {e}")

with t2:
    try:
        p = D.tarjeta_prediccion(h, estacion, fecha)
        st.markdown(f"**Predicción para {p['fecha_predicha']}**")
        corto = {F.CLASES[2]: "Naranja o peor"}.get(p["clase_predicha"], p["clase_predicha"])
        st.metric("Clase predicha", corto, label_visibility="collapsed")
        if corto != p["clase_predicha"]:
            st.caption(f"{p['clase_predicha']} (> {cfg.LIMITE_PM25_24H:g} µg/m³)")
        probs = D.probabilidades(p)
        barras = alt.Chart(probs).mark_bar(cornerRadiusEnd=4, height=14).encode(
            x=alt.X("probabilidad:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)),
            y=alt.Y("clase:N", sort=F.CLASES, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("hex:N", scale=None, legend=None),
            tooltip=[alt.Tooltip("clase:N", title="Clase"), alt.Tooltip("probabilidad:Q", format=".1%")])
        textos = barras.mark_text(align="left", dx=4).encode(text="etiqueta:N", color=alt.value("#666"))
        st.altair_chart((barras + textos).properties(height=110), use_container_width=True)
    except Exception as e:
        if not es_error_herramienta(e):
            raise
        p = None
        st.markdown("**Predicción**")
        st.info(str(e))

with t3:
    st.markdown("**Lo que pasó realmente**")
    if p is None:
        st.caption("Sin predicción para comparar.")
    elif p["real_pm25"] is None:
        st.caption(f"Aún no hay dato válido para {p['fecha_predicha']} "
                   f"(los datos descargados llegan hasta {fecha_max}).")
    else:
        cat, nom, hex_ = D.color_de(p["real_pm25"])
        st.metric("Medido", f"{p['real_pm25']} µg/m³", label_visibility="collapsed")
        st.markdown(chip(hex_, f"ICA: {cat}"), unsafe_allow_html=True)
        st.markdown(("El modelo **acertó**" if p["acierto"] else "El modelo **falló**")
                    + f" · la persistencia decía: {p['persistencia']}")
    if p is not None and "desempeno_en_prueba" in p:
        dp = p["desempeno_en_prueba"]
        st.caption(f"Desempeño en prueba 2024-2026: F1 macro {dp['f1_macro_modelo']:.2f} "
                   f"(persistencia {dp['f1_macro_persistencia']:.2f}).")

st.divider()

# --- Mapa y serie de tiempo -------------------------------------------------------
c_mapa, c_serie = st.columns([1.05, 1])


def al_seleccionar_en_mapa():
    """Clic en una estación del mapa -> la selecciona en la fila de filtros."""
    try:
        objetos = st.session_state["mapa"]["selection"]["objects"].get("estaciones", [])
        if objetos:
            st.session_state.estacion = objetos[0]["estacion"]
    except (KeyError, TypeError, AttributeError):
        pass


with c_mapa:
    st.subheader(f"PM2.5 por estación · {fecha.date()}")
    try:
        m = D.datos_mapa(h, fecha)
        m["radio"] = m["estacion"].map(lambda c: 900 if c == estacion else 550)
        m["borde"] = m["estacion"].map(lambda c: [20, 20, 20, 255] if c == estacion else [40, 40, 40, 140])
        capa = pdk.Layer(
            # deck.gl espera la posición como expresión en texto, no como lista de Python.
            "ScatterplotLayer", id="estaciones", data=m, get_position="[longitud, latitud]",
            # Radio en metros (unidad por defecto). No pasar radius_units="meters": pydeck lo
            # convierte en la expresión "@@=meters" y deck.gl dibuja círculos gigantes.
            get_fill_color="rgb", get_radius="radio", radius_min_pixels=5, radius_max_pixels=18,
            stroked=True,
            get_line_color="borde", line_width_min_pixels=1.5, pickable=True, auto_highlight=True)
        deck = pdk.Deck(
            layers=[capa], map_provider="carto", map_style="light",
            # Centro y zoom para ver del sur (Caldas, 6.09°) al norte (Barbosa, 6.44°).
            initial_view_state=pdk.ViewState(latitude=6.265, longitude=-75.50, zoom=9.6),
            tooltip={"html": "<b>{nombre}</b> ({estacion})<br/>{municipio} {zona}<br/>"
                             "PM2.5: {pm25_texto}<br/>{categoria}"})
        st.pydeck_chart(deck, on_select=al_seleccionar_en_mapa, selection_mode="single-object",
                        key="mapa", use_container_width=True)
        presentes = [c for c in ["Verde", "Amarillo", "Naranja", "Rojo", "Púrpura", "Marrón"]
                     if c in set(m["color"])]
        nombres = {c: n for _, _, n, c in ica.ICA_PM25}
        leyenda = " &nbsp; ".join(chip(ica.COLORES_HEX[c], nombres[c]) for c in presentes)
        if m["pm25"].isna().any():
            leyenda += " &nbsp; " + chip(D.GRIS_SIN_DATO, "Sin dato válido")
        st.markdown(f"<small>{leyenda}</small>", unsafe_allow_html=True)
        st.caption(f"{m['pm25'].notna().sum()} de {len(m)} estaciones en operación con dato ese día · "
                   f"{int(m['supera_37'].sum())} superan 37 µg/m³ · clic en una estación para seleccionarla.")
        with st.expander("Ver tabla del mapa"):
            st.dataframe(m[["estacion", "nombre", "municipio", "pm25", "categoria"]]
                         .sort_values("pm25", ascending=False, na_position="last"),
                         hide_index=True, use_container_width=True)
    except Exception as e:  # el panel no debe caerse por el mapa
        st.error(f"No se pudo dibujar el mapa: {e.__class__.__name__}: {e}")

with c_serie:
    st.subheader(f"Serie diaria · {estacion}")
    try:
        s = D.serie_estacion(h, estacion, fecha, ventanas[ventana])
        if s.empty:
            st.info(f"{estacion} no tiene días válidos en esa ventana.")
        else:
            x = alt.X("fecha:T", title=None, axis=alt.Axis(format="%Y-%m-%d", labelAngle=0, tickCount=5))
            y = alt.Y("pm25:Q", title="PM2.5 promedio diario (µg/m³)",
                      scale=alt.Scale(domainMin=0, nice=True))
            base = alt.Chart(s).encode(x=x, y=y)
            linea = base.mark_line(strokeWidth=2)
            cerca = alt.selection_point(nearest=True, on="pointerover", fields=["fecha"], empty=False)
            puntos = base.mark_point(size=70, filled=True).encode(
                opacity=alt.condition(cerca, alt.value(1), alt.value(0)),
                tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%Y-%m-%d"),
                         alt.Tooltip("pm25:Q", title="PM2.5 (µg/m³)", format=".1f"),
                         alt.Tooltip("categoria:N", title="ICA")]).add_params(cerca)
            cursor = base.mark_rule(color="#888").encode(
                opacity=alt.condition(cerca, alt.value(0.6), alt.value(0)))
            lim = pd.DataFrame({"y": [cfg.LIMITE_PM25_24H], "x": [s["fecha"].min()],
                                "t": [f"Límite diario {cfg.LIMITE_PM25_24H:g} µg/m³ (arriba: Naranja o peor)"]})
            regla = alt.Chart(lim).mark_rule(color=NARANJA, strokeDash=[6, 4], strokeWidth=1.5).encode(y="y:Q")
            rotulo = alt.Chart(lim).mark_text(align="left", baseline="bottom", dx=4, dy=-3, color=NARANJA).encode(
                x="x:T", y="y:Q", text="t:N")
            capas = [linea, regla, rotulo, cursor, puntos]
            if fecha in set(s["fecha"]):
                hoy = s[s["fecha"] == fecha]
                capas.append(alt.Chart(hoy).mark_point(size=110, filled=False, strokeWidth=2, color="#222")
                             .encode(x=x, y=y))
            st.altair_chart(alt.layer(*capas).properties(height=360), use_container_width=True)
            sobre = int((s["pm25"] > cfg.LIMITE_PM25_24H).sum())
            st.caption(f"{len(s)} días válidos · máximo {s['pm25'].max():.1f} µg/m³ · "
                       f"{sobre} días por encima de 37 µg/m³ · el círculo negro marca la fecha elegida.")
            with st.expander("Ver datos de la serie"):
                st.dataframe(s.sort_values("fecha", ascending=False), hide_index=True,
                             use_container_width=True)
    except Exception as e:
        st.error(f"No se pudo dibujar la serie: {e.__class__.__name__}: {e}")

st.divider()

# --- Chat con el agente -------------------------------------------------------------
st.subheader("Pregúntale al asistente")
st.caption("Consulta datos medidos, el pronóstico del modelo o la normativa (Res. 2254, POECA, PIGECA). "
           "Las respuestas sobre normas citan [documento, artículo].")
if "chat" not in st.session_state:
    st.session_state.chat = []


def mostrar_pasos(pasos: list[dict]) -> None:
    for p_ in pasos:
        if p_["error"]:
            st.warning(f"La herramienta {p_['herramienta']} devolvió un error: {p_['error']}")
        with st.expander(f"Herramienta: {p_['herramienta']}" + (" (error)" if p_["error"] else "")):
            st.code(json.dumps(p_, ensure_ascii=False, indent=1, default=str), language="json")


# El chat va dentro de contenedores: así el cuadro de texto queda en su lugar en la página
# (no fijo al fondo) y Streamlit no desplaza la app hasta abajo al abrirla.
zona_mensajes = st.container()
with zona_mensajes:
    for turno in st.session_state.chat:
        with st.chat_message(turno["role"]):
            st.markdown(turno["content"])
            mostrar_pasos(turno.get("pasos", []))

pregunta = st.container().chat_input(
    f"Por ejemplo: ¿qué pasa si se declara nivel de alerta? o ¿cómo estuvo {estacion}?")
if pregunta:
    with zona_mensajes:
        with st.chat_message("user"):
            st.markdown(pregunta)
        historial = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat]
        with st.chat_message("assistant"):
            if cliente is None:
                r = A.RespuestaAgente("El LLM no está configurado; revisa el archivo .env (ver .env.example).",
                                      error_llm="sin configuración")
            else:
                with st.spinner("Consultando..."):
                    r = A.responder(pregunta, h, cliente, historial, estacion_ui=estacion,
                                    fecha_ui=fecha.date().isoformat())
            if r.error_llm:
                st.error(r.texto)
            else:
                st.markdown(r.texto)
            pasos = [{"herramienta": x.herramienta, "argumentos": x.argumentos,
                      "resultado": x.resultado, "error": x.error} for x in r.pasos]
            mostrar_pasos(pasos)
            if r.agoto_t_max:
                st.caption(f"Se alcanzó el límite de {A.T_MAX} iteraciones.")
    st.session_state.chat += [{"role": "user", "content": pregunta},
                              {"role": "assistant", "content": r.texto, "pasos": pasos}]
