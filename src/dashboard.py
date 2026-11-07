"""Datos para el dashboard (sin Streamlit, para poder probarlos).

app.py dibuja; este módulo prepara las tablas del mapa, la serie de tiempo y la
tarjeta de predicción a partir de las mismas Herramientas que usa el agente.
"""
from __future__ import annotations

import pandas as pd

from src import config as cfg
from src import features as F
from src import ica
from src.tools import ErrorHerramienta, Herramientas

GRIS_SIN_DATO = "#9AA0A6"


def color_de(valor: float | None) -> tuple[str, str, str]:
    """(categoría ICA, nombre del color, hex) de una concentración; gris si no hay dato."""
    if valor is None or pd.isna(valor):
        return "Sin dato válido", "Gris", GRIS_SIN_DATO
    cat = ica.categoria(float(valor))
    nombre = ica.color(cat)
    return cat, nombre, ica.COLORES_HEX[nombre]


def _rgb(hex_: str) -> list[int]:
    h = hex_.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def datos_mapa(h: Herramientas, fecha: pd.Timestamp) -> pd.DataFrame:
    """Una fila por estación con coordenadas que estaba operando en `fecha` (entre su primer y
    su último dato válido): PM2.5 del día, categoría y color. Si ese día puntual no midió,
    queda en gris; las estaciones que aún no existían o ya cerraron no se dibujan."""
    fecha = pd.Timestamp(fecha).normalize()
    dia = h.diario[(h.diario["fecha"] == fecha) & h.diario["valido"]].set_index("estacion")["pm25"]
    filas = []
    for e in h.estaciones():
        if e.get("latitud") is None or e.get("longitud") is None:
            continue
        if not (pd.Timestamp(e["desde"]) <= fecha <= pd.Timestamp(e["hasta"])):
            continue
        v = dia.get(e["codigo"])
        cat, nombre_color, hex_ = color_de(v)
        filas.append({
            "estacion": e["codigo"], "nombre": e.get("nombre") or e["codigo"],
            "municipio": e.get("municipio") or "", "zona": e.get("zona") or "",
            "latitud": float(e["latitud"]), "longitud": float(e["longitud"]),
            "pm25": None if v is None else round(float(v), 1),
            "pm25_texto": "sin dato" if v is None else f"{float(v):.1f} µg/m³",
            "categoria": cat, "color": nombre_color, "hex": hex_, "rgb": _rgb(hex_),
            "supera_37": bool(v is not None and v > cfg.LIMITE_PM25_24H),
        })
    return pd.DataFrame(filas)


def serie_estacion(h: Herramientas, estacion: str, hasta: pd.Timestamp, dias: int | None) -> pd.DataFrame:
    """PM2.5 diario válido de una estación hasta `hasta`, en una ventana de `dias` (None = todo)."""
    hasta = pd.Timestamp(hasta).normalize()
    s = h.diario[(h.diario["estacion"] == estacion) & h.diario["valido"] & (h.diario["fecha"] <= hasta)]
    if dias is not None:
        s = s[s["fecha"] > hasta - pd.Timedelta(days=dias)]
    s = s[["fecha", "pm25"]].sort_values("fecha").copy()
    s["pm25"] = s["pm25"].round(1)
    s["categoria"] = s["pm25"].map(ica.categoria)
    return s.reset_index(drop=True)


def tarjeta_prediccion(h: Herramientas, estacion: str, fecha: pd.Timestamp) -> dict:
    """Predicción para el día siguiente a `fecha` y, si existe, lo que realmente pasó.
    Lanza ErrorHerramienta con un mensaje claro si no se puede predecir."""
    p = h.predict_tomorrow(estacion, pd.Timestamp(fecha).date().isoformat())
    objetivo = pd.Timestamp(p["fecha_predicha"])
    real = h.diario[(h.diario["estacion"] == estacion) & (h.diario["fecha"] == objetivo) & h.diario["valido"]]
    p["real_pm25"] = round(float(real["pm25"].iloc[0]), 1) if len(real) else None
    p["real_clase"] = F.CLASES[F.clase_modelo(p["real_pm25"])] if p["real_pm25"] is not None else None
    p["acierto"] = (p["real_clase"] == p["clase_predicha"]) if p["real_clase"] else None
    p["persistencia"] = p["clase_ultimo_dia"]
    return p


def probabilidades(p: dict) -> pd.DataFrame:
    """Tabla larga de probabilidades por clase, con el color de la clase."""
    colores = {F.CLASES[0]: ica.COLORES_HEX["Verde"], F.CLASES[1]: ica.COLORES_HEX["Amarillo"],
               F.CLASES[2]: ica.COLORES_HEX["Naranja"]}
    return pd.DataFrame([{"clase": c, "probabilidad": float(v), "hex": colores[c],
                          "etiqueta": f"{v:.0%}"} for c, v in p["probabilidades"].items()])


__all__ = ["ErrorHerramienta", "color_de", "datos_mapa", "serie_estacion", "tarjeta_prediccion",
           "probabilidades", "GRIS_SIN_DATO"]
