"""Configuración central del proyecto.

Todo lo que dependa de la fuente de datos vive aquí, para que el resto del
código no tenga nombres de columnas "quemados".
"""
from pathlib import Path

# --- Rutas -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_CORPUS = ROOT / "data" / "corpus"

# --- Fuente principal: Repositorio de Datos SIATA (Dataverse) ----------------
# SIATA, 2026, "Material Particulado - PM2.5", https://doi.org/10.83041/AUWZWT, V5.
# Se descarga con `python scripts/download_siata.py`. Verificado el 6 nov 2026:
# un .tab por mes, separado por tabuladores, en formato ANCHO:
#   fecha_hora | BAR-TORR | MED-ARAN | ... (una columna por estación, µg/m³)
# Celda vacía = hora sin dato. Paso horario, de 00:00 a 23:00.
SIATA_DIR = DATA_RAW / "siata_pm25"
SIATA_PATRON = "Estaciones_PM2.5_*.tab"
SIATA_COL_FECHA = "fecha_hora"

# --- Respaldo: datos.gov.co (API SODA / Socrata) -----------------------------
# Recurso citado en el README del proyecto BRUMA como el PM2.5 horario de la
# red AMVA/SIATA publicado por IDEAM (SISAIRE):
# https://github.com/saracastrillons/bruma
# Sin confirmar; solo se usa si falla el SIATA (`python -m src.data inspect`).
DATASET_ID = "g4t8-zkc3"
RESOURCE_URL = f"https://www.datos.gov.co/resource/{DATASET_ID}.json"

# Paginación (https://dev.socrata.com/docs/paging.html):
# máximo 50 000 filas por página en endpoints 2.0; $order=:id da orden estable.
PAGE_SIZE = 50_000

# --- Mapeo de columnas (solo respaldo datos.gov.co / CSV manual) -------------
# El SIATA no lo necesita. Llénalo con lo que imprime `python -m src.data inspect`.
# Clave = rol que usa el código; valor = nombre real de la columna en la API.
# Deja en None lo que no exista.
COLUMNS = {
    "fecha": None,         # fecha y hora de la medición (obligatoria)
    "estacion": None,      # nombre o código de la estación (obligatoria)
    "valor": None,         # concentración (obligatoria)
    "variable": None,      # columna con el nombre del contaminante, si los datos vienen "en largo"
    "unidad": None,        # unidad de medida, si existe
    "departamento": None,
    "municipio": None,
    "latitud": None,
    "longitud": None,
}

# Valores exactos para filtrar en la API (cópialos tal cual salen en `inspect`).
DEPARTAMENTO = None        # p. ej. el valor que aparezca para Antioquia
VARIABLE = None            # p. ej. el valor que aparezca para PM2.5
MUNICIPIOS = None          # lista opcional de municipios del Valle de Aburrá

# --- Reglas de limpieza ------------------------------------------------------
# Fracción mínima de horas válidas para aceptar un promedio diario.
# 0.75 = 18 de 24 h. Verificado en el informe de operación SIATA de enero 2025
# (data/corpus/): "NP: No hay promedio por porcentaje de datos válidos inferior al 75 %".
MIN_FRACCION_HORAS = 0.75

# Heurística (no normativa) para marcar estación-año con unidades sospechosas.
# BRUMA reporta medianas de hasta 2.000 en 14 estaciones entre 2020 y 2023 en
# datos.gov.co, frente a un límite diario de 37 µg/m³. En los datos del SIATA
# (2019-2026) no aparece: medianas anuales entre 6,5 y 30 µg/m³. Una mediana anual mayor a 100 µg/m³
# no es plausible para el Valle de Aburrá y se marca para revisar a mano.
MEDIANA_SOSPECHOSA = 100.0

# --- Norma (Resolución 2254 de 2017, MADS) -----------------------------------
# Art. 2, parágrafo 1: desde el 1 jul 2018 el máximo permisible de PM2.5 en
# 24 h es 37 µg/m³.
# https://normas.cra.gov.co/gestor/docs/resolucion_minambienteds_2254_2017.htm
LIMITE_PM25_24H = 37.0
