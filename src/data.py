"""Sesión 1: descarga y limpieza del PM2.5 de la red AMVA/SIATA.

Uso (desde la raíz del proyecto):

Fuente principal, Repositorio de Datos SIATA:
    python scripts/download_siata.py --desde 2019 --hasta 2026  # 1) bajar los .tab
    python -m src.data process                                   # 2) limpiar y agregar a diario

Respaldo, datos.gov.co o CSV manual (requiere llenar COLUMNS en config):
    python -m src.data inspect                      # ver columnas reales de la API
    python -m src.data download --desde 2019-01-01  # descargar a data/raw/pm25_raw.csv
    python -m src.data process --raw data/raw/pm25_raw.csv

Paginación según https://dev.socrata.com/docs/paging.html
($limit, $offset y $order=:id para páginas estables).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from src import config as cfg

HEADERS = {"User-Agent": "si3003-calidad-aire/0.1 (proyecto academico)"}

# Palabras clave para sugerir el mapeo de columnas (solo una sugerencia).
PALABRAS_CLAVE = {
    "fecha": ["fecha", "date", "hora"],
    "estacion": ["estacion", "estación", "station"],
    "valor": ["concentracion", "concentración", "valor", "value", "medicion"],
    "variable": ["variable", "contaminante", "parametro", "parámetro"],
    "unidad": ["unidad", "unit"],
    "departamento": ["departamento"],
    "municipio": ["municipio"],
    "latitud": ["latitud", "lat"],
    "longitud": ["longitud", "lon"],
}
OBLIGATORIAS = ("fecha", "estacion", "valor")


# --- Acceso a la API ---------------------------------------------------------
def get_json(params: dict, url: str = cfg.RESOURCE_URL, intentos: int = 3) -> list[dict]:
    """GET con reintentos; devuelve la lista de filas JSON."""
    for i in range(intentos):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=90)
            if r.status_code == 400:
                # Error de consulta: no tiene sentido reintentar.
                raise SystemExit(f"La API rechazó la consulta (400):\n{r.text[:500]}")
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if i == intentos - 1:
                raise SystemExit(f"No se pudo consultar {url}: {e}\nUsa el plan B (ver README).")
            time.sleep(2 * (i + 1))
    return []


def sugerir_mapeo(columnas: list[str]) -> dict:
    sugerido = {}
    for rol, claves in PALABRAS_CLAVE.items():
        sugerido[rol] = next(
            (c for c in columnas if any(k in c.lower() for k in claves)), None
        )
    return sugerido


def inspect(n: int = 5) -> None:
    filas = get_json({"$limit": n})
    if not filas:
        print("La API respondió sin filas.")
        return
    columnas = sorted({k for f in filas for k in f})
    print(f"Recurso: {cfg.RESOURCE_URL}\nColumnas ({len(columnas)}):")
    for c in columnas:
        print(f"  - {c}")
    print("\nFilas de ejemplo:")
    print(pd.DataFrame(filas).to_string(max_colwidth=40))
    print("\nMapeo sugerido (REVÍSALO antes de copiarlo a src/config.py):")
    print(json.dumps(sugerir_mapeo(columnas), indent=2, ensure_ascii=False))


# --- Descarga ----------------------------------------------------------------
def exigir_mapeo() -> dict:
    m = cfg.COLUMNS
    faltan = [r for r in OBLIGATORIAS if not m.get(r)]
    if faltan:
        raise SystemExit(
            f"Faltan columnas en src/config.py -> COLUMNS: {faltan}. "
            "Corre primero `python -m src.data inspect`."
        )
    return m


def construir_where(desde: str | None, usar_filtros: bool) -> str | None:
    if not usar_filtros:
        return None
    m, partes = cfg.COLUMNS, []
    if m.get("departamento") and cfg.DEPARTAMENTO:
        partes.append(f"{m['departamento']}='{cfg.DEPARTAMENTO}'")
    if m.get("variable") and cfg.VARIABLE:
        partes.append(f"{m['variable']}='{cfg.VARIABLE}'")
    if m.get("municipio") and cfg.MUNICIPIOS:
        lista = ",".join(f"'{x}'" for x in cfg.MUNICIPIOS)
        partes.append(f"{m['municipio']} in({lista})")
    if desde:
        partes.append(f"{m['fecha']} >= '{desde}T00:00:00'")
    return " AND ".join(partes) or None


def download(desde: str | None, usar_filtros: bool = True) -> Path:
    exigir_mapeo()
    where = construir_where(desde, usar_filtros)
    base = {"$limit": cfg.PAGE_SIZE, "$order": ":id"}
    if where:
        base["$where"] = where
    print(f"Consulta: {base}")

    paginas, offset = [], 0
    while True:
        filas = get_json({**base, "$offset": offset})
        if not filas:
            break
        paginas.append(pd.DataFrame(filas))
        offset += len(filas)
        print(f"  {offset:,} filas descargadas")
        if len(filas) < cfg.PAGE_SIZE:
            break

    if not paginas:
        raise SystemExit("La consulta no devolvió filas. Revisa DEPARTAMENTO/VARIABLE en config.")
    df = pd.concat(paginas, ignore_index=True)

    cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)
    salida = cfg.DATA_RAW / "pm25_raw.csv"
    df.to_csv(salida, index=False)
    # Registro de la consulta para reproducibilidad.
    meta = {
        "fuente": cfg.RESOURCE_URL,
        "parametros": base,
        "filas": len(df),
        "descargado": datetime.now().isoformat(timespec="seconds"),
    }
    (cfg.DATA_RAW / "pm25_raw.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Guardado {salida} ({len(df):,} filas)")
    return salida


# --- Lectura SIATA -----------------------------------------------------------
MAPEO_LARGO = {"fecha": "fecha", "estacion": "estacion", "valor": "valor"}


def leer_siata(carpeta: Path | str | None = None) -> pd.DataFrame:
    """Lee los .tab mensuales del SIATA (formato ancho) y los pasa a formato largo.

    Devuelve columnas fecha, estacion, valor (texto o número, sin limpiar), lista
    para `limpiar_horario(..., MAPEO_LARGO)`. Las celdas vacías quedan como NaN y
    las descarta la limpieza. Si una estación no aparece en un mes, simplemente
    no tiene filas ese mes.
    """
    carpeta = Path(carpeta) if carpeta else cfg.SIATA_DIR
    archivos = sorted(carpeta.glob(cfg.SIATA_PATRON))
    if not archivos:
        raise SystemExit(f"No hay archivos {cfg.SIATA_PATRON} en {carpeta}. "
                         "Corre `python scripts/download_siata.py` o usa --raw.")
    partes = []
    for f in archivos:
        ancho = pd.read_csv(f, sep="\t", dtype=str)
        if cfg.SIATA_COL_FECHA not in ancho.columns:
            raise ValueError(f"{f.name}: falta la columna '{cfg.SIATA_COL_FECHA}'. "
                             f"Columnas: {list(ancho.columns)[:5]}...")
        partes.append(ancho.melt(id_vars=cfg.SIATA_COL_FECHA,
                                 var_name="estacion", value_name="valor"))
    largo = pd.concat(partes, ignore_index=True)
    largo = largo.rename(columns={cfg.SIATA_COL_FECHA: "fecha"})
    largo["estacion"] = largo["estacion"].str.strip()
    print(f"Leídos {len(archivos)} archivos de {carpeta} "
          f"({largo['estacion'].nunique()} estaciones)")
    return largo


# --- Limpieza ----------------------------------------------------------------
def limpiar_horario(raw: pd.DataFrame, columnas: dict | None = None,
                    variable: str | None = None) -> pd.DataFrame:
    """Renombra, tipa y depura. Devuelve columnas estandarizadas:
    estacion, fecha, pm25 (+ latitud, longitud, municipio si existen)."""
    m = {k: v for k, v in (columnas or cfg.COLUMNS).items() if v}
    variable = cfg.VARIABLE if variable is None else variable
    faltan = [m[r] for r in OBLIGATORIAS if m.get(r) and m[r] not in raw.columns]
    if faltan:
        raise KeyError(f"Columnas no encontradas en los datos: {faltan}")

    df = raw.rename(columns={v: k for k, v in m.items()})
    # Los mismos filtros de la API se repiten aquí, por si el CSV vino sin filtrar.
    if "variable" in df.columns and variable:
        df = df[df["variable"] == variable]
    if "departamento" in df.columns and cfg.DEPARTAMENTO:
        df = df[df["departamento"] == cfg.DEPARTAMENTO]
    if "municipio" in df.columns and cfg.MUNICIPIOS:
        df = df[df["municipio"].isin(cfg.MUNICIPIOS)]
    df = df.copy()

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["pm25"] = pd.to_numeric(
        df["valor"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    n0 = len(df)
    df = df.dropna(subset=["fecha", "pm25", "estacion"])
    df = df[df["pm25"] >= 0]
    print(f"Limpieza: {n0 - len(df):,} filas descartadas de {n0:,} "
          "(hora sin dato, fecha/valor inválido o negativo)")

    extras = [c for c in ("latitud", "longitud", "municipio") if c in df.columns]
    for c in ("latitud", "longitud"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Duplicados estación-hora: se promedian.
    agg = {"pm25": "mean", **{c: "first" for c in extras}}
    df = df.groupby(["estacion", "fecha"], as_index=False).agg(agg)
    return df.sort_values(["estacion", "fecha"]).reset_index(drop=True)


def resolucion_horas(horario: pd.DataFrame) -> float:
    """Mediana del paso temporal entre registros, en horas."""
    pasos = horario.groupby("estacion")["fecha"].diff().dropna()
    return pasos.median() / pd.Timedelta(hours=1) if len(pasos) else float("nan")


def promedio_diario(horario: pd.DataFrame,
                    min_frac: float = cfg.MIN_FRACCION_HORAS) -> pd.DataFrame:
    """Promedio diario por estación, descartando días con pocas horas válidas.
    Si los datos ya vienen diarios (paso >= 20 h), no aplica la regla de horas."""
    paso = resolucion_horas(horario)
    d = horario.assign(dia=horario["fecha"].dt.floor("D"))
    out = d.groupby(["estacion", "dia"], as_index=False).agg(
        pm25=("pm25", "mean"), n_horas=("pm25", "size")
    )
    if paso >= 20:
        print(f"Aviso: paso mediano de {paso:.1f} h; los datos parecen diarios, no se filtra por horas.")
        out["valido"] = True
    else:
        minimo = math.ceil(24 * min_frac)
        out["valido"] = out["n_horas"] >= minimo
        print(f"Días con al menos {minimo} h válidas: "
              f"{out['valido'].mean():.1%} de {len(out):,} estación-día")
    return out.rename(columns={"dia": "fecha"})


def reporte_unidades(horario: pd.DataFrame,
                     umbral: float = cfg.MEDIANA_SOSPECHOSA) -> pd.DataFrame:
    """Mediana y percentil 99 por estación y año; marca medianas sospechosas."""
    r = (horario.assign(anio=horario["fecha"].dt.year)
         .groupby(["estacion", "anio"])["pm25"]
         .agg(n="size", mediana="median", p99=lambda s: s.quantile(0.99))
         .reset_index())
    r["sospechoso"] = r["mediana"] > umbral
    return r


RUTA_META_ESTACIONES = cfg.DATA_RAW / "siata_red" / "Estaciones_calidad_aire.tab"


def leer_metadatos_estaciones(ruta: Path = RUTA_META_ESTACIONES) -> pd.DataFrame | None:
    """Nombre, coordenadas y municipio oficiales (SIATA, doi:10.83041/XTI3FH).
    "Medellin - Belén" se separa en municipio "Medellín" y zona "Belén"."""
    if not Path(ruta).exists():
        return None
    m = pd.read_csv(ruta, sep="	")
    muni = m["Municipio"].astype(str).str.strip().str.replace("Medellin", "Medellín", regex=False)
    partes = muni.str.split(" - ", n=1, expand=True)
    return pd.DataFrame({
        "estacion": m["Nombre_Corto_Estacion"].astype(str).str.strip(),
        "nombre": m["Nombre_Completo_Estacion"].astype(str).str.strip(),
        "latitud": pd.to_numeric(m["Latitud"], errors="coerce"),
        "longitud": pd.to_numeric(m["Longitud"], errors="coerce"),
        "municipio": partes[0].str.strip(),
        "zona": partes[1].str.strip() if partes.shape[1] > 1 else None,
    })


def tabla_estaciones(horario: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    cols = [c for c in ("latitud", "longitud", "municipio") if c in horario.columns]
    g = horario.groupby("estacion")
    t = g["fecha"].agg(desde="min", hasta="max").reset_index()
    if cols:
        t = t.merge(g[cols].first().reset_index(), on="estacion")
    if meta is not None:
        t = t.drop(columns=[c for c in ("latitud", "longitud", "municipio") if c in t.columns])
        t = t.merge(meta, on="estacion", how="left")
        faltan = t.loc[t["latitud"].isna(), "estacion"].tolist()
        if faltan:
            print(f"Aviso: sin coordenadas oficiales para {faltan}")
    return t


def process(ruta_raw: Path | None = None) -> None:
    """Sin argumentos usa los .tab del SIATA; con `ruta_raw`, un CSV en formato
    largo (datos.gov.co o descarga manual) mapeado con cfg.COLUMNS."""
    if ruta_raw is None:
        horario = limpiar_horario(leer_siata(), MAPEO_LARGO, variable="")
    else:
        ruta_raw = Path(ruta_raw)
        if not ruta_raw.exists():
            raise SystemExit(f"No existe {ruta_raw}.")
        exigir_mapeo()
        horario = limpiar_horario(pd.read_csv(ruta_raw, low_memory=False))
    diario = promedio_diario(horario)
    unidades = reporte_unidades(horario)
    estaciones = tabla_estaciones(horario, leer_metadatos_estaciones())

    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    horario.to_csv(cfg.DATA_PROCESSED / "pm25_horario.csv", index=False)
    diario.to_csv(cfg.DATA_PROCESSED / "pm25_diario.csv", index=False)
    unidades.to_csv(cfg.DATA_PROCESSED / "reporte_unidades.csv", index=False)
    estaciones.to_csv(cfg.DATA_PROCESSED / "estaciones.csv", index=False)

    print(f"\nEstaciones: {horario['estacion'].nunique()}")
    print(f"Rango: {horario['fecha'].min()} a {horario['fecha'].max()}")
    print(f"Paso mediano: {resolucion_horas(horario):.2f} h")
    sosp = unidades[unidades["sospechoso"]]
    print(f"Estación-año con mediana > {cfg.MEDIANA_SOSPECHOSA:g}: {len(sosp)}")
    if len(sosp):
        print(sosp.to_string(index=False))
    print(f"Archivos en {cfg.DATA_PROCESSED}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect"); s.add_argument("-n", type=int, default=5)
    s = sub.add_parser("download")
    s.add_argument("--desde", help="YYYY-MM-DD")
    s.add_argument("--sin-filtros", action="store_true",
                   help="descarga sin $where (si la API rechaza el filtro)")
    s = sub.add_parser("process")
    s.add_argument("--raw", help="CSV en formato largo (respaldo); sin esto usa los .tab del SIATA")
    a = p.parse_args(argv)

    if a.cmd == "inspect":
        inspect(a.n)
    elif a.cmd == "download":
        download(a.desde, usar_filtros=not a.sin_filtros)
    elif a.cmd == "process":
        process(a.raw)


if __name__ == "__main__":
    main(sys.argv[1:])
