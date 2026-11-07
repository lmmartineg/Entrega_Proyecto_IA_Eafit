"""Descarga el histórico de PM2.5 del Repositorio de Datos SIATA (Dataverse).

Dataset: SIATA, 2026, "Material Particulado - PM2.5",
https://doi.org/10.83041/AUWZWT, Repositorio de Datos SIATA, V5.
Lee las condiciones de uso del dataset antes de redistribuir los datos.

Uso (desde la raíz del proyecto):
    python scripts/download_siata.py --listar                 # ver archivos disponibles
    python scripts/download_siata.py --desde 2019 --hasta 2026  # descargar

Endpoints según la guía oficial de Dataverse:
- Dataset por DOI: GET /api/datasets/:persistentId/?persistentId=<doi>
  https://guides.dataverse.org/en/4.9/api/native-api.html
- Archivo por id: GET /api/access/datafile/<id>  (un .tab ingerido se entrega
  separado por tabuladores con fila de encabezado)
  https://guides.dataverse.org/en/latest/api/dataaccess.html
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as cfg  # noqa: E402

SERVER = "https://datos.siata.gov.co"
DOI = "doi:10.83041/AUWZWT"
DESTINO = cfg.DATA_RAW / "siata_pm25"
# Metadatos de la red (nombre, coordenadas y municipio de cada estación):
# SIATA, "Información de la Red de Calidad del Aire", https://doi.org/10.83041/XTI3FH
DOI_RED = "doi:10.83041/XTI3FH"
DESTINO_RED = cfg.DATA_RAW / "siata_red"
ARCHIVOS_RED = ("Estaciones_calidad_aire.tab", "Fechas_monitoreo_calidad_aire.tab")
HEADERS = {"User-Agent": "si3003-calidad-aire/0.1 (proyecto academico)"}
PATRON = re.compile(r"_(\d{4})_(\d{2})\.tab$", re.IGNORECASE)


def listar_archivos() -> list[dict]:
    """Devuelve [{id, nombre, carpeta, tamano}] de la última versión publicada."""
    url = f"{SERVER}/api/datasets/:persistentId/"
    try:
        r = requests.get(url, params={"persistentId": DOI}, headers=HEADERS, timeout=60)
    except requests.RequestException as e:
        raise SystemExit(f"No se pudo conectar con {SERVER}: {e.__class__.__name__}.\n"
                         "Revisa tu conexión o usa la descarga manual (ver README).")
    if r.status_code != 200:
        raise SystemExit(f"Error {r.status_code} al consultar el dataset:\n{r.text[:400]}\n"
                         "Usa la descarga manual (ver README).")
    try:
        archivos = r.json()["data"]["latestVersion"]["files"]
    except (KeyError, ValueError):
        # La estructura no es la esperada: se guarda para revisarla.
        DESTINO.mkdir(parents=True, exist_ok=True)
        (DESTINO / "_respuesta_dataset.json").write_text(r.text, encoding="utf-8")
        raise SystemExit("La respuesta no tiene data.latestVersion.files. "
                         f"Guardé la respuesta en {DESTINO / '_respuesta_dataset.json'}.")
    salida = []
    for f in archivos:
        df = f.get("dataFile", {})
        salida.append({
            "id": df.get("id"),
            "nombre": df.get("filename") or f.get("label"),
            "carpeta": f.get("directoryLabel", ""),
            "tamano": df.get("filesize"),
        })
    return salida


def filtrar(archivos: list[dict], desde: int, hasta: int) -> list[dict]:
    elegidos = []
    for a in archivos:
        m = PATRON.search(a["nombre"] or "")
        if m and desde <= int(m.group(1)) <= hasta:
            elegidos.append({**a, "anio": int(m.group(1)), "mes": int(m.group(2))})
    return sorted(elegidos, key=lambda a: (a["anio"], a["mes"]))


def descargar(elegidos: list[dict]) -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    ok, fallas = 0, []
    for a in elegidos:
        ruta = DESTINO / a["nombre"]
        if ruta.exists() and ruta.stat().st_size > 0:
            continue
        try:
            r = requests.get(f"{SERVER}/api/access/datafile/{a['id']}", headers=HEADERS, timeout=120)
        except requests.RequestException as e:
            fallas.append((a["nombre"], e.__class__.__name__))
            print(f"[falló conexión] {a['nombre']}")
            continue
        if r.status_code == 200:
            ruta.write_bytes(r.content)
            ok += 1
            print(f"[ok] {a['nombre']} ({len(r.content) / 1024:.0f} KB)")
        else:
            fallas.append((a["nombre"], r.status_code))
            print(f"[falló {r.status_code}] {a['nombre']}")
        time.sleep(0.5)  # no saturar el servidor

    manifiesto = {
        "fuente": f"https://doi.org/{DOI.removeprefix('doi:')}",
        "cita": 'SIATA, 2026, "Material Particulado - PM2.5", '
                "https://doi.org/10.83041/AUWZWT, Repositorio de Datos SIATA, V5",
        "archivos": [a["nombre"] for a in elegidos],
        "fallidos": fallas,
        "descargado": datetime.now().isoformat(timespec="seconds"),
    }
    (DESTINO / "_manifiesto.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"\n{ok} descargados, {len(fallas)} fallidos. Carpeta: {DESTINO}")
    if fallas:
        print("Si los fallos son 401/403, el dataset pide aceptar condiciones: descárgalos desde la web.")


def descargar_red() -> None:
    """Baja los metadatos de las estaciones (archivos pequeños) a data/raw/siata_red/."""
    r = requests.get(f"{SERVER}/api/datasets/:persistentId/", params={"persistentId": DOI_RED},
                     headers=HEADERS, timeout=60)
    r.raise_for_status()
    ids = {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in r.json()["data"]["latestVersion"]["files"]}
    DESTINO_RED.mkdir(parents=True, exist_ok=True)
    for nombre in ARCHIVOS_RED:
        if nombre not in ids:
            print(f"[falta en el dataset] {nombre}")
            continue
        a = requests.get(f"{SERVER}/api/access/datafile/{ids[nombre]}", headers=HEADERS, timeout=60)
        if a.status_code == 200:
            (DESTINO_RED / nombre).write_bytes(a.content)
            print(f"[ok] {nombre} ({len(a.content)} B)")
        else:
            print(f"[falló {a.status_code}] {nombre}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listar", action="store_true", help="solo listar archivos")
    p.add_argument("--desde", type=int, default=2019)
    p.add_argument("--hasta", type=int, default=datetime.now().year)
    a = p.parse_args()

    archivos = listar_archivos()
    if a.listar:
        for x in archivos:
            print(f"{x['carpeta']:>6}  {x['nombre']}  ({x['tamano']} B)  id={x['id']}")
        otros = [x["nombre"] for x in archivos if not PATRON.search(x["nombre"] or "")]
        print(f"\n{len(archivos)} archivos, {len(archivos) - len(otros)} mensuales.")
        if otros:
            print("Otros archivos (metadatos, diccionarios; conviene bajarlos a mano):")
            for n in otros:
                print(f"  - {n}")
        return
    elegidos = filtrar(archivos, a.desde, a.hasta)
    print(f"{len(elegidos)} archivos mensuales entre {a.desde} y {a.hasta}")
    descargar(elegidos)
    print("
Metadatos de las estaciones:")
    descargar_red()


if __name__ == "__main__":
    main()
