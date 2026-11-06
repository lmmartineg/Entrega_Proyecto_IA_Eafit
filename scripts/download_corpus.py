"""Descarga el corpus del RAG a data/corpus/.

Uso: python scripts/download_corpus.py

Cada documento indica cuándo se verificó su URL. Además de los archivos, el
script escribe data/corpus/_fuentes.json (título, URL, fecha) para que el RAG
pueda citar la fuente de cada fragmento. Si una descarga falla, imprime qué
bajar a mano.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as cfg  # noqa: E402

HEADERS = {"User-Agent": "si3003-calidad-aire/0.1 (proyecto academico)"}
NORMOGRAMA_AMVA = "http://alphasig.metropol.gov.co/normograma/compilacion/docs"

DOCUMENTOS = [
    {
        "archivo": "res_2254_2017_mads.html",
        "titulo": "Resolución 2254 de 2017, MADS (norma de calidad del aire)",
        "corto": "Res. 2254 de 2017",
        "url": "https://normas.cra.gov.co/gestor/docs/resolucion_minambienteds_2254_2017.htm",
        "estado": "verificada el 6 nov 2026. Las tablas 1-6 son imágenes: los puntos "
                  "de corte del ICA se transcriben a mano desde el PDF oficial (Diario Oficial 50.415).",
    },
    {
        "archivo": "poeca_acuerdo_04_2018_amva.html",
        "titulo": "POECA, Acuerdo Metropolitano 04 de 2018 (texto compilado, incluye "
                  "las modificaciones del Acuerdo 7 de 2022)",
        "corto": "POECA (Acuerdo Metropolitano 04 de 2018)",
        "url": f"{NORMOGRAMA_AMVA}/a_amva_0004_2018.htm",
        "estado": "verificada el 6 nov 2026 en el normograma oficial del AMVA.",
    },
    {
        "archivo": "pigeca_acuerdo_16_2017_amva.html",
        "titulo": "PIGECA, Acuerdo Metropolitano 16 de 2017",
        "corto": "PIGECA (Acuerdo Metropolitano 16 de 2017)",
        "url": f"{NORMOGRAMA_AMVA}/a_amva_0016_2017.htm",
        "estado": "verificada el 6 nov 2026 en el normograma oficial del AMVA "
                  "(el PDF de medellin.gov.co devuelve 404).",
    },
    {
        "archivo": "siata_informe_operacion_2025_01.pdf",
        "titulo": "Informe de operación Red de Calidad del Aire, enero 2025 (SIATA)",
        "corto": "Informe SIATA enero 2025",
        "url": "https://siata.gov.co/CalidadAire/Informe_Procuraduria/"
               "F-GAA-RA-82_InformeProcuraduria_Enero2025.pdf",
        "estado": "verificada el 6 nov 2026 (17 páginas con texto).",
    },
]


def contenido_valido(nombre: str, contenido: bytes) -> bool:
    """Evita guardar una página de error con extensión .pdf."""
    if nombre.endswith(".pdf"):
        return contenido[:4] == b"%PDF"
    return len(contenido) > 1024


def main() -> None:
    cfg.DATA_CORPUS.mkdir(parents=True, exist_ok=True)
    fallidos = []
    for d in DOCUMENTOS:
        destino = cfg.DATA_CORPUS / d["archivo"]
        if destino.exists() and destino.stat().st_size > 0:
            print(f"[ya existe] {destino.name}")
            continue
        try:
            r = requests.get(d["url"], headers=HEADERS, timeout=90)
            r.raise_for_status()
        except requests.RequestException as e:
            fallidos.append((d, e))
            print(f"[falló] {d['titulo']}: {e}")
            continue
        if not contenido_valido(d["archivo"], r.content):
            fallidos.append((d, "contenido inesperado"))
            print(f"[falló] {d['titulo']}: la respuesta no parece el documento")
            continue
        destino.write_bytes(r.content)
        print(f"[ok] {destino.name} ({len(r.content) / 1024:.0f} KB) — {d['estado']}")

    fuentes = [{k: d[k] for k in ("archivo", "titulo", "corto", "url")} for d in DOCUMENTOS
               if (cfg.DATA_CORPUS / d["archivo"]).exists()]
    (cfg.DATA_CORPUS / "_fuentes.json").write_text(
        json.dumps({"actualizado": date.today().isoformat(), "documentos": fuentes},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(fuentes)} de {len(DOCUMENTOS)} documentos en {cfg.DATA_CORPUS}")

    if fallidos:
        print("\nDescarga manual necesaria:")
        for d, e in fallidos:
            print(f"  - {d['titulo']} ({e})\n    {d['url']}\n    -> data/corpus/{d['archivo']}")


if __name__ == "__main__":
    main()
