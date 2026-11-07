"""Corre la evaluación del agente sobre eval/preguntas.yaml.

Uso (desde la raíz del proyecto, con el LLM configurado en .env):
    python eval/run_eval.py                 # las 12 preguntas
    python eval/run_eval.py --ids P07,P12   # solo algunas
    python eval/run_eval.py --etiqueta v2   # nombre para guardar la corrida

Escribe:
    eval/resultados.csv          última corrida, una fila por pregunta (incluye la respuesta)
    eval/resumen.json            porcentajes de la última corrida
    eval/corridas/<fecha>_<etiqueta>.csv   historial de corridas
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import evaluacion as E  # noqa: E402
from src import llm as L  # noqa: E402
from src.tools import Herramientas  # noqa: E402

CARPETA = RAIZ / "eval"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", help="ids separados por coma (por defecto, todas)")
    ap.add_argument("--etiqueta", default="corrida")
    ap.add_argument("--pausa", type=float, default=2.0, help="segundos entre preguntas (límites de la API)")
    a = ap.parse_args(argv)

    preguntas = yaml.safe_load((CARPETA / "preguntas.yaml").read_text(encoding="utf-8"))
    if a.ids:
        ids = set(a.ids.split(","))
        preguntas = [p for p in preguntas if p["id"] in ids]
    h = Herramientas.desde_disco()
    cliente = L.ClienteLLM()
    print(f"{len(preguntas)} preguntas · LLM: {cliente.proveedor} "
          f"({cliente.modelo or L.PROVEEDORES[cliente.proveedor]['modelo']})\n")

    filas = []
    for i, p in enumerate(preguntas):
        fila = E.evaluar_pregunta(p, h, cliente)
        filas.append(fila)
        marca = lambda v: "-" if v is None else ("✓" if v else "✗")
        print(f"{fila['id']}  herramienta {marca(fila['herramienta_correcta'])} "
              f"({fila['herramientas_llamadas'] or 'ninguna'})  "
              f"cita {marca(fila['cita_presente']) if fila['requiere_cita'] else '-'}  "
              f"citas válidas {marca(fila['citas_validas'])}  contenido {marca(fila['contenido_ok'])}  "
              f"aviso {marca(fila['aviso_medico'])}  {fila['segundos']} s")
        if fila["error_llm"]:
            print(f"     error LLM: {fila['error_llm'][:150]}")
        if fila["contenido_ok"] is False or fila["citas_invalidas"]:
            print(f"     {fila['detalle_contenido']} {fila['citas_invalidas']}")
        if i < len(preguntas) - 1:
            time.sleep(a.pausa)

    df = pd.DataFrame(filas)
    res = E.resumen(df)
    res.update({"llm": cliente.proveedor, "modelo": cliente.modelo or L.PROVEEDORES[cliente.proveedor]["modelo"],
                "fecha": datetime.now().isoformat(timespec="seconds"), "etiqueta": a.etiqueta})
    (CARPETA / "corridas").mkdir(exist_ok=True)
    df.to_csv(CARPETA / "resultados.csv", index=False)
    df.to_csv(CARPETA / "corridas" / f"{datetime.now():%Y%m%d-%H%M%S}_{a.etiqueta}.csv", index=False)
    (CARPETA / "resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nResumen:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("\nPor categoría:")
    print(E.por_categoria(df).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
