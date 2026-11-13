"""Pruebas del evaluador del agente (LLM simulado; no usa la API)."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import evaluacion as E
from src import features as F
from src import llm as L
from src.model import RegresionSoftmax
from src.tools import Herramientas

RAIZ = Path(__file__).resolve().parents[1]


class BuscadorFijo:
    def buscar(self, consulta, k=8):
        return [{"cita": "[POECA (Acuerdo Metropolitano 04 de 2018), Art. 17]", "documento": "POECA",
                 "ubicacion": "Art. 17", "titulo": "", "texto": "mínimo de tres (3) días", "url": "u"}]


@pytest.fixture
def h():
    rng = np.random.default_rng(1)
    fechas = pd.date_range("2026-08-01", periods=40, freq="D")
    d = pd.DataFrame([{"estacion": "MED-ARAN", "fecha": f, "pm25": float(rng.uniform(5, 50)), "n_horas": 24,
                       "valido": True} for f in fechas])
    d.loc[d.fecha == "2026-08-20", "pm25"] = 50.24
    ds = F.construir_dataset(d)
    ds = ds[ds["y"].notna()]
    m = RegresionSoftmax(l2=1e-2).fit(F.matriz_X(ds), ds["y"].astype(int).to_numpy(), 3)
    return Herramientas(d, m, buscador=BuscadorFijo())


class LLMGuion:
    def __init__(self, guion):
        self.guion = list(guion)

    def chat(self, mensajes, herramientas=None):
        return self.guion.pop(0)


def pide(nombre, args):
    return L.MensajeLLM(None, [L.LlamadaHerramienta("c1", nombre, json.dumps(args))])


def test_preguntas_yaml_es_valido():
    preguntas = yaml.safe_load((RAIZ / "eval" / "preguntas.yaml").read_text(encoding="utf-8"))
    assert 10 <= len(preguntas) <= 15
    assert len({p["id"] for p in preguntas}) == len(preguntas)
    for p in preguntas:
        assert set(p["herramientas_esperadas"]) <= {"get_pm25", "predict_tomorrow", "search_docs"}
        for patron in p.get("debe_contener") or []:
            re.compile(patron)


def test_patron_valor_acepta_coma_y_entero():
    assert re.search(E.patron_valor(50.2), "fue de 50,2 µg/m³")
    assert re.search(E.patron_valor(50.2), "fue de 50.2")
    assert not re.search(E.patron_valor(50.2), "fue de 150.2")
    assert re.search(E.patron_valor(17.0), "17 µg/m³") and re.search(E.patron_valor(17.0), "17,0")
    assert not re.search(E.patron_valor(17.0), "17.5")


def test_citas_se_normalizan_y_se_validan(h):
    item = {"id": "X", "pregunta": "¿cuánto duran?", "herramientas_esperadas": ["search_docs"],
            "requiere_cita": True, "debe_contener": ["tres \\(3\\) días"]}
    # El LLM usa un guion no separable (U+2011) y además inventa una cita.
    texto = ("Se mantienen tres (3) días [POECA (Acuerdo Metropolitano 04 de 2018), Art. 17]. "
             "También [Res. 2254 de 2017, Art.‑ 99]. Herramienta analítica, no es consejo médico.")
    fila = E.evaluar_pregunta(item, h, LLMGuion([pide("search_docs", {"consulta": "x"}), L.MensajeLLM(texto)]))
    assert fila["herramienta_correcta"] and fila["cita_presente"] and fila["n_citas"] == 2
    assert fila["citas_validas"] is False and "Art.- 99" in fila["citas_invalidas"]
    assert fila["contenido_ok"] is True and fila["aviso_medico"] is True


def test_verificacion_contra_los_datos(h):
    item = {"id": "Y", "pregunta": "pm25", "herramientas_esperadas": ["get_pm25"], "requiere_cita": False,
            "verificar": {"tipo": "pm25", "estacion": "MED-ARAN", "fecha": "2026-08-20"}}
    bien = E.evaluar_pregunta(item, h, LLMGuion([pide("get_pm25", {"estacion": "MED-ARAN", "fecha": "2026-08-20"}),
                                                 L.MensajeLLM("Fue 50,2 µg/m³.")]))
    mal = E.evaluar_pregunta(item, h, LLMGuion([L.MensajeLLM("Fue 48 µg/m³.")]))
    assert bien["contenido_ok"] and bien["citas_validas"] is None
    assert mal["contenido_ok"] is False and mal["herramienta_correcta"] is False
    assert "NO encontrado" in mal["detalle_contenido"]


def test_resumen():
    filas = pd.DataFrame([
        {"id": "a", "categoria": "datos", "herramienta_correcta": True, "requiere_cita": False,
         "cita_presente": False, "citas_validas": None, "contenido_ok": True, "aviso_medico": True,
         "error_llm": "", "segundos": 1.0},
        {"id": "b", "categoria": "normativa", "herramienta_correcta": False, "requiere_cita": True,
         "cita_presente": True, "citas_validas": False, "contenido_ok": None, "aviso_medico": False,
         "error_llm": "", "segundos": 3.0},
    ])
    r = E.resumen(filas)
    assert r["herramienta_correcta_pct"] == 50.0
    assert r["cita_presente_pct (preguntas que la exigen)"] == 100.0
    assert r["citas_validas_pct (respuestas con citas)"] == 0.0
    assert r["contenido_ok_pct (preguntas con comprobación)"] == 100.0


def test_informe_markdown_muestra_cada_pregunta():
    filas = pd.DataFrame([
        {"id": "P01", "categoria": "datos", "pregunta": "¿PM2.5?", "herramientas_esperadas": "get_pm25",
         "herramientas_llamadas": "get_pm25", "herramienta_correcta": True, "requiere_cita": False,
         "cita_presente": False, "citas_validas": None, "contenido_ok": True,
         "detalle_contenido": "PM2.5 real 17.0 encontrado", "respuesta": "17 µg/m³\nsegunda línea"},
        {"id": "P07", "categoria": "normativa", "pregunta": "¿Medidas?", "herramientas_esperadas": "search_docs",
         "herramientas_llamadas": "", "herramienta_correcta": False, "requiere_cita": True,
         "cita_presente": False, "citas_validas": None, "contenido_ok": False,
         "detalle_contenido": "", "respuesta": "No sé"},
    ])
    md = E.informe_markdown(filas, {"etiqueta": "vX", "fecha": "2026-11-12T10:00", "llm": "groq",
                                    "modelo": "m", "herramienta_correcta_pct": 50.0})
    assert "| P01 | datos | ¿PM2.5? | `get_pm25` | `get_pm25` | ✅ | no se exige |" in md
    assert "| P07 |" in md and "`ninguna`" in md and "❌" in md
    assert "<details><summary><b>P01</b>" in md and "> segunda línea" in md
    assert "| Herramienta correcta | 50 % |" in md


def test_pregunta_fuera_de_tema(h):
    item = {"id": "P13", "pregunta": "¿Capital de Bangladesh?", "herramientas_esperadas": [],
            "sin_herramientas": True, "requiere_cita": False,
            "debe_contener": ["calidad del aire"], "no_debe_contener": ["daca|dhaka"]}
    bien = E.evaluar_pregunta(item, h, LLMGuion([L.MensajeLLM("Solo ayudo con calidad del aire en el valle.")]))
    assert bien["herramienta_correcta"] and bien["contenido_ok"]
    responde = E.evaluar_pregunta(item, h, LLMGuion([L.MensajeLLM("Es Daca. Sobre calidad del aire no sé.")]))
    assert responde["contenido_ok"] is False and "NO (aparece)" in responde["detalle_contenido"]
    usa = E.evaluar_pregunta(item, h, LLMGuion([pide("search_docs", {"consulta": "Bangladesh"}),
                                                L.MensajeLLM("Solo calidad del aire.")]))
    assert usa["herramienta_correcta"] is False


def test_citas_que_entregan_las_herramientas_son_validas():
    class P:
        def __init__(self, resultado):
            self.herramienta, self.resultado = "get_pm25", resultado
    pasos = [P({"norma": "Res. 2254 de 2017 (límite diario: Art. 2; ICA: Tabla 6; niveles: Tabla 4)"}), P(None)]
    c = E.citas_de_herramientas(pasos)
    assert c == {"[Res. 2254 de 2017, Art. 2]", "[Res. 2254 de 2017, Tabla 6]", "[Res. 2254 de 2017, Tabla 4]"}
