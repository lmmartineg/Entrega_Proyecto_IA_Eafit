"""Pruebas del RAG con un embebedor simulado (no descarga el modelo real)."""
import json
import zlib

import numpy as np
import pytest

from src import agent as A
from src import llm as L
from src import rag as R
from src.tools import ErrorHerramienta, Herramientas


class EmbebedorSimulado:
    """Bolsa de palabras con hashing: textos que comparten palabras quedan cerca."""
    modelo = "simulado"

    def __call__(self, textos):
        v = np.zeros((len(textos), 64), dtype="float32")
        for i, t in enumerate(textos):
            for w in R.tokens(t):
                v[i, zlib.crc32(w.encode()) % 64] += 1.0
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)


HTML_PRUEBA = """<html><head><title>x</title><meta content="Ã\x81rea"></head><body>
<p>Menú del sitio</p><p>RESOLUCIÓN 2254 DE 2017</p><p>Considerando que...</p>
<p>CAPÍTULO II.</p><p>Concordancias</p>
<p>ARTÍCULO 10. RANGOS DE CONCENTRACIÓN PARA LA DECLARATORIA. Los rangos de 37 &iacute;g/m3 se fijan.</p>
<p>PARÁGRAFO. Texto del parágrafo.</p>
<p>ARTÍCULO 18. Aprobar el protocolo sin título en mayúsculas.</p>
<p>Gestor Normativo de la Comisión</p><p>Pie de página</p></body></html>"""


def test_extraccion_recorta_navegacion_y_corrige_micro():
    lineas = R.recortar(R.lineas_html(HTML_PRUEBA), r"^RESOLUCIÓN 2254", r"^Gestor Normativo")
    assert lineas[0] == "RESOLUCIÓN 2254 DE 2017"
    assert not any("Menú" in l or "Pie de página" in l for l in lineas)
    assert any("37 µg/m³" in l for l in lineas)


def test_secciones_por_articulo():
    lineas = R.recortar(R.lineas_html(HTML_PRUEBA), r"^RESOLUCIÓN 2254", r"^Gestor Normativo")
    s = R.secciones(lineas)
    assert [x["ubicacion"] for x in s] == ["Encabezado", "Art. 10", "Art. 18"]
    assert s[1]["titulo"] == "RANGOS DE CONCENTRACIÓN PARA LA DECLARATORIA"
    assert "PARÁGRAFO" in s[1]["texto"]                  # el parágrafo queda en su artículo
    assert s[2]["titulo"] == ""                           # sin título en mayúsculas
    assert not any("CAPÍTULO" in x["texto"] or "Concordancias" in x["texto"] for x in s)


def test_subsecciones_por_nivel_dentro_de_un_articulo():
    lineas = ["ARTÍCULO 27. MEDIDAS APLICABLES. Texto inicial.", "Medidas comunes para los niveles x",
              "comun", "NIVEL DE ALERTA (NIVEL III)", "medida de alerta",
              "NIVEL DE EMERGENCIA (NIVEL IV)", "medida de emergencia", "ARTÍCULO 28. OTRO. Texto."]
    subs = {r"^Medidas comunes": "medidas comunes", r"^NIVEL (DE )?ALERTA": "nivel de alerta",
            r"^NIVEL (DE )?EMERGENCIA": "nivel de emergencia"}
    s = R.secciones(lineas, subsecciones=subs)
    assert [x["ubicacion"] for x in s] == ["Art. 27", "Art. 27, medidas comunes", "Art. 27, nivel de alerta",
                                           "Art. 27, nivel de emergencia", "Art. 28"]
    assert "medida de emergencia" not in s[2]["texto"]
    assert s[3]["titulo"] == "MEDIDAS APLICABLES"            # conserva el título del artículo


def test_secciones_numeradas_y_cortes():
    lineas = ["ARTÍCULO 1. Adoptar el plan.", "DOCUMENTO DEL PLAN", "1. LA CONTAMINACIÓN DEL AIRE",
              "texto a", "1. La movilidad en las ciudades", "2.1.1 Antecedentes", "texto b"]
    s = R.secciones(lineas, secciones_numeradas=True, cortes={r"^DOCUMENTO DEL PLAN": "Documento técnico"})
    assert [x["ubicacion"] for x in s] == ["Art. 1", "Secc. 1", "Secc. 2.1.1"]
    assert "La movilidad" in s[1]["texto"]                # ítem de lista, no sección


def test_fragmentar_con_solape():
    texto = " ".join(f"p{i}" for i in range(250))
    trozos = R.fragmentar(texto, max_palabras=100, solape=20)
    assert [len(t.split()) for t in trozos] == [100, 100, 90]
    assert trozos[0].split()[-20:] == trozos[1].split()[:20]
    assert R.fragmentar("corto texto") == ["corto texto"]
    assert R.fragmentar("") == []


def test_tokens_y_plurales_coinciden():
    assert R.tokens("Niveles MÁXIMOS permisibles de PM2.5") == R.tokens("nivel máximo permisible de PM2.5")
    assert R.tokens("Niveles MÁXIMOS permisibles de PM2.5") == ["nivel", "maxim", "permisibl", "pm2.5"]
    assert R.raiz("emisiones") == R.raiz("emision") == "emision"
    assert R.raiz("alertas") == R.raiz("alerta")


def test_bm25_y_fusion_garantiza_el_mejor_de_cada_metodo():
    b = R.BM25(["calidad del aire en la ciudad", "niveles maximos permisibles", "otra cosa distinta"])
    p = b.puntajes("máximo permisible")
    assert p.argmax() == 1 and p[2] == 0
    # El 9 es el primero del método 2 pero no aparece en el método 1: igual debe entrar.
    top = R.fusion_rrf([[1, 2, 3, 4], [9, 2, 3, 4]], k_final=3)
    assert 9 in [i for i, _ in top] and 1 in [i for i, _ in top]


def test_tablas_transcritas_tienen_los_rangos():
    t = {s["ubicacion"]: s["texto"] for s in R.secciones_tablas()}
    assert "Alerta: 56 a 150 µg/m³" in t["Tabla 4"]
    assert "Emergencia: 151 µg/m³ o más" in t["Tabla 4"]
    assert "Naranja" in t["Tabla 5"] and "Material particulado" in t["Tabla 5"]
    assert "13 a 37" in t["Tabla 6"]


def fragmentos_prueba():
    secs = [
        ("Res. 2254 de 2017", "Art. 2", "NIVELES MÁXIMOS PERMISIBLES",
         "El nivel máximo permisible de PM2.5 en 24 horas es 37 µg/m³."),
        ("POECA (Acuerdo Metropolitano 04 de 2018)", "Art. 27", "MEDIDAS APLICABLES",
         "En nivel de alerta se aplica restricción vehicular de cuatro dígitos."),
        ("Informe SIATA enero 2025", "pág. 3", "", "Porcentaje de datos válidos inferior al 75 %."),
    ]
    return [{"id": i, "documento": d, "archivo": "x", "url": "u", "ubicacion": u, "titulo": t, "parte": 1,
             "texto": x} for i, (d, u, t, x) in enumerate(secs)]


@pytest.fixture
def buscador(tmp_path):
    R.construir_indice(fragmentos_prueba(), EmbebedorSimulado(), tmp_path)
    return R.Buscador.cargar(tmp_path, embebedor=EmbebedorSimulado())


def test_indice_en_disco_y_busqueda(buscador, tmp_path):
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_fragmentos"] == 3 and meta["dimension"] == 64
    r = buscador.buscar("restricción vehicular en nivel de alerta", k=2)
    assert r[0]["cita"] == "[POECA (Acuerdo Metropolitano 04 de 2018), Art. 27]"
    assert {"cita", "documento", "ubicacion", "texto"} <= set(r[0])
    for modo in ("denso", "bm25"):
        assert buscador.buscar("límite máximo permisible PM2.5", k=1, modo=modo)[0]["ubicacion"] == "Art. 2"
    with pytest.raises(ValueError):
        buscador.buscar("   ")


def test_herramienta_search_docs(buscador):
    import pandas as pd
    d = pd.DataFrame({"estacion": ["MED-ARAN"], "fecha": [pd.Timestamp("2026-09-30")], "pm25": [20.0],
                      "n_horas": [24], "valido": [True]})
    h = Herramientas(d, buscador=buscador)
    r = h.ejecutar("search_docs", {"consulta": "medidas en nivel de alerta"})
    assert r["fragmentos"][0]["cita"].startswith("[POECA")
    assert "cita" in r["instruccion"]
    with pytest.raises(ErrorHerramienta, match="no válidos"):
        h.ejecutar("search_docs", {"estacion": "MED-ARAN"})
    with pytest.raises(ErrorHerramienta, match="No hay índice"):
        Herramientas(d).search_docs("algo")


def test_agente_usa_search_docs_y_recibe_fragmentos(buscador):
    import pandas as pd
    d = pd.DataFrame({"estacion": ["MED-ARAN"], "fecha": [pd.Timestamp("2026-09-30")], "pm25": [20.0],
                      "n_horas": [24], "valido": [True]})
    h = Herramientas(d, buscador=buscador)

    class LLM:
        def __init__(self):
            self.recibido = []
            self.guion = [L.MensajeLLM(None, [L.LlamadaHerramienta("c1", "search_docs",
                                                                     '{"consulta": "nivel de alerta"}')]),
                          L.MensajeLLM("Restricción de 4 dígitos [POECA (Acuerdo Metropolitano 04 de 2018), Art. 27].")]

        def chat(self, mensajes, herramientas=None):
            self.recibido.append(list(mensajes))
            return self.guion.pop(0)

    llm = LLM()
    r = A.responder("¿Qué pasa en nivel de alerta?", h, llm)
    obs = json.loads(llm.recibido[1][-1]["content"])
    assert any(f["cita"].startswith("[POECA") for f in obs["fragmentos"])
    assert "Art. 27]" in r.texto and r.pasos[0].herramienta == "search_docs"
