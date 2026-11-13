"""Pruebas del agente y de las herramientas con un LLM simulado y datos sintéticos."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from src import agent as A
from src import features as F
from src import llm as L
from src.model import RegresionSoftmax
from src.tools import ErrorHerramienta, Herramientas


def diario_sintetico(dias=60, semilla=4):
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2026-08-01", periods=dias, freq="D")
    filas = []
    for est in ("MED-ARAN", "ITA-CJUS", "ITA-CONC"):
        for f in fechas:
            filas.append({"estacion": est, "fecha": f, "pm25": float(rng.uniform(5, 50)),
                          "n_horas": 24, "valido": True})
    d = pd.DataFrame(filas)
    # Un día con pocas horas en MED-ARAN.
    d.loc[(d.estacion == "MED-ARAN") & (d.fecha == "2026-08-10"), ["n_horas", "valido"]] = [10, False]
    return d


@pytest.fixture
def herr():
    d = diario_sintetico()
    ds = F.construir_dataset(d)
    ds = ds[ds["y"].notna()]
    m = RegresionSoftmax(l2=1e-2).fit(F.matriz_X(ds), ds["y"].astype(int).to_numpy(), 3)
    return Herramientas(d, m, {"entrenado_con": "sintético"})


class LLMSimulado:
    """Devuelve las respuestas del guion en orden y guarda lo que recibió."""
    def __init__(self, guion):
        self.guion = list(guion)
        self.recibido = []

    def chat(self, mensajes, herramientas=None):
        self.recibido.append({"mensajes": copy.deepcopy(mensajes), "herramientas": herramientas})
        r = self.guion.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def pide(nombre, args, id_="c1"):
    return L.MensajeLLM(None, [L.LlamadaHerramienta(id_, nombre, args if isinstance(args, str) else json.dumps(args))])


def texto(t):
    return L.MensajeLLM(t)


# --- Herramientas -----------------------------------------------------------------
def test_get_pm25_sin_fecha_usa_el_ultimo_dato_y_lo_dice(herr):
    r = herr.get_pm25("MED-ARAN")
    assert r["fecha"] == "2026-09-29"
    assert "nota" in r and "tiempo real" in r["nota"]
    assert r["supera_limite_diario"] == (r["pm25_promedio_diario_ug_m3"] > 37)


def test_get_pm25_dia_con_pocas_horas_advierte(herr):
    r = herr.get_pm25("MED-ARAN", "2026-08-10")
    assert not r["dia_valido"] and "advertencia" in r


def test_get_pm25_errores_claros(herr):
    with pytest.raises(ErrorHerramienta, match="varias estaciones"):
        herr.get_pm25("Itagüí")
    with pytest.raises(ErrorHerramienta, match="No reconozco"):
        herr.get_pm25("Narnia")
    with pytest.raises(ErrorHerramienta, match="no tiene datos"):
        herr.get_pm25("MED-ARAN", "2031-01-01")
    with pytest.raises(ErrorHerramienta, match="Fecha inválida"):
        herr.get_pm25("MED-ARAN", "ayer por la tarde")


def test_resolver_estacion_variantes(herr):
    assert herr.resolver_estacion("med aran") == "MED-ARAN"
    assert herr.resolver_estacion("medaran") == "MED-ARAN"
    assert herr.resolver_estacion("ITA-CJUS") == "ITA-CJUS"


def test_predict_tomorrow(herr):
    r = herr.predict_tomorrow("MED-ARAN")
    assert r["fecha_predicha"] == "2026-09-30"
    assert sum(r["probabilidades"].values()) == pytest.approx(1.0, abs=1e-3)
    assert r["clase_predicha"] in F.CLASES


def test_herramientas_avisan_otras_estaciones_del_municipio(herr):
    assert herr.get_pm25("ITA-CJUS")["otras_estaciones_activas_del_municipio"] == ["ITA-CONC"]
    assert herr.predict_tomorrow("ITA-CONC")["otras_estaciones_activas_del_municipio"] == ["ITA-CJUS"]
    assert "otras_estaciones_activas_del_municipio" not in herr.get_pm25("MED-ARAN")


def test_predict_tomorrow_sin_modelo():
    h = Herramientas(diario_sintetico(), modelo=None)
    with pytest.raises(ErrorHerramienta, match="No hay modelo"):
        h.predict_tomorrow("MED-ARAN")


# --- Agente -------------------------------------------------------------------------
def test_agente_llama_get_pm25_y_usa_el_resultado(herr):
    llm = LLMSimulado([pide("get_pm25", {"estacion": "MED-ARAN"}), texto("Hoy hay 20 µg/m³.")])
    r = A.responder("¿Cómo está el aire hoy en MED-ARAN?", herr, llm)
    assert [p.herramienta for p in r.pasos] == ["get_pm25"]
    assert r.pasos[0].error is None and r.pasos[0].resultado["estacion"] == "MED-ARAN"
    assert r.texto == "Hoy hay 20 µg/m³." and r.iteraciones == 2 and not r.agoto_t_max
    # La segunda llamada al LLM recibe la observación de la herramienta.
    ultimo = llm.recibido[1]["mensajes"][-1]
    assert ultimo["role"] == "tool" and ultimo["tool_call_id"] == "c1"
    assert json.loads(ultimo["content"])["fecha"] == "2026-09-29"


def test_error_de_herramienta_vuelve_como_observacion(herr):
    llm = LLMSimulado([pide("get_pm25", {"estacion": "Narnia"}), texto("No encontré esa estación.")])
    r = A.responder("aire en Narnia", herr, llm)
    assert "No reconozco" in r.pasos[0].error
    obs = json.loads(llm.recibido[1]["mensajes"][-1]["content"])
    assert "error" in obs and "No reconozco" in obs["error"]


def test_argumentos_json_invalidos_y_herramienta_desconocida(herr):
    llm = LLMSimulado([pide("get_pm25", "{no es json"), pide("borrar_todo", {}, "c2"), texto("listo")])
    r = A.responder("x", herr, llm)
    assert "Argumentos inválidos" in r.pasos[0].error
    assert "desconocida" in r.pasos[1].error
    assert r.texto == "listo"


def test_t_max_corta_el_loop_y_cierra_sin_herramientas(herr):
    t_max = 3
    llm = LLMSimulado([pide("get_pm25", {"estacion": "MED-ARAN"}, f"c{i}") for i in range(t_max)]
                      + [texto("Con lo que tengo: 20 µg/m³.")])
    r = A.responder("x", herr, llm, t_max=t_max)
    assert r.agoto_t_max and len(r.pasos) == t_max
    assert len(llm.recibido) == t_max + 1
    assert llm.recibido[-1]["herramientas"] is None      # la llamada final no ofrece herramientas
    assert r.texto.startswith("Con lo que tengo")


def test_falla_de_la_api_no_tumba_el_agente(herr):
    llm = LLMSimulado([L.ErrorLLM("groq: RateLimitError")])
    r = A.responder("x", herr, llm)
    assert r.error_llm and "No pude contactar" in r.texto


def test_instruccion_sistema_menciona_limite_de_datos(herr):
    s = A.instruccion_sistema(herr, "MED-ARAN")
    assert "2026-09-29" in s and "MED-ARAN" in s and "no es consejo médico" in s


def test_limpiar_respuesta_normaliza_citas_lenticulares():
    # Casos reales de gpt-oss en la evaluación v2 (P09, P10, P06).
    t = "efectos graves【Res. 2254 de 2017, Tablas 4-6 (transcripción del proyecto), Tabla 5】."
    assert A.limpiar_respuesta(t) == "efectos graves[Res. 2254 de 2017, Tablas 4-6 (transcripción del proyecto), Tabla 5]."
    t = 'límite diario【cita: "[Res. 2254 de 2017, Tablas 4-6 (transcripción del proyecto), Tabla 6]"}】.'
    assert A.limpiar_respuesta(t) == "límite diario[Res. 2254 de 2017, Tablas 4-6 (transcripción del proyecto), Tabla 6]."
    assert A.limpiar_respuesta("es **Buena** 【...predicted data...】.") == "es **Buena**."
    # Casos de la evaluación v3: espacio de ancho cero y espacio duro antes de ":".
    assert A.limpiar_respuesta("38 a 55​[Res. 2254 de 2017, Art. 10]​.") == "38 a 55[Res. 2254 de 2017, Art. 10]."
    assert A.limpiar_respuesta("las siguientes medidas :") == "las siguientes medidas:"
    assert A.limpiar_respuesta("Sin marcas [POECA (Acuerdo Metropolitano 04 de 2018), Art. 17], fin.") == \
        "Sin marcas [POECA (Acuerdo Metropolitano 04 de 2018), Art. 17], fin."


def test_espera_sugerida_respeta_lo_que_pide_la_api():
    class Resp:
        headers = {"retry-after": "7"}

    class ErrConEncabezado(Exception):
        response = Resp()

    assert L.espera_sugerida(ErrConEncabezado("x"), 0) == pytest.approx(7.5)
    msg = Exception("Rate limit reached ... Please try again in 10.42s. Need more tokens?")
    assert L.espera_sugerida(msg, 0) == pytest.approx(10.92)
    assert L.espera_sugerida(Exception("sin pista"), 2) == pytest.approx(8.5)
    assert L.espera_sugerida(Exception("try again in 120s"), 0) == L.ESPERA_MAXIMA


def test_cliente_sin_clave_da_error_claro(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(L, "cargar_entorno", lambda: None)
    c = L.ClienteLLM(proveedor="groq", proveedor_alt="")
    with pytest.raises(L.ErrorLLM, match="GROQ_API_KEY"):
        c.chat([{"role": "user", "content": "hola"}])
    # El aviso dice qué hacer y que el resto de la app funciona sin clave.
    aviso = c.clave_faltante()
    assert "console.groq.com" in aviso and "README" in aviso and "funcionan sin clave" in aviso
    monkeypatch.setenv("GROQ_API_KEY", "x" * 20)
    assert c.clave_faltante() is None
