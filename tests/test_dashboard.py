"""Pruebas de los datos del dashboard y de los metadatos de estaciones (datos sintéticos)."""
import numpy as np
import pandas as pd
import pytest

from src import agent as A
from src import dashboard as D
from src import data
from src import features as F
from src import ica
from src.model import RegresionSoftmax
from src.tools import ErrorHerramienta, Herramientas, es_error_herramienta


def diario(dias=60):
    rng = np.random.default_rng(7)
    fechas = pd.date_range("2026-08-01", periods=dias, freq="D")
    filas = [{"estacion": e, "fecha": f, "pm25": float(rng.uniform(5, 50)), "n_horas": 24, "valido": True}
             for e in ("MED-ARAN", "ITA-CJUS", "SUR-TRAF") for f in fechas]
    d = pd.DataFrame(filas)
    # SUR-TRAF deja de medir a mitad del periodo; MED-ARAN no midió un día puntual.
    d = d[~((d.estacion == "SUR-TRAF") & (d.fecha > "2026-08-20"))]
    d.loc[(d.estacion == "MED-ARAN") & (d.fecha == "2026-09-10"), "valido"] = False
    return d


META = pd.DataFrame({
    "estacion": ["MED-ARAN", "ITA-CJUS", "SUR-TRAF"],
    "nombre": ["I.E. Ciro Mendia", "Casa de Justicia de Itagüi", "Estación Tráfico Sur"],
    "latitud": [6.29, 6.18, 6.15], "longitud": [-75.55, -75.59, -75.62],
    "municipio": ["Medellín", "Itagüí", "Sabaneta"], "zona": ["Aranjuez", None, None],
})


@pytest.fixture
def h():
    d = diario()
    ds = F.construir_dataset(d)
    ds = ds[ds["y"].notna()]
    m = RegresionSoftmax(l2=1e-2).fit(F.matriz_X(ds), ds["y"].astype(int).to_numpy(), 3)
    return Herramientas(d, m, meta_estaciones=META)


def test_metadatos_oficiales_mandan_sobre_el_prefijo(h):
    assert h.municipio("SUR-TRAF") == "Sabaneta"           # el prefijo SUR no dice el municipio
    assert h.resolver_estacion("Aranjuez") == "MED-ARAN"    # por zona
    assert h.resolver_estacion("ciro mendia") == "MED-ARAN"  # por nombre
    r = h.get_pm25("MED-ARAN")
    assert r["nombre_estacion"] == "I.E. Ciro Mendia" and r["zona"] == "Aranjuez"


def test_leer_metadatos_normaliza_municipio(tmp_path):
    ruta = tmp_path / "e.tab"
    ruta.write_text("Código\tNombre_Completo_Estacion\tNombre_Corto_Estacion\tLatitud\tLongitud\tMunicipio\n"
                    "104\tAlmacen General de EPM\tMED-PROV\t6.2297\t-75.5860\tMedellin - Belén\n"
                    "90\tI.E Rafael J. Mejía\tSAB-RAME\t6.1455\t-75.6212\tSabaneta\n", encoding="utf-8")
    m = data.leer_metadatos_estaciones(ruta).set_index("estacion")
    assert m.loc["MED-PROV", "municipio"] == "Medellín" and m.loc["MED-PROV", "zona"] == "Belén"
    assert m.loc["SAB-RAME", "municipio"] == "Sabaneta" and pd.isna(m.loc["SAB-RAME", "zona"])
    assert data.leer_metadatos_estaciones(tmp_path / "no_existe.tab") is None


def test_mapa_solo_estaciones_en_operacion_y_gris_si_falta_el_dia(h):
    m = D.datos_mapa(h, pd.Timestamp("2026-09-10")).set_index("estacion")
    assert "SUR-TRAF" not in m.index                        # ya había cerrado
    assert m.loc["MED-ARAN", "hex"] == D.GRIS_SIN_DATO      # operando, pero sin dato ese día
    assert m.loc["MED-ARAN", "pm25_texto"] == "sin dato"
    it = m.loc["ITA-CJUS"]
    assert it["categoria"] == ica.categoria(it["pm25"])
    assert it["hex"] == ica.COLORES_HEX[ica.color(it["categoria"])]
    assert len(it["rgb"]) == 3
    assert "SUR-TRAF" in D.datos_mapa(h, pd.Timestamp("2026-08-15"))["estacion"].values


def test_serie_respeta_ventana_y_solo_dias_validos(h):
    s = D.serie_estacion(h, "MED-ARAN", pd.Timestamp("2026-09-29"), 30)
    assert s["fecha"].min() > pd.Timestamp("2026-08-30") and s["fecha"].max() == pd.Timestamp("2026-09-29")
    assert pd.Timestamp("2026-09-10") not in set(s["fecha"])
    assert len(D.serie_estacion(h, "MED-ARAN", pd.Timestamp("2026-09-29"), None)) == 59


def test_tarjeta_prediccion_compara_con_lo_real(h):
    p = D.tarjeta_prediccion(h, "MED-ARAN", pd.Timestamp("2026-09-20"))
    assert p["fecha_predicha"] == "2026-09-21"
    assert p["real_pm25"] is not None and p["real_clase"] in F.CLASES
    assert p["acierto"] == (p["real_clase"] == p["clase_predicha"])
    probs = D.probabilidades(p)
    assert probs["probabilidad"].sum() == pytest.approx(1.0, abs=1e-3)
    ultimo = D.tarjeta_prediccion(h, "MED-ARAN", pd.Timestamp("2026-09-29"))
    assert ultimo["real_pm25"] is None and ultimo["acierto"] is None    # aún no hay dato del día siguiente


def test_error_de_herramienta_se_reconoce_aunque_la_clase_se_haya_recargado():
    assert es_error_herramienta(ErrorHerramienta("x"))
    Recargada = type("ErrorHerramienta", (ValueError,), {})   # como tras una recarga de Streamlit
    assert es_error_herramienta(Recargada("x"))
    assert not es_error_herramienta(KeyError("x"))


def test_instruccion_incluye_fecha_y_nombres(h):
    s = A.instruccion_sistema(h, "MED-ARAN", "2026-09-10")
    assert "2026-09-10" in s and "I.E. Ciro Mendia" in s and "Sabaneta" in s
