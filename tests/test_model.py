"""Pruebas del clasificador propio y de las variables (datos sintéticos)."""
import numpy as np
import pandas as pd
import pytest

from src import features as F
from src import model as M


# --- Modelo -------------------------------------------------------------------
def test_gradiente_coincide_con_diferencias_finitas():
    rng = np.random.default_rng(0)
    n, p, K = 40, 4, 3
    Z = rng.normal(size=(n, p))
    y = rng.integers(0, K, n)
    Y = np.eye(K)[y]
    s = M.pesos_muestra(y, K, "balanceado")
    m = M.RegresionSoftmax(l2=0.1)
    W, b = rng.normal(size=(p, K)), rng.normal(size=K)
    _, gW, gb = m.perdida_y_gradiente(W, b, Z, Y, s)
    h = 1e-6
    num_W = np.zeros_like(W)
    for i in range(p):
        for k in range(K):
            Wp, Wm = W.copy(), W.copy()
            Wp[i, k] += h
            Wm[i, k] -= h
            num_W[i, k] = (m.perdida_y_gradiente(Wp, b, Z, Y, s)[0]
                           - m.perdida_y_gradiente(Wm, b, Z, Y, s)[0]) / (2 * h)
    num_b = np.array([(m.perdida_y_gradiente(W, b + h * np.eye(K)[k], Z, Y, s)[0]
                       - m.perdida_y_gradiente(W, b - h * np.eye(K)[k], Z, Y, s)[0]) / (2 * h)
                      for k in range(K)])
    assert np.allclose(gW, num_W, atol=1e-6)
    assert np.allclose(gb, num_b, atol=1e-6)


def nubes(n_por_clase=200, sep=4.0, semilla=1):
    rng = np.random.default_rng(semilla)
    centros = np.array([[0, 0], [sep, 0], [0, sep]])
    X = np.vstack([rng.normal(c, 1.0, size=(n_por_clase, 2)) for c in centros])
    y = np.repeat(np.arange(3), n_por_clase)
    return X, y


def test_aprende_nubes_separables_y_converge():
    X, y = nubes()
    m = M.RegresionSoftmax(l2=1e-3).fit(X, y)
    assert m.convergio_
    assert (m.predict(X) == y).mean() > 0.95
    assert np.allclose(m.predict_proba(X).sum(axis=1), 1.0)
    # La pérdida baja (Nesterov puede oscilar un poco, pero termina mucho más abajo).
    assert m.historia_[-1] < 0.5 * m.historia_[0]


def test_esquemas_de_pesos_normalizados_y_ordenados():
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    sin, raiz, bal = (M.pesos_clase(y, 3, e) for e in (None, "raiz", "balanceado"))
    for w in (sin, raiz, bal):
        assert w[y].mean() == pytest.approx(1.0)        # peso medio por muestra = 1
    assert np.allclose(bal, len(y) / (3 * np.array([80, 15, 5])))   # = sklearn "balanced"
    assert sin[2] < raiz[2] < bal[2]                    # la raíz es un punto intermedio
    with pytest.raises(ValueError):
        M.pesos_clase(y, 3, "otro")


def test_pesos_balanceados_suben_sensibilidad_de_clase_rara():
    X, y = nubes(sep=1.5, semilla=2)
    keep = (y != 2) | (np.arange(len(y)) % 10 == 0)      # clase 2 queda con 20 ejemplos
    X, y = X[keep], y[keep]
    sin = M.RegresionSoftmax(l2=1e-3).fit(X, y)
    con = M.RegresionSoftmax(l2=1e-3, pesos_clase="balanceado").fit(X, y)
    sens = lambda m: M.metricas(y, m.predict(X), 3)["sensibilidad_2"]
    assert sens(con) > sens(sin)


def test_guardar_y_cargar(tmp_path):
    X, y = nubes(n_por_clase=50)
    m = M.RegresionSoftmax(l2=1e-2).fit(X, y)
    m.guardar(tmp_path / "m", {"columnas_x": ["a", "b"]})
    m2, meta = M.RegresionSoftmax.cargar(tmp_path / "m")
    assert meta["columnas_x"] == ["a", "b"]
    assert np.allclose(m.predict_proba(X), m2.predict_proba(X))


# --- Métricas y líneas base -----------------------------------------------------
def test_metricas_ejemplo_a_mano():
    y = np.array([0, 0, 1, 1, 1, 2])
    yhat = np.array([0, 1, 1, 1, 0, 1])
    cm = M.matriz_confusion(y, yhat, 3)
    assert cm.tolist() == [[1, 1, 0], [1, 2, 0], [0, 1, 0]]
    r = M.metricas(y, yhat, 3)
    assert r["exactitud"] == pytest.approx(3 / 6)
    # clase 0: P=1/2, R=1/2, F1=1/2; clase 1: P=2/4, R=2/3, F1=4/7; clase 2: 0
    assert r["f1_0"] == pytest.approx(0.5)
    assert r["f1_1"] == pytest.approx(4 / 7)
    assert r["f1_2"] == 0.0
    assert r["f1_macro"] == pytest.approx((0.5 + 4 / 7 + 0) / 3)


def test_lineas_base():
    assert M.base_persistencia([2, 0, 1]).tolist() == [2, 0, 1]
    assert M.base_mayoritaria(np.array([1, 1, 0, 2, 1]), 3).tolist() == [1, 1, 1]


# --- Variables --------------------------------------------------------------------
def diario_sintetico(dias=40, estaciones=("A", "B"), semilla=3):
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2024-01-01", periods=dias, freq="D")
    filas = [{"estacion": e, "fecha": f, "pm25": float(rng.uniform(5, 60)), "valido": True}
             for e in estaciones for f in fechas]
    return pd.DataFrame(filas)


def test_clase_modelo_agrupa_en_tres():
    assert [F.clase_modelo(v) for v in (5, 12, 12.5, 37, 37.2, 80, 300)] == [0, 0, 1, 1, 2, 2, 2]


def test_rezagos_alineados_por_dia_calendario():
    d = diario_sintetico()
    ds = F.construir_dataset(d)
    fila = ds[(ds["estacion"] == "A") & (ds["fecha_base"] == "2024-01-20")].iloc[0]
    serie = d[d["estacion"] == "A"].set_index("fecha")["pm25"]
    assert np.expm1(fila["lag1"]) == pytest.approx(serie["2024-01-20"])
    assert np.expm1(fila["lag3"]) == pytest.approx(serie["2024-01-18"])
    assert fila["pm25_manana"] == pytest.approx(serie["2024-01-21"])
    assert fila["fecha_objetivo"] == pd.Timestamp("2024-01-21")
    assert fila["dia_6"] == 1.0          # 21 ene 2024 fue domingo


def test_sin_fuga_de_informacion():
    """Cambiar el dato del día t+1 no debe cambiar ninguna variable de la fila t."""
    d = diario_sintetico()
    t, t1 = pd.Timestamp("2024-01-20"), pd.Timestamp("2024-01-21")
    d2 = d.copy()
    d2.loc[d2["fecha"] == t1, "pm25"] = 999.0
    a = F.construir_dataset(d).set_index(["estacion", "fecha_base"])
    b = F.construir_dataset(d2).set_index(["estacion", "fecha_base"])
    for est in ("A", "B"):
        assert np.allclose(a.loc[(est, t), F.COLUMNAS_X].to_numpy(float),
                           b.loc[(est, t), F.COLUMNAS_X].to_numpy(float))
        assert b.loc[(est, t), "y"] == 2       # la etiqueta sí cambia


def test_dia_faltante_no_se_usa_como_hoy_y_rezagos_rellenan():
    d = diario_sintetico()
    hueco = pd.Timestamp("2024-01-15")
    d.loc[(d["estacion"] == "A") & (d["fecha"] == hueco), "valido"] = False
    ds = F.construir_dataset(d)
    a = ds[ds["estacion"] == "A"]
    assert hueco not in set(a["fecha_base"])                     # sin dato de hoy no hay fila
    fila = a[a["fecha_base"] == "2024-01-16"].iloc[0]           # lag2 = 15 ene, rellenado con 14 ene
    serie = d[d["estacion"] == "A"].set_index("fecha")["pm25"]
    assert np.expm1(fila["lag2"]) == pytest.approx(serie["2024-01-14"])
    # El día anterior al hueco no tiene etiqueta (mañana no es válido).
    assert np.isnan(a[a["fecha_base"] == "2024-01-14"].iloc[0]["y"])
