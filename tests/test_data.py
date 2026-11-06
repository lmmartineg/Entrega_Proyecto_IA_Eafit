"""Pruebas de la limpieza con datos SINTÉTICOS (no reales).

Verifican la lógica, no el esquema de la API: los nombres de columna de abajo
son inventados para la prueba.
"""
import numpy as np
import pandas as pd
import pytest

from src import data, ica

MAPEO = {"fecha": "fh", "estacion": "est", "valor": "conc", "variable": "var",
         "latitud": "lat", "longitud": "lon"}


def crudo_sintetico(horas_dia2: int = 10) -> pd.DataFrame:
    """Estación A: día 1 completo (24 h, valor 10), día 2 con pocas horas.
    Incluye una fila de otro contaminante, una negativa y una con coma decimal."""
    filas = []
    for h in range(24):
        filas.append({"fh": f"2024-01-01T{h:02d}:00:00", "est": "A", "conc": "10", "var": "PM2.5",
                      "lat": "6.2", "lon": "-75.5"})
    for h in range(horas_dia2):
        filas.append({"fh": f"2024-01-02T{h:02d}:00:00", "est": "A", "conc": "20", "var": "PM2.5",
                      "lat": "6.2", "lon": "-75.5"})
    filas.append({"fh": "2024-01-01T05:00:00", "est": "A", "conc": "99", "var": "O3", "lat": "6.2", "lon": "-75.5"})
    filas.append({"fh": "2024-01-01T06:00:00", "est": "B", "conc": "-3", "var": "PM2.5", "lat": "6.3", "lon": "-75.6"})
    filas.append({"fh": "2024-01-01T07:00:00", "est": "B", "conc": "12,5", "var": "PM2.5", "lat": "6.3", "lon": "-75.6"})
    return pd.DataFrame(filas)


def test_limpieza_filtra_variable_negativos_y_coma():
    h = data.limpiar_horario(crudo_sintetico(), MAPEO, variable="PM2.5")
    assert set(h.columns) >= {"estacion", "fecha", "pm25", "latitud", "longitud"}
    assert (h["pm25"] >= 0).all()
    assert 99 not in h["pm25"].values           # fila de O3 descartada
    assert 12.5 in h["pm25"].values             # coma decimal convertida
    assert len(h[h["estacion"] == "A"]) == 34   # 24 + 10 horas


def test_promedio_diario_regla_75():
    h = data.limpiar_horario(crudo_sintetico(horas_dia2=10), MAPEO, variable="PM2.5")
    d = data.promedio_diario(h[h["estacion"] == "A"], min_frac=0.75)
    dia1 = d[d["fecha"] == "2024-01-01"].iloc[0]
    dia2 = d[d["fecha"] == "2024-01-02"].iloc[0]
    assert dia1["pm25"] == pytest.approx(10.0) and dia1["valido"]
    assert dia2["n_horas"] == 10 and not dia2["valido"]   # 10 < ceil(24*0.75)=18


def test_promedio_diario_borde_18_horas():
    h = data.limpiar_horario(crudo_sintetico(horas_dia2=18), MAPEO, variable="PM2.5")
    d = data.promedio_diario(h[h["estacion"] == "A"], min_frac=0.75)
    assert d[d["fecha"] == "2024-01-02"].iloc[0]["valido"]  # 18 h exactas = válido


def test_reporte_unidades_marca_mediana_alta():
    fechas = pd.date_range("2021-01-01", periods=48, freq="h")
    h = pd.DataFrame({"estacion": ["X"] * 48 + ["Y"] * 48,
                      "fecha": list(fechas) * 2,
                      "pm25": [15.0] * 48 + [1800.0] * 48})
    r = data.reporte_unidades(h, umbral=100)
    assert r.set_index("estacion").loc["Y", "sospechoso"]
    assert not r.set_index("estacion").loc["X", "sospechoso"]


def test_ica_vacia_falla_con_mensaje():
    with pytest.raises(ValueError, match="Tabla 6"):
        ica.categoria(10.0, tabla=[])


def test_ica_con_tabla_ficticia():
    # Tabla INVENTADA solo para probar la lógica; no son los cortes oficiales.
    t = [(0, 10, "A", "c1"), (10.1, 20, "B", "c2"), (20.1, 1000, "C", "c3")]
    assert ica.categoria(5, t) == "A"
    assert ica.categoria(10.05, t) == "B"   # hueco por redondeo -> rango superior
    assert ica.categoria(500, t) == "C"


def test_ica_oficial_bordes():
    # Tabla 6 de la Res. 2254: los límites superiores son inclusivos y los huecos van arriba.
    ica.validar_tabla(ica.ICA_PM25)
    assert ica.categoria(0) == "Buena"
    assert ica.categoria(12) == "Buena"
    assert ica.categoria(12.4) == "Aceptable"
    assert ica.categoria(37) == "Aceptable"        # 37 = máximo permisible diario
    assert ica.categoria(37.01) == "Dañina a la salud de grupos sensibles"
    assert ica.categoria(150) == "Dañina a la salud"
    assert ica.categoria(600) == "Peligrosa"
    assert ica.color("Aceptable") == "Amarillo"


def test_ica_coherente_con_limite_diario():
    from src import config as cfg
    aceptable = next(t for t in ica.ICA_PM25 if t[2] == "Aceptable")
    assert aceptable[1] == cfg.LIMITE_PM25_24H


def test_niveles_episodio():
    assert ica.nivel_episodio(37) is None
    assert ica.nivel_episodio(37.5) == "Prevención"
    assert ica.nivel_episodio(55) == "Prevención"
    assert ica.nivel_episodio(55.2) == "Alerta"
    assert ica.nivel_episodio(150) == "Alerta"
    assert ica.nivel_episodio(151) == "Emergencia"
    assert ica.nivel_episodio(900) == "Emergencia"


def escribir_tab(ruta, texto):
    ruta.write_text(texto.replace("|", "\t"), encoding="utf-8")


def test_leer_siata_ancho_a_largo(tmp_path):
    # Dos meses sintéticos con el formato del SIATA; la estación C solo aparece en el 2.º.
    escribir_tab(tmp_path / "Estaciones_PM2.5_2024_01.tab",
                 "fecha_hora|A|B\n"
                 "2024-01-31 22:00:00|10.5|\n"
                 "2024-01-31 23:00:00|11|7\n")
    escribir_tab(tmp_path / "Estaciones_PM2.5_2024_02.tab",
                 "fecha_hora|A|C\n"
                 "2024-02-01 00:00:00|12|3.25\n")
    (tmp_path / "_manifiesto.json").write_text("{}")   # no debe leerse
    largo = data.leer_siata(tmp_path)
    assert set(largo.columns) == {"fecha", "estacion", "valor"}
    h = data.limpiar_horario(largo, data.MAPEO_LARGO, variable="")
    assert set(h["estacion"]) == {"A", "B", "C"}
    assert len(h) == 5                                   # la celda vacía de B se descarta
    assert h.loc[h["estacion"] == "C", "pm25"].iloc[0] == pytest.approx(3.25)
    assert h["fecha"].dtype.kind == "M"


def test_leer_siata_sin_archivos(tmp_path):
    with pytest.raises(SystemExit, match="download_siata"):
        data.leer_siata(tmp_path)


def test_leer_siata_sin_columna_fecha(tmp_path):
    escribir_tab(tmp_path / "Estaciones_PM2.5_2024_01.tab", "fecha|A\n2024-01-01 00:00:00|1\n")
    with pytest.raises(ValueError, match="fecha_hora"):
        data.leer_siata(tmp_path)


def test_sugerir_mapeo():
    s = data.sugerir_mapeo(["fecha", "nombre_estacion", "concentracion", "variable", "latitud"])
    assert s["fecha"] == "fecha" and s["estacion"] == "nombre_estacion"
    assert s["valor"] == "concentracion" and s["longitud"] is None
