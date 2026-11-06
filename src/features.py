"""Variables y etiqueta para predecir la categoría de PM2.5 del día siguiente.

Planteamiento: al final del día t (fecha_base) se predice la clase del día
t+1 (fecha_objetivo) en cada estación. Todas las variables usan solo días
<= t, salvo el calendario de t+1, que se conoce de antemano.

Clases (3, agrupando la Tabla 6 de la Res. 2254 porque las categorías Naranja
y superiores suman solo el 2,6 % de los días de 2019-2026):
    0 = Buena (0-12 µg/m³)
    1 = Aceptable (13-37 µg/m³)
    2 = Dañina para grupos sensibles o peor (> 37 µg/m³)

Faltantes: el dato de hoy (lag1) tiene que ser un día válido real. Para los
rezagos 2-7 y la media de 14 días, un día faltante se rellena con el último
valor válido anterior, hasta 2 días seguidos (solo usa el pasado). Las filas
que siguen incompletas se descartan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import ica

CLASES = ["Buena", "Aceptable", "Dañina para grupos sensibles o peor"]
N_LAGS = 7
VENTANA_MEDIA = 14
RELLENO_MAX_DIAS = 2

COLS_CONC = [f"lag{k}" for k in range(1, N_LAGS + 1)] + [f"media{VENTANA_MEDIA}", "valle1"]
COLS_DIA = [f"dia_{d}" for d in range(1, 7)]        # lunes (0) es la referencia
COLS_MES = [f"mes_{m}" for m in range(2, 13)]       # enero (1) es la referencia
COLUMNAS_X = COLS_CONC + COLS_DIA + COLS_MES


def clase_modelo(concentracion: float) -> int:
    """Índice de clase (0, 1, 2) de una concentración diaria de PM2.5."""
    nombres = [t[2] for t in ica.ICA_PM25]
    return min(nombres.index(ica.categoria(concentracion)), len(CLASES) - 1)


def _a_largo(ancho: pd.DataFrame, nombre: str) -> pd.Series:
    """DataFrame fecha x estación -> Series con el orden de `_llaves`."""
    return pd.Series(ancho.to_numpy().ravel(order="C"), name=nombre)


def _tabla_ancha(diario: pd.DataFrame) -> pd.DataFrame:
    d = diario[diario["valido"]] if "valido" in diario.columns else diario
    w = d.pivot(index="fecha", columns="estacion", values="pm25").sort_index()
    # Fechas calendario completas: un shift de 1 es siempre "1 día", no "1 fila".
    return w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="D"))


def construir_dataset(diario: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (estación, fecha_base) con las variables, la clase de hoy
    (para la línea base de persistencia) y la clase de mañana `y` (NaN si el
    día siguiente no tiene dato válido, p. ej. el último día disponible)."""
    w = _tabla_ancha(diario)
    log_obs = np.log1p(w)
    log_rell = np.log1p(w.ffill(limit=RELLENO_MAX_DIAS))

    variables = {"lag1": log_obs}
    for k in range(2, N_LAGS + 1):
        variables[f"lag{k}"] = log_rell.shift(k - 1)
    variables[f"media{VENTANA_MEDIA}"] = log_rell.rolling(
        VENTANA_MEDIA, min_periods=VENTANA_MEDIA // 2).mean()
    valle = log_obs.mean(axis=1)    # promedio del valle en el día t
    variables["valle1"] = pd.DataFrame({c: valle for c in w.columns}, index=w.index)

    n_fechas, n_est = w.shape
    out = pd.DataFrame({
        "fecha_base": np.repeat(w.index.to_numpy(), n_est),
        "estacion": np.tile(w.columns.to_numpy(), n_fechas),
    })
    for nombre, ancho in variables.items():
        out[nombre] = _a_largo(ancho, nombre).to_numpy()
    out["pm25_hoy"] = _a_largo(w, "pm25_hoy").to_numpy()
    out["pm25_manana"] = _a_largo(w.shift(-1), "pm25_manana").to_numpy()

    out["fecha_objetivo"] = out["fecha_base"] + pd.Timedelta(days=1)
    dia = out["fecha_objetivo"].dt.day_of_week
    mes = out["fecha_objetivo"].dt.month
    for d in range(1, 7):
        out[f"dia_{d}"] = (dia == d).astype(float)
    for m in range(2, 13):
        out[f"mes_{m}"] = (mes == m).astype(float)

    out = out.dropna(subset=COLS_CONC).reset_index(drop=True)
    out["clase_hoy"] = out["pm25_hoy"].map(clase_modelo).astype(int)
    out["y"] = out["pm25_manana"].map(lambda v: np.nan if pd.isna(v) else clase_modelo(v))
    return out


def matriz_X(dataset: pd.DataFrame) -> np.ndarray:
    return dataset[COLUMNAS_X].to_numpy(dtype=float)
