"""Entrena y evalúa el clasificador de la Parte 1.

Uso (desde la raíz del proyecto, después de `python -m src.data process`):

    python -m src.train              # selección, entrenamiento final, métricas y guardado
    python -m src.train --sklearn    # además, prueba de cordura contra scikit-learn

Partición temporal por fecha_objetivo (el día que se predice):
    desarrollo  2019-2023   (selección de hiperparámetros y entrenamiento final)
    prueba      2024-01 a 2026-09
El plan original decía "último año completo" (2025), pero 2025 tiene solo 7
días-estación de la clase 2; por eso la prueba abarca desde 2024 y además se
reporta por año.

Selección de l2 y pesos de clase: validación temporal con ventana creciente.
Para cada año de validación v en 2021, 2022 y 2023 se entrena con los años
2019..v-1 y se predice v. El criterio es la F1 macro de las tres validaciones
juntas (223 casos de la clase 2), más estable que la de un solo año. El modelo
final se reentrena con 2019-2023 y se evalúa una sola vez en prueba.

Historia (para el informe): la primera versión elegía con un solo año de
validación (2023, 57 casos de la clase 2) y sin el esquema de pesos "raiz".
Se cambió después de ver la prueba una vez, porque esa validación era
demasiado pequeña para elegir con estabilidad; el cambio usa solo datos de
desarrollo. Resultados de ambas versiones en el README.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from src import config as cfg
from src import features as F
from src import model as M

ANIOS_VALIDACION = [2021, 2022, 2023]
ANIO_INI_TEST = 2024
GRILLA = {"l2": [1e-4, 1e-3, 1e-2, 1e-1], "pesos_clase": list(M.ESQUEMAS_PESOS)}
K = len(F.CLASES)
RUTA_MODELO = cfg.MODELS / "softmax_pm25"


def particionar(ds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ds = ds[ds["y"].notna()].copy()
    ds["y"] = ds["y"].astype(int)
    anio = ds["fecha_objetivo"].dt.year
    return {"desarrollo": ds[anio < ANIO_INI_TEST], "test": ds[anio >= ANIO_INI_TEST]}


def evaluar_metodos(parte: pd.DataFrame, modelo, y_train) -> dict[str, np.ndarray]:
    n = len(parte)
    return {
        "softmax": modelo.predict(F.matriz_X(parte)),
        "persistencia": M.base_persistencia(parte["clase_hoy"].to_numpy()),
        "mayoritaria": M.base_mayoritaria(y_train, n),
    }


def seleccionar(dev: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Validación temporal con ventana creciente sobre los años de desarrollo."""
    anio = dev["fecha_objetivo"].dt.year
    pliegues = [(dev[anio < v], dev[anio == v]) for v in ANIOS_VALIDACION]
    filas = []
    for l2, pesos in itertools.product(GRILLA["l2"], GRILLA["pesos_clase"]):
        y_all, yhat_all, fila = [], [], {"l2": l2, "pesos_clase": pesos or "ninguno"}
        for v, (tr, va) in zip(ANIOS_VALIDACION, pliegues):
            m = M.RegresionSoftmax(l2=l2, pesos_clase=pesos).fit(
                F.matriz_X(tr), tr["y"].to_numpy(), K)
            yhat = m.predict(F.matriz_X(va))
            y_all.append(va["y"].to_numpy())
            yhat_all.append(yhat)
            fila[f"f1_macro_{v}"] = M.metricas(y_all[-1], yhat, K)["f1_macro"]
            fila[f"convergio_{v}"] = m.convergio_
        r = M.metricas(np.concatenate(y_all), np.concatenate(yhat_all), K)
        fila.update({"f1_macro_val": r["f1_macro"], "exactitud_val": r["exactitud"],
                     "f1_2_val": r["f1_2"], "sensibilidad_2_val": r["sensibilidad_2"]})
        filas.append(fila)
        por_anio = " ".join("{:.3f}".format(fila["f1_macro_" + str(v)]) for v in ANIOS_VALIDACION)
        print(f"  l2={l2:<7g} pesos={pesos or 'ninguno':<10} F1 macro val={r['f1_macro']:.3f}  "
              f"exactitud={r['exactitud']:.3f}  F1 clase 2={r['f1_2']:.3f}  (por año: {por_anio})")
    tabla = pd.DataFrame(filas)
    mejor = tabla.loc[tabla["f1_macro_val"].idxmax()]
    elegido = {"l2": float(mejor["l2"]),
               "pesos_clase": None if mejor["pesos_clase"] == "ninguno" else mejor["pesos_clase"]}
    return elegido, tabla


def bootstrap_vs_base(parte: pd.DataFrame, yhat_modelo, yhat_base, n_rep: int = 2000,
                      semilla: int = 0) -> pd.DataFrame:
    """Diferencia modelo - línea base con bootstrap por bloques de 7 días.
    Se remuestrean semanas completas (todas las estaciones) para respetar la
    autocorrelación temporal y la correlación entre estaciones del mismo día."""
    y = parte["y"].to_numpy()
    dias = (parte["fecha_objetivo"] - parte["fecha_objetivo"].min()).dt.days.to_numpy()
    semana = dias // 7
    bloques = [np.flatnonzero(semana == b) for b in np.unique(semana)]
    rng = np.random.default_rng(semilla)
    claves = ["f1_macro", "exactitud", "f1_2"]
    dif = {k: [] for k in claves}
    for _ in range(n_rep):
        idx = np.concatenate([bloques[i] for i in rng.integers(0, len(bloques), len(bloques))])
        a, b = M.metricas(y[idx], yhat_modelo[idx], K), M.metricas(y[idx], yhat_base[idx], K)
        for k in claves:
            dif[k].append(a[k] - b[k])
    a, b = M.metricas(y, yhat_modelo, K), M.metricas(y, yhat_base, K)
    return pd.DataFrame([{
        "metrica": k, "diferencia": a[k] - b[k],
        "ic95_inf": np.percentile(dif[k], 2.5), "ic95_sup": np.percentile(dif[k], 97.5),
        "prob_modelo_mejor": float(np.mean(np.array(dif[k]) > 0)),
        "n_semanas": len(bloques), "n_rep": n_rep,
    } for k in claves])


def tabla_metricas(partes: dict[str, pd.DataFrame], preds: dict[str, dict]) -> pd.DataFrame:
    filas = []
    for periodo, parte in partes.items():
        y = parte["y"].to_numpy()
        for metodo, yhat in preds[periodo].items():
            filas.append({"periodo": periodo, "metodo": metodo, **M.metricas(y, yhat, K)})
    return pd.DataFrame(filas)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sklearn", action="store_true", help="comparar con scikit-learn")
    a = ap.parse_args(argv)

    diario = pd.read_csv(cfg.DATA_PROCESSED / "pm25_diario.csv", parse_dates=["fecha"])
    ds = F.construir_dataset(diario)
    p = particionar(ds)
    for nombre, parte in p.items():
        c = np.bincount(parte["y"], minlength=K)
        print(f"{nombre:<5} {len(parte):>6} filas  "
              f"{parte['fecha_objetivo'].min().date()} a {parte['fecha_objetivo'].max().date()}  "
              f"clases={c.tolist()}")

    print(f"\nSelección de hiperparámetros (ventana creciente, validación {ANIOS_VALIDACION}):")
    elegido, tabla_sel = seleccionar(p["desarrollo"])
    print(f"Elegido: {elegido}")

    trval = p["desarrollo"]
    y_trval = trval["y"].to_numpy()
    modelo = M.RegresionSoftmax(**elegido).fit(F.matriz_X(trval), y_trval, K)
    print(f"Modelo final: {modelo.n_iter_} iteraciones, convergió={modelo.convergio_}")

    test = p["test"]
    anio = test["fecha_objetivo"].dt.year
    partes = {"test": test, **{f"test_{a}": test[anio == a] for a in sorted(anio.unique())}}
    preds = {k: evaluar_metodos(v, modelo, y_trval) for k, v in partes.items()}
    metr = tabla_metricas(partes, preds)

    # --- Guardado ------------------------------------------------------------
    cfg.MODELS.mkdir(parents=True, exist_ok=True)
    modelo.guardar(RUTA_MODELO, {
        "columnas_x": F.COLUMNAS_X, "clases": F.CLASES,
        "entrenado_con": f"{trval['fecha_objetivo'].min().date()} a {trval['fecha_objetivo'].max().date()}",
        "n_entrenamiento": int(len(trval)),
        "seleccion": f"F1 macro, validación temporal con ventana creciente en {ANIOS_VALIDACION}",
        "fecha": datetime.now().isoformat(timespec="seconds"),
    })
    metr.to_csv(cfg.MODELS / "metricas.csv", index=False)
    tabla_sel.to_csv(cfg.MODELS / "seleccion_hiperparametros.csv", index=False)
    y_test = test["y"].to_numpy()
    filas_cm = []
    for metodo, yhat in preds["test"].items():
        cm = M.matriz_confusion(y_test, yhat, K)
        for i in range(K):
            filas_cm.append({"metodo": metodo, "real": F.CLASES[i],
                             **{f"pred_{F.CLASES[j]}": int(cm[i, j]) for j in range(K)}})
    pd.DataFrame(filas_cm).to_csv(cfg.MODELS / "matrices_confusion_test.csv", index=False)

    # --- Resumen ---------------------------------------------------------------
    cols = ["periodo", "metodo", "n", "exactitud", "f1_macro", "f1_0", "f1_1", "f1_2",
            "sensibilidad_2", "n_real_2"]
    print("\nMétricas en prueba:")
    print(metr[cols].round(3).to_string(index=False))
    print("\nMatriz de confusión, prueba completa (filas = real, columnas = predicho):")
    for metodo, yhat in preds["test"].items():
        print(f"  {metodo}:\n{M.matriz_confusion(y_test, yhat, K)}")
    boot = bootstrap_vs_base(test, preds["test"]["softmax"], preds["test"]["persistencia"])
    boot.to_csv(cfg.MODELS / "bootstrap_vs_persistencia.csv", index=False)
    print("\nSoftmax - persistencia en prueba (bootstrap por semanas, IC 95 %):")
    for _, r in boot.iterrows():
        print(f"  {r['metrica']:<10} {r['diferencia']:+.3f}  [{r['ic95_inf']:+.3f}, {r['ic95_sup']:+.3f}]"
              f"  P(modelo mejor)={r['prob_modelo_mejor']:.2f}")
    print(f"\nGuardado en {cfg.MODELS}")

    if a.sklearn:
        from src.sanity_sklearn import comparar
        comparar(modelo, F.matriz_X(trval), y_trval, F.matriz_X(test), y_test, K)


if __name__ == "__main__":
    main(sys.argv[1:])
