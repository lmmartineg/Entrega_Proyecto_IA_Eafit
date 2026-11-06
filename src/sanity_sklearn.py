"""Prueba de cordura: misma regresión softmax con scikit-learn.

Solo verifica la implementación propia (src/model.py); no se usa para
predecir. Se entrena con las mismas variables estandarizadas, la misma
penalización y los mismos pesos de clase.

Equivalencia de la penalización: sklearn minimiza C * sum_i s_i CE_i + ||W||^2 / 2.
Dividiendo entre C * sum_i s_i queda la pérdida de src/model.py con
l2 = 1 / (C * sum_i s_i), es decir, C = 1 / (l2 * sum_i s_i). Los pesos de
src/model.py están normalizados (peso medio 1), así que sum_i s_i = n con
cualquier esquema; a sklearn se le pasan esos mismos pesos por clase.
"""
from __future__ import annotations

import numpy as np

from src import model as M


def comparar(propio: M.RegresionSoftmax, Xtr, ytr, Xte, yte, n_clases: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    Ztr, Zte = propio._estandarizar(Xtr), propio._estandarizar(Xte)
    w = M.pesos_clase(np.asarray(ytr, dtype=int), n_clases, propio.pesos_clase)
    C = 1.0 / (propio.l2 * w[np.asarray(ytr, dtype=int)].sum())
    sk = LogisticRegression(C=C, class_weight={k: float(w[k]) for k in range(n_clases)},
                            max_iter=5000, tol=1e-8)
    sk.fit(Ztr, ytr)

    p_prop, p_sk = propio.predict_proba(Xte), sk.predict_proba(Zte)
    yhat_prop, yhat_sk = p_prop.argmax(1), p_sk.argmax(1)
    # La softmax no es identificable (sumar una constante a todas las clases no
    # cambia nada): se comparan los pesos centrados por fila.
    Wp = propio.W_ - propio.W_.mean(axis=1, keepdims=True)
    Ws = sk.coef_.T - sk.coef_.T.mean(axis=1, keepdims=True)
    r = {
        "acuerdo_predicciones": float((yhat_prop == yhat_sk).mean()),
        "max_dif_probabilidad": float(np.abs(p_prop - p_sk).max()),
        "corr_pesos": float(np.corrcoef(Wp.ravel(), Ws.ravel())[0, 1]),
        "max_dif_pesos": float(np.abs(Wp - Ws).max()),
        "f1_macro_propio": float(M.metricas(yte, yhat_prop, n_clases)["f1_macro"]),
        "f1_macro_sklearn_metrica_propia": float(M.metricas(yte, yhat_sk, n_clases)["f1_macro"]),
        "f1_macro_sklearn_metrica_sklearn": float(f1_score(yte, yhat_sk, average="macro")),
    }
    print(f"\nPrueba de cordura contra scikit-learn (mismos datos, l2={propio.l2:g}, "
          f"pesos={propio.pesos_clase or 'ninguno'}):")
    for k, v in r.items():
        print(f"  {k:<34} {v:.6f}")
    return r
