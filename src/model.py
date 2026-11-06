"""Parte 1: regresión logística multinomial (softmax) implementada en NumPy.

Pérdida: entropía cruzada media (opcionalmente ponderada por clase) más
penalización L2 sobre los pesos (no sobre el sesgo):

    J(W, b) = sum_i s_i * CE_i / sum_i s_i  +  (l2 / 2) * ||W||^2

Optimización: descenso de gradiente con aceleración de Nesterov y paso fijo
1/L, donde L es la cota de Böhning de la curvatura de la softmax:
L = 0.5 * lambda_max(Z' S Z / sum s) + l2. Con ese paso el descenso no
diverge, sin tener que ajustar a mano la tasa de aprendizaje.

Las variables se estandarizan con la media y la desviación del conjunto de
entrenamiento, que se guardan con el modelo.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EPS = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


ESQUEMAS_PESOS = (None, "raiz", "balanceado")


def pesos_clase(y: np.ndarray, n_clases: int, esquema: str | None) -> np.ndarray:
    """Peso de cada clase, normalizado para que el peso medio por muestra sea 1.

    None         -> 1 para todas las clases.
    "balanceado" -> n / (K * n_k), igual que class_weight="balanced" de sklearn.
    "raiz"       -> raíz cuadrada del balanceado: compromiso entre los dos.
    """
    if esquema not in ESQUEMAS_PESOS:
        raise ValueError(f"Esquema de pesos desconocido: {esquema}")
    if esquema is None:
        return np.ones(n_clases)
    conteo = np.bincount(y, minlength=n_clases).astype(float)
    if (conteo == 0).any():
        raise ValueError(f"Hay clases sin ejemplos en el entrenamiento: {conteo}")
    w = len(y) / (n_clases * conteo)
    if esquema == "raiz":
        w = np.sqrt(w)
    return w * len(y) / (w * conteo).sum()


def pesos_muestra(y: np.ndarray, n_clases: int, esquema: str | None) -> np.ndarray:
    return pesos_clase(np.asarray(y, dtype=int), n_clases, esquema)[np.asarray(y, dtype=int)]


class RegresionSoftmax:
    def __init__(self, l2: float = 1e-3, pesos_clase: str | None = None,
                 max_iter: int = 10000, tol: float = 1e-6):
        self.l2 = l2
        self.pesos_clase = pesos_clase
        self.max_iter = max_iter
        self.tol = tol

    # --- utilidades ----------------------------------------------------------
    def _estandarizar(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.media_) / self.escala_

    def perdida_y_gradiente(self, W, b, Z, Y, s):
        """Pérdida J y gradientes (dJ/dW, dJ/db). Pública para la prueba de gradiente."""
        P = softmax(Z @ W + b)
        stot = s.sum()
        ce = -(s * np.log(P[Y.astype(bool)] + EPS)).sum() / stot
        J = ce + 0.5 * self.l2 * (W ** 2).sum()
        G = (P - Y) * (s / stot)[:, None]
        return J, Z.T @ G + self.l2 * W, G.sum(axis=0)

    # --- entrenamiento -------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, n_clases: int | None = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_clases_ = int(n_clases or y.max() + 1)
        self.media_ = X.mean(axis=0)
        self.escala_ = X.std(axis=0)
        self.escala_[self.escala_ == 0] = 1.0
        Z = self._estandarizar(X)
        n, p = Z.shape
        K = self.n_clases_
        Y = np.zeros((n, K))
        Y[np.arange(n), y] = 1.0
        s = pesos_muestra(y, K, self.pesos_clase)

        # Paso 1/L con la cota de Böhning (columna de unos = sesgo).
        Z1 = np.hstack([Z, np.ones((n, 1))])
        L = 0.5 * np.linalg.eigvalsh((Z1 * (s / s.sum())[:, None]).T @ Z1).max() + self.l2
        paso = 1.0 / L

        W = np.zeros((p, K))
        b = np.zeros(K)
        W_ant, b_ant = W.copy(), b.copy()
        self.historia_ = []
        for t in range(1, self.max_iter + 1):
            # Nesterov: gradiente en el punto "adelantado".
            mom = (t - 1) / (t + 2)
            Wv = W + mom * (W - W_ant)
            bv = b + mom * (b - b_ant)
            J, gW, gb = self.perdida_y_gradiente(Wv, bv, Z, Y, s)
            W_ant, b_ant = W, b
            W = Wv - paso * gW
            b = bv - paso * gb
            self.historia_.append(J)
            norma = np.sqrt((gW ** 2).sum() + (gb ** 2).sum())
            if norma < self.tol:
                break
        self.W_, self.b_ = W, b
        self.n_iter_ = t
        self.convergio_ = bool(norma < self.tol)
        return self

    # --- predicción ----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self._estandarizar(X) @ self.W_ + self.b_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # --- persistencia --------------------------------------------------------
    def guardar(self, ruta: Path | str, metadatos: dict | None = None) -> None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ruta.with_suffix(".npz"), W=self.W_, b=self.b_,
                 media=self.media_, escala=self.escala_)
        meta = {"l2": self.l2, "pesos_clase": self.pesos_clase, "n_clases": self.n_clases_,
                "n_iter": self.n_iter_, "convergio": self.convergio_, **(metadatos or {})}
        ruta.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                             encoding="utf-8")

    @classmethod
    def cargar(cls, ruta: Path | str) -> tuple["RegresionSoftmax", dict]:
        ruta = Path(ruta)
        meta = json.loads(ruta.with_suffix(".json").read_text(encoding="utf-8"))
        arr = np.load(ruta.with_suffix(".npz"))
        m = cls(l2=meta["l2"], pesos_clase=meta["pesos_clase"])
        m.W_, m.b_, m.media_, m.escala_ = arr["W"], arr["b"], arr["media"], arr["escala"]
        m.n_clases_ = meta["n_clases"]
        return m, meta


# --- Líneas base ---------------------------------------------------------------
def base_persistencia(clase_hoy: np.ndarray) -> np.ndarray:
    """Mañana tendrá la misma clase que hoy."""
    return np.asarray(clase_hoy, dtype=int)


def base_mayoritaria(y_train: np.ndarray, n: int) -> np.ndarray:
    """Siempre la clase más frecuente del entrenamiento."""
    return np.full(n, np.bincount(np.asarray(y_train, dtype=int)).argmax())


# --- Métricas (implementación propia) -----------------------------------------
def matriz_confusion(y: np.ndarray, yhat: np.ndarray, n_clases: int) -> np.ndarray:
    """Filas = clase real, columnas = clase predicha."""
    m = np.zeros((n_clases, n_clases), dtype=int)
    np.add.at(m, (np.asarray(y, dtype=int), np.asarray(yhat, dtype=int)), 1)
    return m


def metricas(y: np.ndarray, yhat: np.ndarray, n_clases: int) -> dict:
    """Exactitud, F1 macro y precisión/sensibilidad/F1 por clase. Si un
    denominador es 0, la métrica vale 0 (igual que zero_division=0 en sklearn)."""
    m = matriz_confusion(y, yhat, n_clases)
    vp = np.diag(m).astype(float)
    pred = m.sum(axis=0)
    real = m.sum(axis=1)
    prec = np.divide(vp, pred, out=np.zeros_like(vp), where=pred > 0)
    sens = np.divide(vp, real, out=np.zeros_like(vp), where=real > 0)
    f1 = np.divide(2 * prec * sens, prec + sens, out=np.zeros_like(vp), where=(prec + sens) > 0)
    out = {"n": int(m.sum()), "exactitud": vp.sum() / m.sum(), "f1_macro": f1.mean()}
    for k in range(n_clases):
        out[f"precision_{k}"] = prec[k]
        out[f"sensibilidad_{k}"] = sens[k]
        out[f"f1_{k}"] = f1[k]
        out[f"n_real_{k}"] = int(real[k])
    return out
