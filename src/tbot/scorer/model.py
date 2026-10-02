"""Capa rápida: responde todas las preguntas fijas en una llamada, con probabilidades calibradas.

Misma interfaz que tendría Jev (`score(snapshot) → {pregunta: {clase: probabilidad}}`), así se puede
reemplazar sin tocar el resto. Implementación: gradient boosting por pregunta + calibración isotónica
por clase sobre un tramo cronológico posterior al de entrenamiento.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from .questions import QUESTIONS


class _FastTrees:
    """Evalúa los árboles de un HistGradientBoostingClassifier ya entrenado con numpy vectorizado.

    Da el mismo resultado que `predict_proba` pero sin el costo fijo de sklearn por llamada (~2-3 ms),
    que con 5 preguntas superaba los 10 ms por vela.
    """

    def __init__(self, model: HistGradientBoostingClassifier):
        preds = model._predictors  # lista [iteración][árbol por clase]
        self.k = len(preds[0])
        self.baseline = np.asarray(model._baseline_prediction, float).reshape(-1)
        feat, thr, left, right, leaf, val, missing_left, roots, cls = [], [], [], [], [], [], [], [], []
        off = 0
        for it in preds:
            for c, tree in enumerate(it):
                n = tree.nodes
                roots.append(off)
                cls.append(c)
                feat.append(n["feature_idx"].astype(np.int64))
                thr.append(n["num_threshold"].astype(float))
                left.append(n["left"].astype(np.int64) + off)
                right.append(n["right"].astype(np.int64) + off)
                leaf.append(n["is_leaf"].astype(bool))
                val.append(n["value"].astype(float))
                missing_left.append(n["missing_go_to_left"].astype(bool))
                off += len(n)
        self.feat, self.thr = np.concatenate(feat), np.concatenate(thr)
        self.left, self.right = np.concatenate(left), np.concatenate(right)
        self.leaf, self.val = np.concatenate(leaf), np.concatenate(val)
        self.missing_left = np.concatenate(missing_left)
        self.roots, self.cls = np.asarray(roots), np.asarray(cls)
        self.depth = int(max(t.nodes["depth"].max() for it in preds for t in it)) + 1

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        idx = self.roots.copy()
        for _ in range(self.depth):
            leaf = self.leaf[idx]
            if leaf.all():
                break
            xv = x[self.feat[idx]]
            go_left = np.where(np.isnan(xv), self.missing_left[idx], xv <= self.thr[idx])
            idx = np.where(leaf, idx, np.where(go_left, self.left[idx], self.right[idx]))
        raw = self.baseline + np.bincount(self.cls, weights=self.val[idx], minlength=self.k)
        if self.k == 1:
            p1 = 1 / (1 + np.exp(-raw[0]))
            return np.array([1 - p1, p1])
        e = np.exp(raw - raw.max())
        return e / e.sum()


class _Head:
    def __init__(self, n_classes: int, seed: int):
        self.n = n_classes
        self.model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=seed)
        self.classes_: np.ndarray | None = None
        self.iso: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n_classes
        self.prior = np.full(n_classes, 1 / n_classes)

    def _raw(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), self.n))
        if self.classes_ is None:
            out[:] = self.prior
            return out
        out[:, self.classes_] = self.model.predict_proba(X)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray, calib_frac: float = 0.25) -> None:
        y = y.astype(int)
        self.prior = np.bincount(y, minlength=self.n) / max(1, len(y))
        n_cal = int(len(X) * calib_frac)
        Xf, yf, Xc, yc = X[: len(X) - n_cal], y[: len(X) - n_cal], X[len(X) - n_cal :], y[len(X) - n_cal :]
        if len(np.unique(yf)) < 2:
            return  # sin variación: usamos la frecuencia observada
        self.model.fit(Xf, yf)
        self.classes_ = self.model.classes_.astype(int)
        self.fast = _FastTrees(self.model)
        if n_cal >= 100:
            raw = self._raw(Xc)
            for k in range(self.n):
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw[:, k], (yc == k).astype(float))
                self.iso[k] = (iso.X_thresholds_, iso.y_thresholds_)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        """Una sola fila, por el camino rápido (mismo resultado que predict)."""
        raw = np.zeros(self.n)
        if self.classes_ is None:
            raw[:] = self.prior
        else:
            raw[self.classes_] = self.fast.predict_proba(x)
        cal = np.array([np.interp(raw[k], *self.iso[k]) if self.iso[k] is not None else raw[k] for k in range(self.n)])
        s = cal.sum()
        return cal / s if s > 0 else self.prior

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw(X)
        cal = np.column_stack([np.interp(raw[:, k], *self.iso[k]) if self.iso[k] is not None else raw[:, k] for k in range(self.n)])
        s = cal.sum(axis=1, keepdims=True)
        return np.where(s > 0, cal / np.where(s > 0, s, 1), self.prior)


class CalibratedScorer:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.heads: dict[str, _Head] = {}

    def fit(self, X: np.ndarray, labels: pd.DataFrame) -> CalibratedScorer:
        for q, classes in QUESTIONS.items():
            y = labels[q].to_numpy()
            ok = np.isfinite(y)
            head = _Head(len(classes), self.seed)
            head.fit(X[ok], y[ok])
            self.heads[q] = head
        return self

    def score_batch(self, X: np.ndarray) -> dict[str, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, float))
        return {q: self.heads[q].predict(X) for q in QUESTIONS}

    def score(self, x: np.ndarray) -> dict[str, dict[str, float]]:
        """Una vela → todas las preguntas, en una llamada (camino rápido)."""
        x = np.asarray(x, float).reshape(-1)
        out = {}
        for q, classes in QUESTIONS.items():
            p = self.heads[q].predict_one(x)
            out[q] = {c: float(p[k]) for k, c in enumerate(classes)}
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> CalibratedScorer:
        return pickle.loads(Path(path).read_bytes())
