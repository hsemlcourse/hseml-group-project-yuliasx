from __future__ import annotations

import pathlib
import pickle
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

SEED = 42
MODELS_DIR = pathlib.Path(__file__).resolve().parents[1] / "models"


# Оценка качества
def clf_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    metrics = {
        f"{prefix}accuracy":  accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall":    recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}f1":        f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}roc_auc":   roc_auc_score(y_true, y_prob),
        f"{prefix}pr_auc":    average_precision_score(y_true, y_prob),
    }
    return metrics

# Оценка модели на всех выборках
def full_eval(
    model: Any,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    threshold: float = 0.5,
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    for split_name, X, y in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)
        results.update(clf_metrics(y, pred, prob, prefix=f"{split_name}_"))

    # отчет по тесту
    prob_test = model.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= threshold).astype(int)
    results["test_confusion_matrix"] = confusion_matrix(y_test, pred_test)
    results["test_report"] = classification_report(y_test, pred_test, target_names=["alive", "bankrupt"])
    results["test_fpr"], results["test_tpr"], _ = roc_curve(y_test, prob_test)

    return results


# Модели
def train_logistic_baseline(
    X_train, y_train, X_val, y_val, X_test, y_test,
    C: float = 1.0,
) -> dict[str, Any]:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=SEED, solver="lbfgs"))
    ])
    pipeline.fit(X_train, y_train)

    res = full_eval(pipeline, X_train, y_train, X_val, y_val, X_test, y_test)
    res["model"] = pipeline

    if hasattr(X_train, "columns"):
        coefs = pd.Series(pipeline.named_steps["logreg"].coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
        res["coef"] = coefs

    return res

def train_polynomial_logistic(
    X_train, y_train, X_val, y_val, X_test, y_test,
    degree: int = 2, C: float = 1.0,
) -> dict[str, Any]:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("logreg", LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=SEED, solver="lbfgs"))
    ])
    pipeline.fit(X_train, y_train)

    res = full_eval(pipeline, X_train, y_train, X_val, y_val, X_test, y_test)
    res["model"] = pipeline

    if hasattr(X_train, "columns"):
        poly_features = pipeline.named_steps["poly"].get_feature_names_out(X_train.columns)
        coefs = pd.Series(pipeline.named_steps["logreg"].coef_[0], index=poly_features).sort_values(key=abs, ascending=False)
        res["coef"] = coefs

    return res

def train_pca_logistic(
    X_train, y_train, X_val, y_val, X_test, y_test,
    n_components: int | float = 0.95, C: float = 1.0,
) -> dict[str, Any]:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=SEED)),
        ("logreg", LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=SEED, solver="lbfgs"))
    ])
    pipeline.fit(X_train, y_train)

    res = full_eval(pipeline, X_train, y_train, X_val, y_val, X_test, y_test)
    res["model"] = pipeline
    return res

# Подбор порога
def find_best_threshold(y_val: np.ndarray, prob_val: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        pred = (prob_val >= t).astype(int)
        score = f1_score(y_val, pred, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return float(best_t)


# Сохранение и загрузка
def save_model(model: Any, name: str) -> pathlib.Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Сохранено в {path}")
    return path

def load_model(name: str) -> Any:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
