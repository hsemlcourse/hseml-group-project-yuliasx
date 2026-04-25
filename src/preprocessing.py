from __future__ import annotations

import pathlib
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "american_bankruptcy.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


TARGET = "status_label"
TARGET_BIN = "bankrupt"          

# Финансовые показатели (стандартные)
RAW_FEATURES = [f"X{i}" for i in range(1, 19)]  

FEATURES = RAW_FEATURES.copy()

SEED = 42


# Загрузка
def load_raw_data(path: str | pathlib.Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    return df


# Кодировка таргета
def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_BIN] = (df[TARGET] == "failed").astype(int)
    return df


# Очистка дубликатов
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"Удалено дубликатов: {n_before - len(df)}")
    return df

# Заполнение пропусков медианой
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("Нет пропусков")
        return df

    print(f"  Пустые значения : {(missing > 0).sum()}")
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df

# Обрезка по квантилям
def clip_outliers(
    df: pd.DataFrame,
    features: list[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    df = df.copy()
    for col in features:
        lo = df[col].quantile(lower_q)
        hi = df[col].quantile(upper_q)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# разбиение на трейн, валидацию и тест
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    strat = y if stratify else None

    # тест
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )

    # трейн и валидация
    val_ratio = val_size / (1.0 - test_size)
    strat_tmp = y_tmp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=val_ratio,
        random_state=seed,
        stratify=strat_tmp,
    )

    print(
        f"  Размеры выборок — train: {len(X_train)}, "
        f"val: {len(X_val)}, test: {len(X_test)}"
    )
    print(
        f"  Доля rate — train: {y_train.mean():.3f}, "
        f"val: {y_val.mean():.3f}, test: {y_test.mean():.3f}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# Полный пайплайн
def build_dataset(
    path: str | pathlib.Path = RAW_DATA_PATH,
    clip: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    print("Загрузка данных...")
    df = load_raw_data(path)
    print(f"  Rows: {len(df)}, Cols: {df.shape[1]}")

    print("Кодировка целевой переменной...")
    df = encode_target(df)

    print("Очистка...")
    df = drop_duplicates(df)
    df = handle_missing(df)

    if clip:
        print("Сглаживание выбросов...")
        df = clip_outliers(df, RAW_FEATURES)

    features = RAW_FEATURES

    X = df[features].copy()
    y = df[TARGET_BIN].copy()

    print(f"  Количество признаков: {len(features)}")
    print(f"  Баланс классов — alive: {(y==0).sum()}, bankrupt: {(y==1).sum()}")

    return X, y, features


# Сохранение
def save_processed(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    for name, obj in splits.items():
        obj.to_pickle(PROCESSED_DIR / f"{name}.pkl")
    print(f"  Сохранено в {PROCESSED_DIR}")

# Загрузка
def load_processed() -> dict:
    names = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
    out = {n: pd.read_pickle(PROCESSED_DIR / f"{n}.pkl") for n in names}
    return out
