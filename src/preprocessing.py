"""
preprocessing.py
Carga del dataset, discretización del target, split estratificado y técnicas de balanceo.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Semilla global para reproducibilidad
RANDOM_STATE = 42

# Rutas
DATA_PATH = Path(__file__).parent.parent / "data" / "crop_yield_data.csv"


def cargar_datos(ruta: Path = DATA_PATH) -> pd.DataFrame:
    """Carga el dataset desde CSV."""
    df = pd.read_csv(ruta)
    return df


def discretizar_target(df: pd.DataFrame, columna: str = "crop_yield") -> pd.DataFrame:
    """
    Discretiza la columna continua en 3 clases usando percentiles 33 y 66.
    Clases: 'Bajo', 'Medio', 'Alto'
    """
    df = df.copy()
    p33 = df[columna].quantile(0.33)
    p66 = df[columna].quantile(0.66)

    condiciones = [
        df[columna] < p33,
        (df[columna] >= p33) & (df[columna] <= p66),
        df[columna] > p66,
    ]
    etiquetas = ["Bajo", "Medio", "Alto"]
    df["clase"] = np.select(condiciones, etiquetas, default="Medio")

    print(f"Percentil 33: {p33:.1f} | Percentil 66: {p66:.1f}")
    print(f"Distribución de clases:\n{df['clase'].value_counts()}")
    return df, p33, p66


def preparar_splits(df: pd.DataFrame):
    """
    Divide el dataset en train (70%), validation (15%) y test (15%) de forma estratificada.
    Retorna: X_train, X_val, X_test, y_train, y_val, y_test, scaler, le
    """
    features = ["rainfall_mm", "soil_quality_index", "farm_size_hectares",
                 "sunlight_hours", "fertilizer_kg"]
    X = df[features].values
    y = df["clase"].values

    # Codificación de etiquetas
    le = LabelEncoder()
    le.fit(["Alto", "Bajo", "Medio"])   # orden alfabético → 0,1,2
    y_enc = le.transform(y)

    # Split 70/30 → luego 30 en 50/50 (15% val + 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_enc, test_size=0.30, stratify=y_enc, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    # Estandarización (fit solo en train)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    print(f"Train: {X_train_sc.shape[0]} | Val: {X_val_sc.shape[0]} | Test: {X_test_sc.shape[0]}")
    return (X_train_sc, X_val_sc, X_test_sc,
            y_train, y_val, y_test,
            scaler, le, features)


def aplicar_smote(X_train: np.ndarray, y_train: np.ndarray):
    """Oversampling con SMOTE."""
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def aplicar_undersampling(X_train: np.ndarray, y_train: np.ndarray):
    """Undersampling aleatorio."""
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return X_res, y_res


def calcular_class_weights(y_train: np.ndarray) -> dict:
    """Calcula pesos de clase inversamente proporcionales a la frecuencia."""
    clases, conteos = np.unique(y_train, return_counts=True)
    total = len(y_train)
    n_clases = len(clases)
    pesos = {int(c): total / (n_clases * cnt) for c, cnt in zip(clases, conteos)}
    return pesos
