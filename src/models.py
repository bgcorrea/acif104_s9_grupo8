"""
models.py
Definición de modelos ML clásicos y arquitecturas de Deep Learning.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)


# ─────────────────────────────────────────────
#  MODELOS ML CLÁSICOS
# ─────────────────────────────────────────────

def crear_logistic_regression(class_weight=None):
    """Regresión Logística con regularización L2."""
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
        multi_class="multinomial",
        solver="lbfgs",
    )


def crear_random_forest(class_weight=None):
    """Random Forest Classifier."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
        n_jobs=-1,
    )


def crear_svm(class_weight=None):
    """SVM con kernel RBF."""
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )


# ─────────────────────────────────────────────
#  ARQUITECTURAS DEEP LEARNING
# ─────────────────────────────────────────────

def crear_mlp_simple(n_features: int, n_clases: int = 3):
    """MLP Simple: 1 hidden layer (64 neuronas, ReLU)."""
    modelo = keras.Sequential(
        [
            layers.Input(shape=(n_features,)),
            layers.Dense(64, activation="relu", name="hidden_1"),
            layers.Dense(n_clases, activation="softmax", name="output"),
        ],
        name="MLP_Simple",
    )
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def crear_mlp_profundo(n_features: int, n_clases: int = 3):
    """MLP Profundo: 3 hidden layers (128-64-32) con BatchNorm y Dropout."""
    modelo = keras.Sequential(
        [
            layers.Input(shape=(n_features,)),
            layers.Dense(128, activation="relu", name="hidden_1"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu", name="hidden_2"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu", name="hidden_3"),
            layers.BatchNormalization(),
            layers.Dense(n_clases, activation="softmax", name="output"),
        ],
        name="MLP_Profundo",
    )
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def crear_mlp_regularizado(n_features: int, n_clases: int = 3):
    """MLP con regularización L2 y learning rate scheduling."""
    modelo = keras.Sequential(
        [
            layers.Input(shape=(n_features,)),
            layers.Dense(
                128, activation="relu",
                kernel_regularizer=regularizers.l2(1e-3),
                name="hidden_1",
            ),
            layers.Dense(
                64, activation="relu",
                kernel_regularizer=regularizers.l2(1e-3),
                name="hidden_2",
            ),
            layers.Dense(n_clases, activation="softmax", name="output"),
        ],
        name="MLP_Regularizado",
    )
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def callbacks_entrenamiento(patience: int = 10, reduce_lr: bool = True):
    """Early stopping + reducción de LR opcional."""
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
    ]
    if reduce_lr:
        cbs.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0,
            )
        )
    return cbs
