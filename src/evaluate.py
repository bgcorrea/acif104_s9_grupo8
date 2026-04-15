"""
evaluate.py
Métricas, gráficos comparativos y análisis SHAP.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

FIGURAS_DIR = Path(__file__).parent.parent / "outputs" / "figures"
RESULTADOS_DIR = Path(__file__).parent.parent / "outputs" / "results"
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

PALETA = "Set2"
DPI = 150
NOMBRES_CLASES = ["Alto", "Bajo", "Medio"]   # orden LabelEncoder


# ─────────────────────────────────────────────
#  MÉTRICAS
# ─────────────────────────────────────────────

def calcular_metricas(y_true, y_pred, y_prob=None) -> dict:
    """Calcula métricas de clasificación multiclase."""
    metricas = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob is not None:
        try:
            metricas["auc_ovr"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except Exception:
            metricas["auc_ovr"] = np.nan
    else:
        metricas["auc_ovr"] = np.nan
    return metricas


# ─────────────────────────────────────────────
#  MATRIZ DE CONFUSIÓN
# ─────────────────────────────────────────────

def graficar_confusion(y_true, y_pred, titulo: str, nombre_archivo: str):
    """Guarda la matriz de confusión como PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=NOMBRES_CLASES, yticklabels=NOMBRES_CLASES, ax=ax
    )
    ax.set_title(titulo, fontsize=13)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    plt.tight_layout()
    ruta = FIGURAS_DIR / nombre_archivo
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    return ruta


# ─────────────────────────────────────────────
#  GRÁFICOS COMPARATIVOS
# ─────────────────────────────────────────────

def graficar_comparacion_f1(df_resultados: pd.DataFrame, nombre_archivo: str = "comparison_f1_scores.png"):
    """Gráfico de barras agrupado comparando F1-macro de todos los modelos."""
    fig, ax = plt.subplots(figsize=(14, 6))
    modelos = df_resultados["modelo"].unique()
    balanceos = df_resultados["balanceo"].unique()
    x = np.arange(len(modelos))
    ancho = 0.25
    colores = plt.get_cmap(PALETA)(np.linspace(0, 1, len(balanceos)))

    for i, bal in enumerate(balanceos):
        sub = df_resultados[df_resultados["balanceo"] == bal].set_index("modelo")
        vals = [sub.loc[m, "f1_macro"] if m in sub.index else 0 for m in modelos]
        ax.bar(x + i * ancho, vals, ancho, label=bal, color=colores[i])

    ax.set_xticks(x + ancho)
    ax.set_xticklabels(modelos, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1-Score Macro")
    ax.set_title("Comparación F1-Score Macro por Modelo y Técnica de Balanceo")
    ax.legend(title="Balanceo")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    ruta = FIGURAS_DIR / nombre_archivo
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    return ruta


def graficar_ml_vs_dl(df_resultados: pd.DataFrame, nombre_archivo: str = "comparison_ml_vs_dl.png"):
    """Boxplot ML vs DL según F1-macro."""
    df = df_resultados.copy()
    df["tipo"] = df["modelo"].apply(
        lambda m: "Deep Learning" if "MLP" in m else "ML Clásico"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    colores_tipo = {"ML Clásico": "#4C72B0", "Deep Learning": "#DD8452"}
    for tipo, sub in df.groupby("tipo"):
        ax.scatter(
            [tipo] * len(sub), sub["f1_macro"],
            alpha=0.7, s=80, color=colores_tipo[tipo], label=tipo
        )
    medias = df.groupby("tipo")["f1_macro"].mean()
    ax.bar(medias.index, medias.values, alpha=0.25,
           color=[colores_tipo[t] for t in medias.index])
    ax.set_ylabel("F1-Score Macro")
    ax.set_title("ML Clásico vs Deep Learning — Distribución de F1")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    ruta = FIGURAS_DIR / nombre_archivo
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    return ruta


def graficar_curvas_entrenamiento(historiales: dict, nombre_archivo: str = "dl_training_curves.png"):
    """
    historiales: {nombre_modelo: history_object}
    Genera subplot con loss y accuracy por modelo DL.
    """
    n = len(historiales)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]
    colores = plt.get_cmap("tab10")(np.linspace(0, 1, n))

    for idx, (nombre, hist) in enumerate(historiales.items()):
        color = colores[idx]
        ax_loss, ax_acc = axes[idx]

        epochs = range(1, len(hist.history["loss"]) + 1)
        ax_loss.plot(epochs, hist.history["loss"], label="Train", color=color)
        ax_loss.plot(epochs, hist.history["val_loss"], label="Val",
                     color=color, linestyle="--")
        ax_loss.set_title(f"{nombre} — Loss")
        ax_loss.set_xlabel("Época")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()

        ax_acc.plot(epochs, hist.history["accuracy"], label="Train", color=color)
        ax_acc.plot(epochs, hist.history["val_accuracy"], label="Val",
                    color=color, linestyle="--")
        ax_acc.set_title(f"{nombre} — Accuracy")
        ax_acc.set_xlabel("Época")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()

    plt.tight_layout()
    ruta = FIGURAS_DIR / nombre_archivo
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    return ruta


# ─────────────────────────────────────────────
#  SHAP
# ─────────────────────────────────────────────

def analizar_shap(modelo, X_train, X_val, nombres_features: list,
                  nombre_modelo: str = "mejor_modelo"):
    """
    Genera gráficos SHAP para el modelo dado.
    Soporta modelos sklearn y Keras (wrapping con función de predicción).
    """
    import shap
    shap.initjs()

    es_keras = hasattr(modelo, "predict_proba") is False and hasattr(modelo, "predict")
    if hasattr(modelo, "predict_proba"):
        # Modelos sklearn
        explainer = shap.Explainer(modelo.predict_proba, X_train, feature_names=nombres_features)
    else:
        # Modelos Keras: wrapping
        def pred_fn(x):
            return modelo.predict(x, verbose=0)
        explainer = shap.Explainer(pred_fn, X_train, feature_names=nombres_features)

    shap_values = explainer(X_val[:200])

    # 1. Summary plot
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(shap_values, X_val[:200], feature_names=nombres_features,
                      show=False, plot_type="bar")
    plt.title(f"SHAP — Importancia Global de Features ({nombre_modelo})")
    plt.tight_layout()
    ruta_summary = FIGURAS_DIR / f"shap_summary_{nombre_modelo}.png"
    plt.savefig(ruta_summary, dpi=DPI, bbox_inches="tight")
    plt.close()

    # 2. Dependence plots top 3 features
    # Determinar top 3 por importancia media
    mean_abs = np.abs(shap_values.values).mean(axis=(0, 2)) if shap_values.values.ndim == 3 \
        else np.abs(shap_values.values).mean(axis=0)
    top3 = np.argsort(mean_abs)[::-1][:3]
    for rank, feat_idx in enumerate(top3):
        fname = nombres_features[feat_idx]
        fig, ax = plt.subplots(figsize=(7, 4))
        if shap_values.values.ndim == 3:
            sv = shap_values.values[:, feat_idx, :]
            shap.dependence_plot(
                feat_idx,
                shap_values.values[:, :, 0],
                X_val[:200],
                feature_names=nombres_features,
                ax=ax, show=False,
            )
        else:
            shap.dependence_plot(
                feat_idx, shap_values.values, X_val[:200],
                feature_names=nombres_features, ax=ax, show=False,
            )
        ax.set_title(f"SHAP Dependence — {fname}")
        plt.tight_layout()
        ruta_dep = FIGURAS_DIR / f"shap_dependence_{rank+1}_{fname}.png"
        fig.savefig(ruta_dep, dpi=DPI)
        plt.close(fig)

    # 3. Force / waterfall plots para 3 ejemplos individuales
    for i in range(3):
        fig, ax = plt.subplots(figsize=(10, 3))
        shap.plots.waterfall(shap_values[i], max_display=5, show=False)
        plt.title(f"SHAP Waterfall — Ejemplo {i+1}")
        plt.tight_layout()
        ruta_wf = FIGURAS_DIR / f"shap_waterfall_ejemplo_{i+1}.png"
        plt.savefig(ruta_wf, dpi=DPI, bbox_inches="tight")
        plt.close()

    return shap_values
