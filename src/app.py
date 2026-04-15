"""
app.py
Aplicación Streamlit — Clasificador de Rendimiento de Cultivos
Frontend + Backend con predicción individual, explicabilidad SHAP,
monitoreo de métricas y predicción batch.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap
import io
import streamlit as st
from pathlib import Path

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "results" / "mejor_modelo.joblib"
META_PATH  = BASE_DIR / "outputs" / "results" / "model_meta.joblib"

# ─── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Cultivos",
    page_icon="🌾",
    layout="wide",
)

# ─── Estilos personalizados ───────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card { background:#f0f2f6; border-radius:8px; padding:12px; text-align:center; }
    .pred-badge-Alto  { background:#2ecc71; color:white; padding:6px 16px; border-radius:20px; font-size:1.2rem; }
    .pred-badge-Medio { background:#f39c12; color:white; padding:6px 16px; border-radius:20px; font-size:1.2rem; }
    .pred-badge-Bajo  { background:#e74c3c; color:white; padding:6px 16px; border-radius:20px; font-size:1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Carga de modelo y metadatos ──────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    if not MODEL_PATH.exists():
        return None, None
    modelo = joblib.load(MODEL_PATH)
    meta   = joblib.load(META_PATH) if META_PATH.exists() else {}
    return modelo, meta


modelo, meta = cargar_modelo()
FEATURES = ["rainfall_mm", "soil_quality_index", "farm_size_hectares",
            "sunlight_hours", "fertilizer_kg"]
LABELS  = ["Alto", "Bajo", "Medio"]   # orden LabelEncoder


def predecir_individual(vals: np.ndarray):
    """Devuelve (clase_str, probs_array)."""
    if modelo is None:
        return "Sin modelo", np.array([0.33, 0.33, 0.33])
    prob = modelo.predict_proba(vals.reshape(1, -1))[0]
    pred = modelo.predict(vals.reshape(1, -1))[0]
    return LABELS[pred], prob


def shap_waterfall_fig(vals: np.ndarray):
    """Genera figura matplotlib del SHAP waterfall para un punto."""
    try:
        scaler = meta.get("scaler")
        X_train_sc = meta.get("X_train_sc")
        if X_train_sc is None or scaler is None:
            return None
        x_sc = scaler.transform(vals.reshape(1, -1))
        explainer  = shap.Explainer(modelo.predict_proba, X_train_sc[:100],
                                    feature_names=FEATURES)
        shap_vals  = explainer(x_sc)
        fig, ax = plt.subplots(figsize=(9, 3))
        shap.plots.waterfall(shap_vals[0], max_display=5, show=False)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.warning(f"No se pudo generar SHAP: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/color/96/wheat.png", width=80)
st.sidebar.title("Navegación")
pagina = st.sidebar.radio(
    "",
    ["🔮 Predicción Individual", "📊 Monitoreo del Modelo", "📂 Predicción Batch"],
)

if modelo is None:
    st.sidebar.warning(
        "⚠️ Modelo no encontrado.\n\n"
        "Ejecuta el notebook `01_eda_and_modeling.ipynb` primero para entrenar y guardar el modelo."
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Proyecto:** ACIF104 — Aprendizaje de Máquinas  \n"
    "**Dataset:** Crop Yield (3 000 filas)  \n"
    "**Modelo:** Random Forest + SMOTE"
)

# ══════════════════════════════════════════════════════════════════════════════
#  PÁGINA 1 — PREDICCIÓN INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🔮 Predicción Individual":
    st.title("🌾 Clasificador de Rendimiento de Cultivos")
    st.markdown("Ajusta los parámetros del campo para obtener la predicción de rendimiento.")

    col_inputs, col_resultado = st.columns([1, 1], gap="large")

    with col_inputs:
        st.subheader("Parámetros del Campo")
        rainfall     = st.slider("Lluvia (mm)", 500, 2000, 1200, step=10,
                                  help="Precipitación anual en milímetros")
        soil_quality = st.slider("Calidad del Suelo (1-10)", 1, 10, 5,
                                  help="Índice de calidad del suelo")
        farm_size    = st.slider("Tamaño de la Granja (ha)", 10, 1000, 300, step=10,
                                  help="Tamaño en hectáreas")
        sunlight     = st.slider("Horas de Sol (h/día)", 4, 12, 8,
                                  help="Promedio de horas de sol diarias")
        fertilizer   = st.slider("Fertilizante (kg)", 100, 3000, 1000, step=50,
                                  help="Cantidad de fertilizante utilizado")

        predecir_btn = st.button("🚀 Predecir", type="primary", use_container_width=True)

    with col_resultado:
        st.subheader("Resultado")
        if predecir_btn or True:   # Mostrar siempre con valores por defecto
            vals_raw = np.array([rainfall, soil_quality, farm_size, sunlight, fertilizer],
                                 dtype=float)
            # Si hay scaler en meta, escalar
            scaler = meta.get("scaler") if meta else None
            vals_sc = scaler.transform(vals_raw.reshape(1, -1))[0] if scaler else vals_raw

            clase, probs = predecir_individual(vals_sc)

            colores_clase = {"Alto": "🟢", "Medio": "🟡", "Bajo": "🔴"}
            st.markdown(
                f"### Clase predicha: {colores_clase.get(clase, '')} **{clase}**"
            )
            st.progress(float(probs.max()), text=f"Confianza: {probs.max():.1%}")

            # Probabilidades por clase
            df_probs = pd.DataFrame({
                "Clase": LABELS,
                "Probabilidad": probs,
            }).sort_values("Probabilidad", ascending=False)

            fig_prob, ax = plt.subplots(figsize=(5, 3))
            colores = ["#2ecc71" if c == "Alto" else "#f39c12" if c == "Medio" else "#e74c3c"
                       for c in df_probs["Clase"]]
            ax.barh(df_probs["Clase"], df_probs["Probabilidad"], color=colores)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilidad")
            ax.set_title("Distribución de Probabilidades")
            for i, (_, row) in enumerate(df_probs.iterrows()):
                ax.text(row["Probabilidad"] + 0.01, i, f"{row['Probabilidad']:.2%}",
                        va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_prob, use_container_width=True)
            plt.close(fig_prob)

    # SHAP explicación
    st.markdown("---")
    st.subheader("🔍 Explicabilidad SHAP")
    st.markdown("Contribución de cada feature a la predicción individual.")

    with st.spinner("Calculando SHAP..."):
        vals_raw_shap = np.array([rainfall, soil_quality, farm_size, sunlight, fertilizer],
                                  dtype=float)
        fig_shap = shap_waterfall_fig(vals_raw_shap)
        if fig_shap:
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)
        else:
            st.info("SHAP no disponible — ejecuta el notebook para guardar los datos de entrenamiento en el modelo.")

# ══════════════════════════════════════════════════════════════════════════════
#  PÁGINA 2 — MONITOREO DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Monitoreo del Modelo":
    st.title("📊 Monitoreo del Modelo")

    if meta:
        metricas = meta.get("metricas_test", {})

        # KPIs
        kpi_cols = st.columns(4)
        kpi_data = [
            ("Accuracy", metricas.get("accuracy", 0), "%.4f"),
            ("F1 Macro", metricas.get("f1_macro", 0), "%.4f"),
            ("Precision Macro", metricas.get("precision_macro", 0), "%.4f"),
            ("AUC OvR", metricas.get("auc_ovr", 0), "%.4f"),
        ]
        for col, (nombre, valor, fmt) in zip(kpi_cols, kpi_data):
            col.metric(nombre, fmt % valor)

        st.markdown("---")

        col_cm, col_dist = st.columns(2)

        # Matriz de confusión
        with col_cm:
            st.subheader("Matriz de Confusión (Test)")
            cm = meta.get("confusion_matrix_test")
            if cm is not None:
                import seaborn as sns
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=LABELS, yticklabels=LABELS, ax=ax)
                ax.set_xlabel("Predicción")
                ax.set_ylabel("Real")
                ax.set_title("Confusion Matrix — Test Set")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # Distribución de predicciones en test
        with col_dist:
            st.subheader("Distribución de Predicciones")
            y_pred_test = meta.get("y_pred_test")
            if y_pred_test is not None:
                uniq, cnt = np.unique(y_pred_test, return_counts=True)
                df_dist = pd.DataFrame({"Clase": [LABELS[u] for u in uniq], "Conteo": cnt})
                colores_dist = {"Alto": "#2ecc71", "Medio": "#f39c12", "Bajo": "#e74c3c"}
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(df_dist["Clase"], df_dist["Conteo"],
                       color=[colores_dist.get(c, "#4C72B0") for c in df_dist["Clase"]])
                ax.set_ylabel("Conteo")
                ax.set_title("Distribución de Predicciones en Test Set")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # Reporte de clasificación
        st.markdown("---")
        st.subheader("Reporte de Clasificación Detallado")
        from sklearn.metrics import classification_report
        y_true = meta.get("y_test")
        y_pred = meta.get("y_pred_test")
        if y_true is not None and y_pred is not None:
            reporte = classification_report(y_true, y_pred,
                                             target_names=LABELS, output_dict=True)
            df_reporte = pd.DataFrame(reporte).T
            st.dataframe(df_reporte.style.format("{:.3f}"), use_container_width=True)

        # Feature importances (solo para RF)
        st.markdown("---")
        st.subheader("Importancia de Features")
        if hasattr(modelo, "feature_importances_"):
            fi = modelo.feature_importances_
            df_fi = pd.DataFrame({"Feature": FEATURES, "Importancia": fi})
            df_fi = df_fi.sort_values("Importancia", ascending=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(df_fi["Feature"], df_fi["Importancia"],
                    color=plt.get_cmap("Set2")(np.linspace(0, 1, len(FEATURES))))
            ax.set_xlabel("Importancia")
            ax.set_title("Importancia de Features — Random Forest")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        st.info("Ejecuta el notebook para generar los metadatos del modelo.")

# ══════════════════════════════════════════════════════════════════════════════
#  PÁGINA 3 — PREDICCIÓN BATCH
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📂 Predicción Batch":
    st.title("📂 Predicción Batch")
    st.markdown(
        "Carga un archivo CSV con columnas: "
        "`rainfall_mm`, `soil_quality_index`, `farm_size_hectares`, "
        "`sunlight_hours`, `fertilizer_kg`"
    )

    uploaded = st.file_uploader("Subir CSV", type=["csv"])
    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)
            st.write(f"**{len(df_input)} filas cargadas.** Vista previa:")
            st.dataframe(df_input.head(5), use_container_width=True)

            faltantes = [c for c in FEATURES if c not in df_input.columns]
            if faltantes:
                st.error(f"Columnas faltantes: {faltantes}")
            else:
                X_batch = df_input[FEATURES].values
                scaler = meta.get("scaler") if meta else None
                if scaler:
                    X_batch = scaler.transform(X_batch)

                if modelo is not None:
                    preds = modelo.predict(X_batch)
                    probs = modelo.predict_proba(X_batch)
                    df_result = df_input.copy()
                    df_result["clase_predicha"] = [LABELS[p] for p in preds]
                    df_result["prob_Alto"]  = probs[:, 0]
                    df_result["prob_Bajo"]  = probs[:, 1]
                    df_result["prob_Medio"] = probs[:, 2]

                    st.success(f"✅ Predicciones completadas para {len(df_result)} registros.")
                    st.dataframe(df_result, use_container_width=True)

                    # Distribución de predicciones batch
                    fig, ax = plt.subplots(figsize=(6, 3))
                    conteo = pd.Series([LABELS[p] for p in preds]).value_counts()
                    colores_batch = {"Alto": "#2ecc71", "Medio": "#f39c12", "Bajo": "#e74c3c"}
                    ax.bar(conteo.index, conteo.values,
                           color=[colores_batch.get(c, "gray") for c in conteo.index])
                    ax.set_title("Distribución de Predicciones")
                    ax.set_ylabel("Cantidad")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    # Descarga
                    csv_bytes = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Descargar resultados CSV",
                        data=csv_bytes,
                        file_name="predicciones_batch.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.error("Modelo no disponible. Ejecuta el notebook primero.")
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
