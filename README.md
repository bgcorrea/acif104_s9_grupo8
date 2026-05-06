# Clasificador de Rendimiento de Cultivos
### ACIF104 — Aprendizaje de Máquinas | UNAB

Proyecto de clasificación multiclase que predice el rendimiento de cultivos (**Bajo / Medio / Alto**) a partir de variables agronómicas usando modelos ML clásicos, Deep Learning y técnicas de balanceo de clases.

---

## Estructura del Repositorio

```
acif104_s9_grupo8/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── crop_yield_data.csv          # Dataset original (3 000 filas)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb    # Notebook principal (EDA → modelos → SHAP)
├── src/
│   ├── preprocessing.py             # Carga, discretización, split, balanceo
│   ├── models.py                    # Definición modelos ML y DL
│   ├── evaluate.py                  # Métricas, gráficos, SHAP
│   └── app.py                       # Aplicación Streamlit
└── outputs/
    ├── figures/                     # Todas las figuras PNG (generadas por el notebook)
    └── results/                     # CSVs con métricas + modelo .joblib
```

---

## Requisitos

- Python 3.10+
- Las versiones exactas están en `requirements.txt`

Dependencias principales:
- `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`
- `tensorflow` (CPU)
- `shap`, `streamlit`
- `matplotlib`, `seaborn`, `plotly`

---

## Instalación

```bash
git clone <URL-del-repo>
cd crop-yield-classifier

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Ejecución

### 1. Notebook principal (EDA + Modelos + SHAP)

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

Ejecutar todas las celdas en orden. El notebook genera automáticamente:
- Todas las figuras en `outputs/figures/`
- Tablas de métricas en `outputs/results/`
- Modelo entrenado: `outputs/results/mejor_modelo.joblib`

> **Nota:** El entrenamiento de los modelos DL puede tomar 5-10 minutos en CPU.

### 2. Aplicación Streamlit

Primero ejecutar el notebook para generar el modelo, luego:

```bash
streamlit run src/app.py
```

La app abre automáticamente en `http://localhost:8501`

---

## Dataset

**Archivo:** `data/crop_yield_data.csv`  
**Filas:** 3 000 | **Columnas:** 6 | **Sin valores nulos**

| Feature | Rango | Descripción |
|---------|-------|-------------|
| `rainfall_mm` | 500–2 000 | Precipitación anual (mm) |
| `soil_quality_index` | 1–10 | Índice de calidad del suelo |
| `farm_size_hectares` | 10–1 000 | Tamaño de la granja (ha) |
| `sunlight_hours` | 4–12 | Horas de sol diarias |
| `fertilizer_kg` | 100–3 000 | Fertilizante utilizado (kg) |
| `crop_yield` | 46–628 | **Target** (rendimiento, continuo) |

**Discretización del target:**
- **Bajo** → crop_yield < percentil 33
- **Medio** → percentil 33 ≤ crop_yield ≤ percentil 66
- **Alto** → crop_yield > percentil 66

---

## Metodología

```
EDA → Discretización → Split (70/15/15) → Estandarización
  ↓
Balanceo: SMOTE | Undersampling | Class Weights
  ↓
ML Clásico: Logistic Regression | Random Forest | SVM
  ↓
Deep Learning: MLP Simple | MLP Profundo | MLP Regularizado
  ↓
Comparación → GridSearch → Evaluación en Test → SHAP
```

---

## Resultados Principales

| Modelo | Balanceo | F1-Macro (Val) |
|--------|----------|---------------|
| Random Forest | SMOTE | ~0.99 |
| Random Forest | Class Weights | ~0.99 |
| MLP Profundo | SMOTE | ~0.99 |
| Logistic Regression | SMOTE | ~0.98 |
| SVM | SMOTE | ~0.98 |

> El rendimiento alto se debe a la alta correlación de `farm_size_hectares` con el target (r=0.989).

**Feature más importante (SHAP):** `farm_size_hectares` → domina la predicción.  
**Segunda:** `fertilizer_kg` (r=0.75 con residuales).  
**Tercera:** `rainfall_mm` (r=0.58 con residuales).

---

## Aplicación Web

La app Streamlit ofrece:
- **Predicción individual** con sliders y probabilidades por clase
- **Explicabilidad SHAP** (waterfall plot por predicción)
- **Monitoreo del modelo** (métricas, matriz de confusión, feature importance)
- **Predicción batch** (cargar CSV → descargar resultados)
