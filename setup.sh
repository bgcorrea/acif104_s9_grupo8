#!/usr/bin/env bash
# setup.sh — Crea el entorno virtual e instala dependencias del proyecto.
set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

echo "==> Verificando Python..."
$PYTHON --version

echo "==> Creando entorno virtual en $VENV_DIR ..."
$PYTHON -m venv "$VENV_DIR"

echo "==> Activando entorno virtual..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Actualizando pip..."
pip install --upgrade pip --quiet

echo "==> Instalando dependencias desde requirements.txt ..."
pip install -r requirements.txt

echo "==> Registrando kernel de Jupyter..."
python -m ipykernel install --user --name=crop-yield --display-name "Python (crop-yield)"

echo ""
echo "✓ Entorno listo."
echo ""
echo "Próximos pasos:"
echo "  1. Activar el entorno:     source .venv/bin/activate"
echo "  2. Entrenar el modelo:     make train"
echo "     (o abre el notebook)   jupyter notebook notebooks/01_eda_and_modeling.ipynb"
echo "  3. Lanzar la app:          make app"
