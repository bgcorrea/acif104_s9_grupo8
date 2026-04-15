VENV   = .venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip
NB     = notebooks/01_eda_and_modeling.ipynb

.PHONY: setup train notebook app clean

## Crea el entorno virtual e instala dependencias
setup:
	bash setup.sh

## Ejecuta el notebook en modo headless (entrena y guarda el modelo)
train:
	$(VENV)/bin/jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=600 \
		--output-dir notebooks \
		--output 01_eda_and_modeling.ipynb \
		$(NB)
	@echo "Modelo guardado en outputs/results/"

## Abre el notebook en el navegador
notebook:
	$(VENV)/bin/jupyter notebook $(NB)

## Lanza la app Streamlit
app:
	$(VENV)/bin/streamlit run src/app.py

## Elimina el entorno virtual y los artefactos generados
clean:
	rm -rf $(VENV) outputs __pycache__ src/__pycache__ .ipynb_checkpoints
