from streamlit import (
    button,
    columns,
    error,
    text_input,
    number_input,
    selectbox,
    set_page_config,
    spinner,
    success,
    text_area,
    title,
    warning
)
from utils import MODELS, load_models, plot_predictions, predict

set_page_config(page_title="MAIA Despliegue de Proyectos")
title("Clasificador de textos de humor en español")
model_selection, model_threshold = columns([2, 1])
with model_selection:
    model_name = selectbox("Modelo de clasificación", options=MODELS.keys(), index=3)
model = MODELS[model_name]
api_key = None
with model_threshold:
    threshold = number_input(
        "Probabilidad mínima",
        0.0,
        1.0,
        model.threshold,
        disabled=model.file_name == "gemini",
    )
if model.file_name == "gemini":
    api_key = text_input("Llave de acceso a API de Gemini", type="password")
prompt = text_area("Texto a evaluar", height=200)
results = None
with columns([1, 1, 1])[1]:
    if button("Ejecutar inferencia", use_container_width=True):
        with spinner("Ejecutando inferencia"):
            results = predict(*load_models(model_name), prompt, threshold, api_key)
if results is not None:
    if results[0].iloc[0]["labels"] == -1:
        error("Error: el modelo no seleccionó una opción válida.")
    elif results[1] is not None:
        classification_label = results[1].iloc[0]["labels"]
        if classification_label == -1:
            warning("Clasificación: humor, sin puntuación ya que el modelo no seleccionó una opción válida.")
        else:
            success(f"Clasificación: humor, con puntuación de {int(classification_label) + 1}.")
    else:
        success("Clasificación: no humor.")
    plot_predictions(*results)
