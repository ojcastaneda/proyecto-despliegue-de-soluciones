from streamlit import (
    button,
    columns,
    text_input,
    number_input,
    selectbox,
    set_page_config,
    spinner,
    text_area,
    title,
    write,
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
classification_model, detection_model = load_models(model)
prompt = text_area("Texto a evaluar", height=200)
results = None
with columns([1, 1, 1])[1]:
    if button("Ejecutar inferencia", use_container_width=True):
        with spinner("Ejecutando inferencia"):
            results = predict(
                classification_model, detection_model, prompt, threshold, api_key
            )
if results is not None:
    detection_label = results[0].iloc[0]["labels"]
    label = "No humor."
    if detection_label == -1:
        label = "El modelo no seleccionó una opción válida."
    if results[1] is not None:
        classification_label = results[1].iloc[0]["labels"]
        if classification_label == -1:
            label = "Humor, sin puntuación ya que el modelo no seleccionó una opción válida."
        else:
            label = f"Humor, con puntuación de {int(classification_label) + 1}."
    write(label)
    plot_predictions(*results)
