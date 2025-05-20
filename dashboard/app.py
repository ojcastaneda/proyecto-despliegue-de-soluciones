import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch
torch.classes.__path__ = []

from typing import Callable
from humor_detection.encoder import load_model as encoder_load_model
from humor_detection.decoder import load_model as decoder_load_model
from humor_detection.predict import predict_classification
from humor_detection.predict import predict_detection
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import streamlit as st
import matplotlib.pyplot as plt


MODELS_PATH = './models'

MODEL_NAMES = {
    'encoder': [
        "distilbert",
        "canine",
        "dccuchile"
    ],
    'decoder': [
        "distilgpt2",
        "distilgpt2-finetuned",
        "DeepSeek-R1-Distill-Qwen-1.5B",
        "bloom-1b1-lora"
        "gemini",
        "distilbert-base-multilingual-cased-sentiments-student",
        "xlmr-large-toxicity-classifier"
    ]
}

PROMPT_TEMPLATES = {
    'classification': """<INPUT>
========================================
- Question:
How funny is this text?
1) Slightly
2) Mildly
3) Moderately
4) Very
5) Incredibly
The answer is """,
    'detection': """Detect if the following text is funny 1 or not 0 following the provided examples.

Text: Un piano me caería excelente en estos momentos.
Score: 1

Text: Ni Jesús te ama.
Score: 0

Text: Jajajajajajajaj Idiota.
Score: 0

Text: —¿Qué es eso que traes en tu bolsa?
—Un AK-47.
—No, al lado del AK-47.
—Unos Chettos bolita.
—¡No puedes entrar al Cine con comida!
Score: 1

Text: <INPUT>
Score: """
}


st.set_page_config(page_title="MAIA Despliegue de Proyectos", layout="wide")

st.title("Clasificador de textos de humor en español")
st.markdown("Ingrese un texto en español para:\n1. Clasificarlo en niveles de humor del 0 al 5.\n2. Detectar humor 1 y no humor 0.")

with st.sidebar:
    model_type = st.selectbox(
        "Seleccionar tipo de modelo",
        options=[
            "encoder",
            "decoder"
        ]
    )

    default_arguments = {
        "bf16": True,
        "bf16_full_eval": True,
        "disable_tqdm": False,
        "per_device_eval_batch_size": 10,
    }

    model_name = st.selectbox(
        "Seleccionar modelo",
        options=MODEL_NAMES[model_type]
    )

    task_type = st.selectbox(
        "Seleccionar tipo de tarea",
        options=[
            "classification",
            "detection"
        ]
    )

    prompt_template = st.text_area("Plantilla del Prompt", value=PROMPT_TEMPLATES[task_type], height=400)

model_path = os.path.join(MODELS_PATH, model_name)

def fix_tokenizer(tokenizer: PreTrainedTokenizerBase):
    tokenizer.pad_token = tokenizer.eos_token

def run_classification(prompter: Callable[[str], str], prompt: str):
    if model_type == "encoder":
        model, tokenizer = encoder_load_model(model_name, model_path)
    elif model_type == "decoder":
        model, tokenizer = decoder_load_model(model_name, model_path)
        fix_tokenizer(tokenizer)
    
    arguments = TrainingArguments(**default_arguments)
    results = predict_classification(model, tokenizer, [prompt], arguments, prompter)
    return results

def run_detection(
    prompter: Callable[[str], str],
    prompt: str,
    threshold: float | None,
):
    if model_type == "encoder":
        model, tokenizer = encoder_load_model(model_name, model_path)
    elif model_type == "decoder":
        model, tokenizer = decoder_load_model(model_name, model_path)
        fix_tokenizer(tokenizer)
    
    arguments = TrainingArguments(
        max_steps=8000,
        eval_steps=1000,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"num_cycles": 0.6, "min_lr": 1e-5},
        **default_arguments,
    )
    results = predict_detection(model, tokenizer, [prompt], arguments, prompter, threshold)
    return results

def generic_prompter(input: str):
     prompt = prompt_template.replace("<INPUT>", input)
     return prompt

input_sentence = st.text_area("Ingresa el texto a evaluar", height=200)

if st.button("Ejecutar inferencia"):
    with st.spinner("Cargando modelo y ejecutando inferencia"):
        results = "OK"
        if task_type == "classification":
            results = run_classification(generic_prompter, input_sentence)
        elif task_type == "detection":
            results = run_detection(generic_prompter, input_sentence, None)

        st.info(str(results))