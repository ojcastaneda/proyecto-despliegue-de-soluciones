from humor_detection.encoder import (
    classification_model,
    detection_model,
    load_model,
    save_model,
)
from humor_detection.test import test_classification, test_detection
from humor_detection.train import train_classification, train_detection
from humor_detection.predict import predict_classification, predict_detection
from transformers.training_args import TrainingArguments

# Nombre del modelo en HuggingFace
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
# Ajustes de trainer de Transformers https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer#transformers.TrainingArguments
#  Lo más importante es usar bf16 o fp16 para VRAM, batch_sizes para la velocidad y train_epochs para los epochs
arguments = TrainingArguments(
    bf16=True,
    bf16_full_eval=True,
    eval_strategy="epoch",
    num_train_epochs=2,
    optim="adamw_8bit",
    per_device_eval_batch_size=150,
    per_device_train_batch_size=150,
    save_strategy="no"
)
# Generar el modelo acepta el mismo LoRA usado en PLN  https://huggingface.co/docs/transformers/v4.51.3/en/peft#peft
# lora = LoraConfig("CAUSAL_LM", lora_alpha=16, lora_dropout=0.1, r=128)

# Prompts por si quieren verificar algo manualmente
prompts = [
    "¿Cuál es el último animal que subió al arca de Noé? El del-fin.",
    "El otro día unas chicas llamarón a mi puerta y me pidieron una pequeña donación para una piscina local.\nLes di un garrafa de agua.",
    "The brain surgeon changed my life. He really opened my mind.",
    "djasndoasndoa",
    "jajaja",
]

# Tarea de clasificación 1 a 5 (Los labels son 0 a 4).
def run_classification():
    # Función para crear el modelo, tokenizador y añadir un lora si es necesario
    model, tokenizer = classification_model(model_name)
    # Entrenamiento con datos en español, Con full_dataset=True entrenan el modelo final, english_data=True añade el dataset en inglés 
    # y prompter permite usar una función para añadir modificar el texto de cada chiste
    train_classification(model, tokenizer, arguments)
    # Recolección de datos de test con dataset hecho por nosotros
    print(test_classification(model, tokenizer, arguments))
    path = f"./models/bert-nlptown/classification"
    # Función para guardar el modelo
    save_model(model, path)
    # Función para cargar el modelo
    model, _ = load_model(model_name, path)
    # Predicción manual de prompts
    print(predict_classification(model, tokenizer, prompts, arguments))

# Tarea de detección 0 o 1.
def run_detection():
    model, tokenizer = detection_model(model_name)
    train_detection(model, tokenizer, arguments)
    print(test_detection(model, tokenizer, arguments))
    path = f"./models/bert-nlptown/detection"
    save_model(model, path)
    model, _ = load_model(model_name, path)
    print(predict_detection(model, tokenizer, prompts, arguments))


if __name__ == "__main__":
    run_classification()
    run_detection()
