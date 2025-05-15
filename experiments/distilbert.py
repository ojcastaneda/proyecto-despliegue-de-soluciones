from humor_detection.encoder import classification_model, detection_model
from humor_detection.test import (
    test_classification,
    test_detection,
    test_exclusive,
    test_lengths,
    test_repetition,
)
from humor_detection.train import train_classification, train_detection
from humor_detection.predict import predict_classification, predict_detection
from humor_detection.utils import relative_path, set_random_seeds
from pprint import pprint
from transformers.training_args import TrainingArguments
# from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

# Nombre del modelo en HuggingFace
model_name = "distilbert/distilbert-base-multilingual-cased"
# Carpeta para agrupar y guardar modelos de clasificación y detección
save_path = relative_path("../models/distilbert")
# Argumentos que nunca van a cambiar en entrenamiento/pruebas, dependen de sus GPUs
default_arguments = {
    "bf16": True,
    "bf16_full_eval": True,
    "disable_tqdm": False,
    "per_device_eval_batch_size": 20,
    "per_device_train_batch_size": 30,
}
# Prompts para predicciones
prompts = [
    "- Martínez, queda usted despedido.\n- Pero, si yo no he hecho nada.\n- Por eso, por eso.",  # Humor alto
    "¿Cuál es el último animal que subió al arca de Noé? El del-fin.",  # Humor
    "El otro día unas chicas llamarón a mi puerta y me pidieron una pequeña donación para una piscina local.\nLes di un garrafa de agua.",  # Humor
    "The brain surgeon changed my life. He really opened my mind.",  # No humor
    "djasndoasndoa",  # No humor
    "jajaja",  # No humor
]


# Tarea de clasificación 1 a 5 (Los labels son 0 a 4).
def run_classification(full_dataset: bool):
    # Cambiar semilla random
    set_random_seeds()
    # Función para crear el modelo, tokenizador y añadir un lora si es necesario
    model, tokenizer = classification_model(model_name)
    # Ajustes de trainer de Transformers https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer#transformers.TrainingArguments
    #  Lo más importante es usar bf16 o fp16 para VRAM, batch_sizes para la velocidad y train_epochs para los epochs
    arguments = TrainingArguments(
        num_train_epochs=10,
        lr_scheduler_type="cosine_with_restarts",
        max_grad_norm=0.01,
        **default_arguments,
    )
    # Entrenamiento con datos en español, Con full_dataset=True entrenan el modelo final, english_data=True añade el dataset en inglés
    train_logs, metrics = train_classification(
        model,
        tokenizer,
        arguments,
        full_dataset=full_dataset,  # Entrenamiento final
        english_data=False,  # Usar dataset de StupidStuff (incluso con traducciones no parece ser buena idea)
        class_weights=[1, 1.25, 1.25, 2, 4],  # Pesos de clases para desbalance
        sample=False,  # Parámetro par "under" o "over" sample, por el momento no se puede modificar el factor de mágnitud así que no da buenos resultados,
        best_model_metric="macro_f1",  # "macro_f1" por defecto, "weighted_f1" o "accuracy" para guardar el mejor epoch del modelo con la mejor métrica seleccionada
        save_path=(
            f"{save_path}/classification" if full_dataset else None
        ),  # Dónde guardar los logs de entrenamiento y el modelo final entrenado, puede ser None si no es necesario
    )
    pprint(train_logs)
    pprint(metrics)
    # Función para cargar el modelo guardado
    # model, _ = load_model(model_name, f"{save_path}/classification")
    if not full_dataset:
        # Recolección de datos de test con dataset hecho por nosotros en el caso de full_dataset ya se hace en train
        pprint(test_classification(model, tokenizer, arguments))
    # Predicción manual de prompts
    pprint(predict_classification(model, tokenizer, prompts, arguments))
    # Función para guardar el modelo manualmente (Borra las carpetas antiguas por lo que puede eliminar las métricas)
    # save_model(model, path)


def run_detection(full_dataset: bool, threshold: float | None):
    set_random_seeds()
    model, tokenizer = detection_model(model_name)
    arguments = TrainingArguments(
        num_train_epochs=3,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"num_cycles": 0.7, "min_lr": 1e-5},
        **default_arguments,
    )
    train_logs, metrics = train_detection(
        model,
        tokenizer,
        arguments,
        full_dataset=full_dataset,
        class_weights=[1.1, 1],
        threshold=threshold,  # Threshold para predecir humor más alto requiere máyor probabiliad (0.75 es un buen valor pero depende del modelo)
        save_path=f"{save_path}/detection" if full_dataset else None,
    )
    pprint(train_logs)
    pprint(metrics)
    if not full_dataset:
        pprint(test_detection(model, tokenizer, arguments, threshold=threshold))
    if full_dataset:
        pprint(test_exclusive(model, tokenizer, arguments, threshold=threshold))
        pprint(test_lengths(model, tokenizer, arguments, threshold=threshold))
        pprint(test_repetition(model, tokenizer, arguments, threshold=threshold))
    pprint(predict_detection(model, tokenizer, prompts, arguments, threshold=threshold))


if __name__ == "__main__":
    run_classification(True)
    run_detection(True, None)
