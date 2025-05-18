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
import sys

model_name = "google/canine-c"
save_path = relative_path("../models/canine")
default_arguments = {
    "bf16": True,
    "bf16_full_eval": True,
    "disable_tqdm": False,
    "per_device_eval_batch_size": 20,
    "per_device_train_batch_size": 30,
    "torch_empty_cache_steps": 1,
    "eval_accumulation_steps":1,
}
prompts = [
    "- Martínez, queda usted despedido.\n- Pero, si yo no he hecho nada.\n- Por eso, por eso.",
    "¿Cuál es el último animal que subió al arca de Noé? El del-fin.",
    "El otro día unas chicas llamarón a mi puerta y me pidieron una pequeña donación para una piscina local.\nLes di un garrafa de agua.",
    "The brain surgeon changed my life. He really opened my mind.",
    "djasndoasndoa",
    "jajaja",
]


def run_classification(full_dataset: bool):
    set_random_seeds()
    model, tokenizer = classification_model(model_name)
    arguments = TrainingArguments(
        num_train_epochs=4,
        learning_rate=1,
        **default_arguments,
    )
    train_logs, metrics = train_classification(
        model,
        tokenizer,
        arguments,
        full_dataset=full_dataset,
        save_path=(f"{save_path}/classification" if full_dataset else None),
    )
    pprint(train_logs)
    pprint(metrics)
    if not full_dataset:
        pprint(test_classification(model, tokenizer, arguments))
    pprint(predict_classification(model, tokenizer, prompts, arguments))


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
        threshold=threshold,
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
    if sys.argv[1] == "classification":
        run_classification(True)
    if sys.argv[1] == "detection":
        run_detection(True, None)
