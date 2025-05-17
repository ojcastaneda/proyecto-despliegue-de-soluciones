from typing import Callable
from humor_detection.decoder import classification_model, detection_model
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
from peft.tuners.lora import LoraConfig
from pprint import pprint
from transformers.training_args import TrainingArguments
import sys

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
save_path = relative_path("../models/deepseek-r1-distill-qwen")
default_arguments = {
    "bf16": True,
    "bf16_full_eval": True,
    "disable_tqdm": False,
    "per_device_eval_batch_size": 10,
    "per_device_train_batch_size": 15,
}
prompts = [
    "- Martínez, queda usted despedido.\n- Pero, si yo no he hecho nada.\n- Por eso, por eso.",
    "¿Cuál es el último animal que subió al arca de Noé? El del-fin.",
    "El otro día unas chicas llamarón a mi puerta y me pidieron una pequeña donación para una piscina local.\nLes di un garrafa de agua.",
    "The brain surgeon changed my life. He really opened my mind.",
    "djasndoasndoa",
    "jajaja",
]


def run_classification(full_dataset: bool, train: bool, prompter: Callable[[str], str]):
    set_random_seeds()
    arguments = TrainingArguments(
        num_train_epochs=4,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"num_cycles": 0.8, "min_lr": 1e-5},
        **default_arguments,
    )
    lora = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=128,
        task_type="CAUSAL_LM",
    )
    model, tokenizer = classification_model(model_name, lora_configuration=lora)
    if train:
        train_logs, metrics = train_classification(
            model,
            tokenizer,
            arguments,
            prompter=prompter,
            full_dataset=full_dataset,
            class_weights=[1, 1.3, 1.2, 1.75, 4],
            save_path=f"{save_path}/classification" if full_dataset else None,
        )
        pprint(train_logs)
        pprint(metrics)
    if not full_dataset or not train:
        pprint(test_classification(model, tokenizer, arguments))
    pprint(predict_classification(model, tokenizer, prompts, arguments, prompter))


def run_detection(
    full_dataset: bool,
    train: bool,
    prompter: Callable[[str], str],
    threshold: float | None,
):
    set_random_seeds()
    arguments = TrainingArguments(
        num_train_epochs=4,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"num_cycles": 0.7, "min_lr": 1e-5},
        **default_arguments,
    )
    lora = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=128,
        task_type="CAUSAL_LM",
    )
    model, tokenizer = detection_model(model_name, lora_configuration=lora)
    if train:
        train_logs, metrics = train_detection(
            model,
            tokenizer,
            arguments,
            prompter=prompter,
            full_dataset=full_dataset,
            sample="under",
            threshold=threshold,
            # class_weights=[1.3, 1],
            save_path=f"{save_path}/detection" if full_dataset else None,
        )
        pprint(train_logs)
        pprint(metrics)
    if not full_dataset or not train:
        pprint(test_detection(model, tokenizer, arguments, prompter, threshold))
    if full_dataset:
        pprint(test_exclusive(model, tokenizer, arguments, prompter, threshold))
        pprint(test_lengths(model, tokenizer, arguments, prompter, threshold))
        pprint(test_repetition(model, tokenizer, arguments, prompter, threshold))
    pprint(predict_detection(model, tokenizer, prompts, arguments, prompter, threshold))


def classification_prompter(input: str):
    return f"Rate the humor of the following text on a scale from 1 to 5, where 1 means not funny and 5 means very funny.\n{input}"


def detection_prompter(input: str):
    return f"Detect if the following text is funny 1 or not 0.\n{input}"


if __name__ == "__main__":
    if sys.argv[1] == "classification":
        run_classification(True, False, classification_prompter)
    if sys.argv[1] == "train_classification":
        run_classification(True, True, classification_prompter)
    if sys.argv[1] == "detection":
        run_detection(True, False, detection_prompter, None)
    if sys.argv[1] == "train_detection":
        run_detection(True, True, detection_prompter, None)
