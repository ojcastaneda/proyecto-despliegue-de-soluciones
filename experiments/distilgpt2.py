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
from pprint import pprint
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments
import sys

model_name = "distilbert/distilgpt2"
save_path = relative_path("../models/distilgpt2")
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


# Para GPT2 es necesario asignar un pad_token y puede que para otros modelos también
def fix_tokenizer(tokenizer: PreTrainedTokenizerBase):
    tokenizer.pad_token = tokenizer.eos_token


def run_classification(full_dataset: bool, train: bool, prompter: Callable[[str], str]):
    set_random_seeds()
    arguments = TrainingArguments(
        num_train_epochs=4,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"num_cycles": 0.8, "min_lr": 1e-5},
        **default_arguments,
    )
    model, tokenizer = classification_model(model_name)
    fix_tokenizer(tokenizer)
    if train:
        train_logs, metrics = train_classification(
            model,
            tokenizer,
            arguments,
            prompter=prompter,
            full_dataset=full_dataset,
            save_path=f"{save_path}/classification" if full_dataset else None,
        )
        pprint(train_logs)
        pprint(metrics)
    if not full_dataset or not train:
        pprint(test_classification(model, tokenizer, arguments, prompter))
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
        lr_scheduler_kwargs={"num_cycles": 0.6, "min_lr": 1e-5},
        **default_arguments,
    )
    model, tokenizer = detection_model(model_name)
    fix_tokenizer(tokenizer)
    if train:
        train_logs, metrics = train_detection(
            model,
            tokenizer,
            arguments,
            prompter=prompter,
            full_dataset=full_dataset,
            threshold=threshold,
            class_weights=[1.2, 1],
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
    return f"""{input}
How funny is this text?
1) Slightly
2) Mildly
3) Moderately
4) Very
5) Incredibly
The text is """


def detection_prompter(input: str):
    return f"""{input}
Detect if the text is funny 1 or not 0.
Is the text funny? The text is """


if __name__ == "__main__":
    if sys.argv[1] == "classification":
        run_classification(True, False, classification_prompter)
    if sys.argv[1] == "train_classification":
        run_classification(True, True, classification_prompter)
    if sys.argv[1] == "detection":
        run_detection(True, False, detection_prompter, None)
    if sys.argv[1] == "train_detection":
        run_detection(True, True, detection_prompter, None)
