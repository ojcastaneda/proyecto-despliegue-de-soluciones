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

set_random_seeds()
model_name = "distilbert/distilgpt2"
save_path = relative_path("../models/distilgpt2")
default_arguments = {
    "bf16": True,
    "bf16_full_eval": True,
    "disable_tqdm": False,
    "per_device_eval_batch_size": 20,
    "per_device_train_batch_size": 40,
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
    arguments = TrainingArguments(
        num_train_epochs=3,
        lr_scheduler_type="cosine_with_restarts",
        **default_arguments,
    )
    model, tokenizer = classification_model(
        model_name,
        lora_configuration=None,  # Hacer uso de configuración LoRA para causal LM ejemplo: LoraConfig(task_type="CAUSAL_LM")
        tokenizer_name=None,  # Nombre de tokenizador especial en caso de no tener tokenizador
        # classes=[] Lista de tokens para clasificación en caso de usar distinto a 0-1 para detección y 1-5 para clasificación
    )
    fix_tokenizer(tokenizer)
    if train:
        train_logs, metrics = train_classification(
            model,
            tokenizer,
            arguments,
            prompter=prompter,  # Función para modificar los prompts, solo es útil en decoders
            full_dataset=full_dataset,
            class_weights=[1, 1.25, 1.25, 2, 4],
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
    arguments = TrainingArguments(
        num_train_epochs=3,
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
            sample="under",
            threshold=threshold,
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
    return f"""You are a latino guy, rate the humor of the following text on a scale from 1 to 5, where 1 means not funny and 5 means very funny. Do not answer anything else.

Text: Un piano me caería excelente en estos momentos.
Output: 5

Text: Ni Jesús te ama.
Output: 1

Text: Jajajajajajajaj Idiota.
Output: 1

Text: {input}
Output: 
"""


def classification_tune_prompter(input: str):
    return f"""{input}
You are a latino guy, rate the humor of the following text on a scale from 1 to 5, where 1 means not funny and 5 means very funny. Do not answer anything else."""


def detection_prompter(input: str):
    return f"""You are a latino guy, rate the humor of the following text on a scale from 0 to 1, where 0 means not funny and 1 means funny. Do not answer anything else.

Text: Un piano me caería excelente en estos momentos.
Output: 1

Text: Ni Jesús te ama.
Output: 0

Text: Jajajajajajajaj Idiota.
Output: 0

Text: {input}
"""


def detection_tune_prompter(input: str):
    return f"""{input}
You are a latino guy, rate the humor of the following text on a scale from 0 to 1, where 0 means not funny and 1 means funny. Do not answer anything else."""


if __name__ == "__main__":
    # run_classification(True, False, classification_tune_prompter)
    run_classification(True, True, classification_tune_prompter)
    # run_detection(True, False, detection_tune_prompter, None)
    run_detection(True, True, detection_tune_prompter, None)
