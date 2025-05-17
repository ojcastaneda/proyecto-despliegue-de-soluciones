from typing import Callable
from humor_detection.decoder import classification_model, detection_model
from humor_detection.test import (
    test_classification,
    test_detection,
    test_exclusive,
    test_lengths,
    test_repetition,
)
from humor_detection.predict import predict_classification, predict_detection
from humor_detection.utils import set_random_seeds
from pprint import pprint
from transformers.training_args import TrainingArguments
import sys

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
default_arguments = {
    "bf16": True,
    "bf16_full_eval": True,
    "disable_tqdm": False,
    "per_device_eval_batch_size": 10,
}
prompts = [
    "- Martínez, queda usted despedido.\n- Pero, si yo no he hecho nada.\n- Por eso, por eso.",
    "¿Cuál es el último animal que subió al arca de Noé? El del-fin.",
    "El otro día unas chicas llamarón a mi puerta y me pidieron una pequeña donación para una piscina local.\nLes di un garrafa de agua.",
    "The brain surgeon changed my life. He really opened my mind.",
    "djasndoasndoa",
    "jajaja",
]


def run_classification(prompter: Callable[[str], str]):
    set_random_seeds()
    model, tokenizer = classification_model(model_name)
    arguments = TrainingArguments(**default_arguments)
    pprint(test_classification(model, tokenizer, arguments, prompter))
    pprint(predict_classification(model, tokenizer, prompts, arguments, prompter))


def run_detection(prompter: Callable[[str], str], threshold: float | None):
    set_random_seeds()
    model, tokenizer = detection_model(model_name)
    arguments = TrainingArguments(**default_arguments)
    pprint(test_detection(model, tokenizer, arguments, prompter, threshold))
    pprint(test_exclusive(model, tokenizer, arguments, prompter, threshold))
    pprint(test_lengths(model, tokenizer, arguments, prompter, threshold))
    pprint(test_repetition(model, tokenizer, arguments, prompter, threshold))
    pprint(predict_detection(model, tokenizer, prompts, arguments, prompter, threshold))


def classification_prompter(input: str):
    return f"""Rate the funniness of the following text on a scale from 1 to 5, based on how likely it is to make a person laugh or smile. Consider factors like cleverness, absurdity, surprise, or wordplay.

Text: Un piano me caería excelente en estos momentos.
Score: 5

Text: —¿Tan linda y sin novio? —¿Tan grande y tan bobo?
Score: 1

Text: —¿Qué es eso que traes en tu bolsa?
—Un AK-47.
—No, al lado del AK-47.
—Unos Chettos bolita.
—¡No puedes entrar al Cine con comida!
Score: 3

Text: {input}
Score: """


def detection_prompter(input: str):
    return f"""Detect if the following text is funny 1 or not 0 following the provided examples.

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

Text: {input}
Score: """


if __name__ == "__main__":
    if sys.argv[1] == "classification":
        run_classification(classification_prompter)
    if sys.argv[1] == "detection":
        run_detection(detection_prompter, 0.35)
