from .decoder import predict as predict_decoder
from .encoder import predict as predict_encoder
from .utils import (
    classification_classes,
    detection_classes,
    optimize_arguments,
    setup,
)
from pandas import DataFrame
from peft import PeftModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from typing import Callable


def predict(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None,
    classes: list[str],
):
    optimize_arguments(arguments, model)
    tokens, data_collator, preprocess_dataset, _ = setup(model, tokenizer, classes)
    data = preprocess_dataset(
        DataFrame(
            {"text": prompts, "score": [0 if tokens is None else ""] * len(prompts)}
        ),
        tokenizer,
        prompter,
    )
    prediction = Trainer(
        model,
        arguments,
        data_collator,
        eval_dataset=data,
    ).predict(
        data  # type: ignore
    )
    if tokens is None:
        return predict_encoder(prediction)
    return predict_decoder(prediction, tokenizer, tokens)


def predict_classification(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None = None,
):
    return predict(
        model, tokenizer, prompts, arguments, prompter, classification_classes
    )


def predict_detection(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None = None,
):
    return predict(model, tokenizer, prompts, arguments, prompter, detection_classes)
