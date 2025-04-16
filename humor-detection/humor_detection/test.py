from .dataset import Test, load_csv
from .utils import (
    classification_classes,
    detection_classes,
    optimize_arguments,
    preprocess_logits,
    setup,
)
from pandas import DataFrame
from peft import PeftModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from typing import Callable


def test(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: DataFrame,
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None,
    classes: list[str],
):
    optimize_arguments(arguments, model)
    tokens, data_collator, preprocess_dataset, compute_metrics = setup(
        model, tokenizer, classes
    )
    return Trainer(
        model,
        arguments,
        data_collator,
        eval_dataset=preprocess_dataset(dataset, tokenizer, prompter),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits(tokens),
    ).evaluate()


def test_classification(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None = None,
):
    return test(
        model,
        tokenizer,
        load_csv(Test.classification_path),
        arguments,
        prompter,
        classification_classes,
    )


def test_detection(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None = None,
):
    return test(
        model,
        tokenizer,
        load_csv(Test.detection_path),
        arguments,
        prompter,
        detection_classes,
    )
