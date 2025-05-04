from .dataset import Test, load_csv
from .utils import (
    CustomTrainer,
    log_metrics_mlflow,
    optimize_arguments,
    preprocess_logits,
    setup,
)
from pandas import DataFrame
from peft import PeftModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments
from typing import Callable


def test(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: DataFrame,
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None,
    threshold: float | None,
):
    optimize_arguments(arguments, model)
    token_ids, data_collator, preprocess_dataset, compute_metrics = setup(
        model, tokenizer, arguments
    )
    trainer = CustomTrainer(
        model,
        arguments,
        data_collator,
        eval_dataset=preprocess_dataset(dataset, tokenizer, prompter),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits(threshold, token_ids),
        token_ids=token_ids,
    )
    log_metrics_mlflow(trainer.evaluate(), {}, arguments, model, tokenizer, False)


def test_classification(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None = None,
):
    return test(
        model, tokenizer, load_csv(Test.classification_path), arguments, prompter, None
    )


def test_detection(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None = None,
    threshold: float | None = None,
):
    return test(
        model, tokenizer, load_csv(Test.detection_path), arguments, prompter, threshold
    )
