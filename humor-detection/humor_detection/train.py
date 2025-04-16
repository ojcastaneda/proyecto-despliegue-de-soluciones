from .dataset import Test, Train, TrainMultilingual, load_csv
from .utils import (
    classification_classes,
    detection_classes,
    optimize_arguments,
    preprocess_logits,
    setup,
)
from pandas import DataFrame
from peft import PeftModel
from sklearn.model_selection import train_test_split
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from typing import Callable


def train(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: DataFrame,
    validation_dataset: DataFrame,
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None,
    classes: list[str],
):
    optimize_arguments(arguments, model)
    tokens, data_collator, preprocess_dataset, compute_metrics = setup(
        model, tokenizer, classes
    )
    trainer = Trainer(
        model,
        arguments,
        data_collator,
        train_dataset=preprocess_dataset(train_dataset, tokenizer, prompter),
        eval_dataset=preprocess_dataset(validation_dataset, tokenizer, prompter),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits(tokens),
    )
    trainer.train()


def train_classification(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    arguments: TrainingArguments,
    full_dataset=False,
    english_data=False,
    prompter: Callable[[str], str] | None = None,
):
    train_dataset = load_csv(
        (TrainMultilingual if english_data else Train).classification_path
    )
    if full_dataset:
        validation_dataset = load_csv(Test)
    else:
        train_dataset, validation_dataset = train_test_split(
            train_dataset, test_size=0.2
        )
    train(
        model,
        tokenizer,
        train_dataset,
        validation_dataset,
        arguments,
        prompter,
        classification_classes,
    )


def train_detection(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    arguments: TrainingArguments,
    full_dataset=False,
    english_data=False,
    prompter: Callable[[str], str] | None = None,
):
    train_dataset = load_csv(
        (TrainMultilingual if english_data else Train).detection_path
    )
    if full_dataset:
        validation_dataset = load_csv(Test)
    else:
        train_dataset, validation_dataset = train_test_split(
            train_dataset, test_size=0.2
        )
    train(
        model,
        tokenizer,
        train_dataset,
        validation_dataset,
        arguments,
        prompter,
        detection_classes,
    )
