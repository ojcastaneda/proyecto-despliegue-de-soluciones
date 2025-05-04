from .dataset import Test, Train, TrainMultilingual, load_csv
from .utils import (
    CustomTrainer,
    log_metrics_mlflow,
    optimize_arguments,
    preprocess_logits,
    setup,
)
from pandas import DataFrame, concat
from peft import PeftModel
from sklearn.model_selection import train_test_split
from torch import tensor
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments
from typing import Callable, Literal
from sklearn.utils import resample


def train(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: DataFrame,
    validation_dataset: DataFrame,
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None,
    class_weights: list[float] | None,
    sample: Literal["under", "over", False],
    threshold: float | None,
    full_dataset: bool,
):
    optimize_arguments(arguments, model)
    if sample:
        if sample == "under":
            count = train_dataset["score"].value_counts().min()
            replace = False
        else:
            count = train_dataset["score"].value_counts().max()
            replace = True
        train_dataset = concat(
            [
                resample(group, replace=replace, n_samples=count)
                for _, group in train_dataset.groupby("score")
            ]  # type: ignore
        ).reset_index(drop=True)
    token_ids, data_collator, preprocess_dataset, compute_metrics = setup(
        model, tokenizer, arguments
    )
    weights = None
    if class_weights is not None:
        weights = [0.0] * (
            model.config.num_labels if token_ids is None else len(model.config.classes)  # type: ignore
        )
        for i in range(len(class_weights)):
            weights[i] = class_weights[i]
        weights = tensor(weights)
    eval_dataset = preprocess_dataset(validation_dataset, tokenizer, prompter)
    if full_dataset:
        arguments.set_evaluate("no")
    trainer = CustomTrainer(
        model,
        arguments,
        data_collator,
        train_dataset=preprocess_dataset(train_dataset, tokenizer, prompter),
        eval_dataset=None if full_dataset else eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits(threshold, token_ids),
        weights=weights,
        token_ids=token_ids,
    )
    trainer.train()
    if full_dataset:
        trainer.eval_dataset = eval_dataset
    log_metrics_mlflow(
        trainer.evaluate(),
        {
            "class_weights": class_weights,
            "sample": sample,
            "threshold": threshold,
            "prompter": None if prompter is None else prompter("<PLACEHOLDER>"),
        },
        arguments,
        model,
        tokenizer,
        not full_dataset,
    )


def train_classification(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    arguments: TrainingArguments,
    full_dataset=False,
    english_data=False,
    prompter: Callable[[str], str] | None = None,
    class_weights: list[float] | None = None,
    sample: Literal["under", "over", False] = False,
):
    train_dataset = load_csv(
        (TrainMultilingual if english_data else Train).classification_path
    )
    if full_dataset:
        validation_dataset = load_csv(Test.classification_path)
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
        class_weights,
        sample,
        None,
        full_dataset,
    )


def train_detection(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    arguments: TrainingArguments,
    full_dataset=False,
    english_data=False,
    prompter: Callable[[str], str] | None = None,
    class_weights: list[float] | None = None,
    sample: Literal["under", "over", False] = False,
    threshold: float | None = None,
):
    train_dataset = load_csv(
        (TrainMultilingual if english_data else Train).detection_path
    )
    if full_dataset:
        validation_dataset = load_csv(Test.detection_path)
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
        class_weights,
        sample,
        threshold,
        full_dataset,
    )
