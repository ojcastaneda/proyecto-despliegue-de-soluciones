from .dataset import Test, Train, TrainMultilingual, load_csv
from .decoder import save_model as save_decoder
from .encoder import save_model as save_encoder
from .utils import CustomTrainer, log_metrics_mlflow
from pandas import DataFrame, concat
from peft import PeftModel
from sklearn.model_selection import train_test_split
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
    best_model_metric: Literal["macro_f1", "weighted_f1", "accuracy"],
    save_path: str | None,
):
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
    trainer = CustomTrainer(
        model,
        tokenizer,
        arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        class_weights=class_weights,
        prompter=prompter,
        best_model_metric=best_model_metric,
        threshold=threshold,
    )
    _, train_logs, eval_metrics = trainer.train()
    log_metrics_mlflow(
        eval_metrics,
        {
            "class_weights": class_weights,
            "sample": sample,
            "threshold": threshold,
            "prompter": None if prompter is None else repr(prompter("<PLACEHOLDER>")),
        },
        arguments,
        model,
        tokenizer,
        not full_dataset,
    )
    train_logs = DataFrame(train_logs)
    if save_path is not None:
        (save_encoder if trainer.token_ids is None else save_decoder)(model, save_path)
        train_logs.to_csv(f"{save_path}/train_logs.csv", index=False)
    return train_logs, eval_metrics


def train_classification(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    arguments: TrainingArguments,
    full_dataset=False,
    english_data=False,
    prompter: Callable[[str], str] | None = None,
    class_weights: list[float] | None = None,
    sample: Literal["under", "over", False] = False,
    best_model_metric: Literal["macro_f1", "weighted_f1", "accuracy"] = "macro_f1",
    save_path: str | None = None,
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
    return train(
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
        best_model_metric,
        save_path,
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
    best_model_metric: Literal["macro_f1", "weighted_f1", "accuracy"] = "macro_f1",
    save_path: str | None = None,
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
    return train(
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
        best_model_metric,
        save_path,
    )
