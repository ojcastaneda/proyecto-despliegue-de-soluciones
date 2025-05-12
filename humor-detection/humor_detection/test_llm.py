from .dataset import (
    Exclusive,
    LongLengths,
    Repetition,
    ShortLengths,
    Test,
    load_csv,
)
from .utils import calculate_metrics, log_metrics_mlflow
from pandas import DataFrame
from transformers.training_args import TrainingArguments
from typing import Callable


def test(dataset: DataFrame, prompter: Callable[[str], str], prefix: str):
    # ==============MODIFICAR CON LLM===================
    name = ""
    results: DataFrame = DataFrame()
    # ==================================================
    output = calculate_metrics(list(results["labels"]), list(results["predictions"]))
    log_metrics_mlflow(
        output,
        {"prompter": None if prompter is None else repr(prompter("<PLACEHOLDER>"))},
        TrainingArguments(),
        (name, results["labels"].unique().tolist()),
        None,
        prefix,
    )
    return output


def test_classification(prompter: Callable[[str], str]):
    return test(load_csv(Test.classification_path), prompter, "test")


def test_detection(prompter: Callable[[str], str]):
    return test(load_csv(Test.detection_path), prompter, "test")


def test_exclusive(prompter: Callable[[str], str]):
    return test(load_csv(Exclusive), prompter, "test_exclusive")


def test_lengths(prompter: Callable[[str], str]):
    long = test(load_csv(LongLengths), prompter, "test_long_lengths")
    short = test(load_csv(ShortLengths), prompter, "test_short_lengths")
    return short, long


def test_repetition(prompter: Callable[[str], str]):
    return test(load_csv(Repetition), prompter, "test_repetition")
