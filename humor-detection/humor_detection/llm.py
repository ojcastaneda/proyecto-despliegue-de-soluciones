from concurrent.futures import ThreadPoolExecutor
from .dataset import (
    Exclusive,
    LongLengths,
    Repetition,
    ShortLengths,
    Test,
    load_csv,
)
from .utils import calculate_metrics, log_metrics_mlflow
from google.genai import Client
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)
from os import environ
from pandas import DataFrame
from transformers.training_args import TrainingArguments
from typing import Callable


def _predict(args: tuple[str, str, list[str], str | None]):
    text, model, classes, token = args
    client = Client(api_key=token or environ.get("GEMINI_API_KEY"))
    config = GenerateContentConfig(
        temperature=0,
        max_output_tokens=1,
        safety_settings=[
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
        ],
    )
    prediction = client.models.generate_content(
        model=model, contents=text, config=config
    ).text
    return classes.index(prediction) if prediction in classes else -1


def predict(
    prompts: list[str],
    classes: list[str],
    prompter: Callable[[str], str],
    model="gemini-2.0-flash",
    threads=10,
    api_key: str | None = None,
):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        labels = list(
            executor.map(
                _predict,
                [(prompter(text), model, classes, api_key) for text in prompts],
            )
        )
    probabilities = [float("nan")] * len(prompts)
    output = {f"score_{i}": probabilities for i in range(len(classes))}
    output["labels"] = labels  # type: ignore
    return DataFrame(output)


def test(
    dataset: DataFrame,
    model: str,
    prompter: Callable[[str], str],
    classes: list[str],
    threads: int,
    prefix: str,
    api_key: str | None = None,
):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        dataset["prediction"] = list(
            executor.map(
                _predict,
                [(prompter(text), model, classes, api_key) for text in dataset["text"]],
            )
        )
    output = calculate_metrics(list(dataset["score"]), list(dataset["prediction"]))
    log_metrics_mlflow(
        output,
        {"prompter": None if prompter is None else repr(prompter("<PLACEHOLDER>"))},
        TrainingArguments(),
        (model, classes),
        None,
        prefix,
    )
    return output


def test_classification(
    prompter: Callable[[str], str],
    model="gemini-2.0-flash",
    classes=["1", "2", "3", "4", "5"],
    threads=10,
    api_key: str | None = None,
):
    return test(
        load_csv(Test.classification_path),
        model,
        prompter,
        classes,
        threads,
        "test",
        api_key,
    )


def test_detection(
    prompter: Callable[[str], str],
    model="gemini-2.0-flash",
    classes=["0", "1"],
    threads=10,
    api_key: str | None = None,
):
    return test(
        load_csv(Test.detection_path),
        model,
        prompter,
        classes,
        threads,
        "test",
        api_key,
    )


def test_exclusive(
    prompter: Callable[[str], str],
    model="gemini-2.0-flash",
    classes=["0", "1"],
    threads=10,
    api_key: str | None = None,
):
    return test(
        load_csv(Exclusive),
        model,
        prompter,
        classes,
        threads,
        "test_exclusive",
        api_key,
    )


def test_lengths(
    prompter: Callable[[str], str],
    model="gemini-2.0-flash",
    classes=["0", "1"],
    threads=10,
    api_key: str | None = None,
):
    long = test(
        load_csv(LongLengths),
        model,
        prompter,
        classes,
        threads,
        "test_long_lengths",
        api_key,
    )
    short = test(
        load_csv(ShortLengths),
        model,
        prompter,
        classes,
        threads,
        "test_short_lengths",
        api_key,
    )
    return short, long


def test_repetition(
    prompter: Callable[[str], str],
    model="gemini-2.0-flash",
    classes=["0", "1"],
    threads=10,
    api_key: str | None = None,
):
    return test(
        load_csv(Repetition),
        model,
        prompter,
        classes,
        threads,
        "test_repetition",
        api_key,
    )
