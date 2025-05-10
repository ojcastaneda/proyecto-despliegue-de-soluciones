from .utils import calculate_metrics
from datasets import Dataset
from numpy.typing import NDArray
from os.path import exists, isdir
from pandas import DataFrame
from peft import LoraConfig, PeftModel, get_peft_model
from scipy.special import softmax
from shutil import rmtree
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from typing import Callable


def create_model(
    model_name: str,
    classes: int,
    lora_configuration: LoraConfig | None,
    tokenizer_name: str | None,
) -> tuple[PeftModel | PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name if tokenizer_name is None else tokenizer_name
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=classes, ignore_mismatched_sizes=True
    )
    if lora_configuration is not None:
        model = get_peft_model(model, lora_configuration)
    return model, tokenizer  # type: ignore


def classification_model(
    model_name: str,
    lora_configuration: LoraConfig | None = None,
    tokenizer_name: str | None = None,
):
    return create_model(
        model_name,
        5,
        lora_configuration,
        tokenizer_name,
    )


def detection_model(
    model_name: str,
    lora_configuration: LoraConfig | None = None,
    tokenizer_name: str | None = None,
):
    return create_model(model_name, 2, lora_configuration, tokenizer_name)


def load_model(model_name: str, path: str, tokenizer_name: str | None = None):
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name if tokenizer_name is None else tokenizer_name
    )
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(path)
    return model, tokenizer


def save_model(model: PreTrainedModel | PeftModel, path: str):
    if exists(path) and isdir(path):
        rmtree(path)
    model.save_pretrained(path)


def compute_labels(logits: NDArray, threshold: float | None) -> NDArray:
    if threshold is None:
        return logits.argmax(-1)
    return (softmax(logits, -1)[:, 1] > threshold).astype(int)


def compute_metrics(threshold: float | None):
    def _compute_metrics(prediction: EvalPrediction):
        return calculate_metrics(
            prediction.label_ids, compute_labels(prediction.predictions, threshold)  # type: ignore
        )

    return _compute_metrics


def predict(prediction: PredictionOutput, threshold: float | None):
    return compute_labels(prediction.predictions, threshold)  # type: ignore


def preprocess_dataset(
    tokenizer: PreTrainedTokenizerBase, prompter: Callable[[str], str] | None
):
    tokenizer.truncation_side = "left"

    def tokenize_function(examples):
        if prompter is not None:
            examples["text"] = [prompter(text) for text in examples["text"]]
        tokenized = tokenizer(
            examples["text"], padding=False, truncation=True, add_special_tokens=False
        )
        tokenized["labels"] = examples["score"]
        return tokenized

    def _preprocess_dataset(dataset: DataFrame):
        return Dataset.from_pandas(dataset).map(tokenize_function, batched=True)

    return _preprocess_dataset
