from .utils import calculate_metrics
from datasets import Dataset
from numpy import array, where
from numpy.typing import NDArray
from os.path import exists, isdir
from pandas import DataFrame
from peft import LoraConfig, PeftModel, get_peft_model
from scipy.special import softmax
from shutil import rmtree
from torch import Tensor, arange
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from typing import Callable


def create_model(
    model_name: str,
    lora_configuration: LoraConfig | None,
    tokenizer_name: str | None,
    classes: list[str],
) -> tuple[PeftModel | PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name if tokenizer_name is None else tokenizer_name
    )
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if lora_configuration is not None:
        model = get_peft_model(model, lora_configuration)
    model.config.classes = classes  # type: ignore
    return model, tokenizer  # type: ignore


def classification_model(
    model_name: str,
    lora_configuration: LoraConfig | None = None,
    tokenizer_name: str | None = None,
    classes: list[str] | None = None,
):
    return create_model(
        model_name,
        lora_configuration,
        tokenizer_name,
        classes or ["1", "2", "3", "4", "5"],
    )


def detection_model(
    model_name: str,
    lora_configuration: LoraConfig | None = None,
    tokenizer_name: str | None = None,
    classes: list[str] | None = None,
):
    return create_model(
        model_name, lora_configuration, tokenizer_name, classes or ["0", "1"]
    )


def load_model(model_name: str, path: str, tokenizer_name: str | None = None):
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name if tokenizer_name is None else tokenizer_name
    )
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(path)
    return model, tokenizer


def compute_labels(logits: NDArray, threshold: float | None):
    return (
        logits.argmax(-1)
        if threshold is None
        else where(softmax(logits, -1)[..., 1] > threshold, 1, 0)
    )


def compute_metrics(lookup_token: NDArray, threshold: float | None):
    def _compute_metrics(prediction: EvalPrediction):
        labels, predictions = extract_predictions(
            prediction.label_ids,  # type: ignore
            compute_labels(prediction.predictions, threshold),  # type: ignore
            lookup_token,
        )
        return calculate_metrics(labels, predictions)

    return _compute_metrics


def extract_predictions(label_ids: NDArray, prediction: NDArray, lookup_token: NDArray):
    batch_size = label_ids.shape[0]
    mask = label_ids != -100
    labels = []
    predictions = []
    for i in range(batch_size):
        positive_indices = where(mask[i])[0]
        index = 0 if len(positive_indices) < 0 else positive_indices[-1]
        labels.append(label_ids[i, index])
        predictions.append(prediction[i, index])
    return lookup_token[array(labels)], array(predictions)


def predict(
    prediction: PredictionOutput,
    lookup_token: NDArray,
    threshold: float | None,
):
    _, predictions = extract_predictions(prediction.label_ids, prediction.predictions, lookup_token)  # type: ignore
    return (compute_labels(predictions, threshold), predictions)


def preprocess_dataset(
    tokenizer: PreTrainedTokenizerBase,
    prompter: Callable[[str], str] | None,
    classes: list[str],
):
    tokenizer.truncation_side = "left"

    def tokenize_function(examples):
        texts = (
            examples["text"]
            if prompter is None
            else [prompter(text) for text in examples["text"]]
        )
        tokenized = tokenizer(
            texts, padding=False, truncation=True, add_special_tokens=False
        )
        score: list[list[int]] = tokenizer(
            examples["score"], max_length=1, truncation=True, add_special_tokens=False
        )[
            "input_ids"
        ]  # type: ignore
        input_ids: list[list[int]] = tokenized["input_ids"]  # type: ignore
        attention_mask: list[list[int]] = tokenized["attention_mask"]  # type: ignore
        max_length = tokenizer.model_max_length or 10000
        for i in range(len(input_ids)):
            input_ids[i].append(score[i][0])
            attention_mask[i].append(1)
            if len(input_ids[i]) > max_length:
                input_ids[i] = input_ids[i][1:]
                attention_mask[i] = attention_mask[i][1:]
        return tokenized

    def _preprocess_dataset(dataset: DataFrame):
        if "score" in dataset.columns:
            dataset["score"] = dataset["score"].astype(int).map(lambda x: classes[x])
        return Dataset.from_pandas(dataset).map(tokenize_function, batched=True)

    return _preprocess_dataset


def preprocess_logits(token_ids: NDArray):
    def _preprocess_logits(logits: NDArray, _: NDArray):
        return logits[..., token_ids]

    return _preprocess_logits


def save_model(model: PreTrainedModel | PeftModel, path: str):
    if exists(path) and isdir(path):
        rmtree(path)
    model.save_pretrained(path)


class LastTokenCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels: Tensor = batch["labels"]
        last = (
            ((labels != -100).float() * arange(labels.size(1), device=labels.device))
            .max(dim=-1)
            .values.long()
        )
        mask = arange(labels.size(1), device=labels.device)[None, :] != last[:, None]
        labels[mask] = -100
        return batch
