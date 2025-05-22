from .utils import calculate_metrics
from dataclasses import asdict, dataclass, field
from datasets import Dataset

# from json import load
from numpy import array, where
from numpy.typing import NDArray
from os.path import exists, join, isdir
from pandas import DataFrame
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraRuntimeConfig
from scipy.special import softmax
from shutil import rmtree
from torch import Tensor, arange
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from typing import Callable


@dataclass
class DecoderLoraConfig(LoraConfig):
    classes: list[str] = field(default_factory=list)

    @classmethod
    def from_lora_config(cls, config: LoraConfig | PeftConfig, classes: list[str]):
        data = asdict(config)
        if "runtime_config" in data and isinstance(data["runtime_config"], dict):
            data["runtime_config"] = LoraRuntimeConfig(**data["runtime_config"])
        return cls(**data, classes=classes)

    @classmethod
    def from_peft_type(cls, **kwargs):
        classes = kwargs.pop("classes")
        return DecoderLoraConfig.from_lora_config(
            LoraConfig.from_peft_type(**kwargs), classes
        )


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
        model.enable_input_require_grads()
        model = get_peft_model(
            model, DecoderLoraConfig.from_lora_config(lora_configuration, classes)
        )
    else:
        model.config.classes = classes  # type: ignore
    return model, tokenizer  # type: ignore


def classification_model(
    model_name: str,
    lora_configuration: LoraConfig | None = None,
    tokenizer_name: str | None = None,
    classes: list[str] = ["1", "2", "3", "4", "5"],
):
    return create_model(model_name, lora_configuration, tokenizer_name, classes)


def detection_model(
    model_name: str,
    lora_configuration: LoraConfig | None = None,
    tokenizer_name: str | None = None,
    classes: list[str] = ["0", "1"],
):
    return create_model(model_name, lora_configuration, tokenizer_name, classes)


def load_model(model_name: str, path: str, tokenizer_name: str | None = None):
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name if tokenizer_name is None else tokenizer_name
    )
    try:
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(model_name),
            path,
            config=DecoderLoraConfig.from_pretrained(path),
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(path)
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


def extract_predictions(
    label_ids: NDArray | list[list[int]], prediction: NDArray, lookup_token: NDArray
):
    batch_size = len(label_ids)
    labels = []
    predictions = []
    is_prediction = isinstance(label_ids, list)
    shift_correction = int(not is_prediction)
    for i in range(batch_size):
        batch_labels = label_ids[i]
        if is_prediction:
            batch_labels = array(batch_labels)
        positive_indices = where(batch_labels != -100)[0]
        index = 0 if len(positive_indices) < 1 else positive_indices[-1]
        labels.append(batch_labels[index])
        predictions.append(prediction[i, index - shift_correction])
    labels = array(labels)
    return labels if is_prediction else lookup_token[labels], array(predictions)


def predict(
    original: Dataset,
    prediction: PredictionOutput,
    lookup_token: NDArray,
    threshold: float | None,
):
    _, predictions = extract_predictions(original["input_ids"], prediction.predictions, lookup_token)  # type: ignore
    return (compute_labels(predictions, threshold), predictions)


def preprocess_dataset(
    tokenizer: PreTrainedTokenizerBase,
    prompter: Callable[[str], str] | None,
    classes: list[str],
    config: PretrainedConfig,
):
    tokenizer.truncation_side = "left"

    def tokenize_function(examples):
        texts = (
            examples["text"]
            if prompter is None
            else [prompter(text) for text in examples["text"]]
        )
        max_length = (
            config.max_position_embeddings
            if hasattr(config, "max_position_embeddings")
            else 1024
        )
        tokenized = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        scores: list[list[int]] = tokenizer(
            examples["score"], max_length=1, truncation=True, add_special_tokens=False
        )[
            "input_ids"
        ]  # type: ignore
        input_ids: list[list[int]] = tokenized["input_ids"]  # type: ignore
        attention_mask: list[list[int]] = tokenized["attention_mask"]  # type: ignore
        for i in range(len(input_ids)):
            if scores[i][0] == tokenizer.eos_token_id:
                if input_ids[i][-1] == tokenizer.eos_token_id:
                    input_ids[i].pop()
                    attention_mask[i].pop()
                continue
            if input_ids[i][-1] == tokenizer.eos_token_id:
                input_ids[i][-1] = scores[i][0]
                attention_mask[i][-1] = 1
                continue
            input_ids[i].append(scores[i][0])
            attention_mask[i].append(1)
            if len(input_ids[i]) > max_length:
                input_ids[i] = input_ids[i][1:]
                attention_mask[i] = attention_mask[i][1:]
        return tokenized

    def score_to_class(value: str):
        return tokenizer.eos_token if value is None else classes[int(value)]

    def _preprocess_dataset(dataset: DataFrame):
        if "score" in dataset.columns:
            dataset["score"] = dataset["score"].map(score_to_class)
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
