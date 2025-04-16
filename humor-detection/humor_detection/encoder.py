from datasets import Dataset
from os.path import exists, isdir
from pandas import DataFrame
from peft import LoraConfig, PeftModel, get_peft_model
from shutil import rmtree
from sklearn.metrics import classification_report
from torch import Tensor, argmax, tensor
from transformers.data.data_collator import DataCollatorWithPadding
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


def metrics(prediction: EvalPrediction) -> dict:
    return classification_report(
        prediction.label_ids, prediction.predictions, output_dict=True, zero_division=1
    )  # type: ignore


def save_model(model: PreTrainedModel | PeftModel, path: str):
    if exists(path) and isdir(path):
        rmtree(path)
    model.save_pretrained(path)


def predict(prediction: PredictionOutput):
    return preprocess_logits(tensor(prediction.predictions), tensor([]))


def preprocess(
    dataset: DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    prompter: Callable[[str], str] | None,
):
    def tokenize_function(examples):
        if prompter is not None:
            examples["text"] = [prompter(text) for text in examples["text"]]
        tokenized = tokenizer(examples["text"], padding=False, truncation=True)
        tokenized["labels"] = examples["score"]
        return tokenized

    return Dataset.from_pandas(dataset).map(tokenize_function, batched=True)


def preprocess_logits(logits: Tensor, _: Tensor | None = None):
    return argmax(logits, dim=-1)


def setup(tokenizer: PreTrainedTokenizerBase):
    return None, DataCollatorWithPadding(tokenizer), preprocess, metrics
