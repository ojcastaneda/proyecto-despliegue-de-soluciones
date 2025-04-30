from collections import OrderedDict
from datasets import Dataset
from numpy import where
from os.path import exists, isdir
from pandas import DataFrame
from peft import LoraConfig, PeftModel, get_peft_model
from shutil import rmtree
from sklearn.metrics import classification_report
from torch import Tensor, argmax, softmax, tensor, where as where_torch, int32
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from transformers.training_args import TrainingArguments
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
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(path)
    return model, tokenizer


def metrics(
    token_dictionary: dict[int, str],
    prediction: EvalPrediction,
) -> dict:
    if isinstance(prediction.label_ids, tuple) or isinstance(
        prediction.predictions, tuple
    ):
        return {}
    batch_size = prediction.label_ids.shape[0]
    mask = prediction.label_ids != -100
    labels = [0] * batch_size
    predictions = [0] * batch_size
    for i in range(batch_size):
        positive_indices = where(mask[i])[0]
        index = 0 if len(positive_indices) < 0 else positive_indices[-1]
        labels[i] = token_dictionary[prediction.label_ids[i, index]]
        predictions[i] = token_dictionary[prediction.predictions[i, index]]
    return classification_report(
        labels,
        predictions,
        output_dict=True,
        zero_division=1,
    )  # type: ignore


def predict(
    prediction: PredictionOutput,
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Tensor,
    threshold: float | None,
):
    return tokenizer.batch_decode(
        where(
            (prediction.label_ids == -100),
            tokenizer.all_special_ids[0],
            preprocess_logits(
                token_ids,
                tensor(prediction.predictions),
                tensor([]),
                threshold=threshold,
            ).numpy(),
        ),
        True,
    )


def preprocess(
    dataset: DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    prompter: Callable[[str], str] | None,
):
    def tokenize_function(examples):
        if prompter is not None:
            examples["text"] = [
                prompter(text) + score
                for text, score in zip(examples["text"], examples["score"])
            ]
        return tokenizer(examples["text"], padding=False, truncation=True)

    return Dataset.from_pandas(dataset).map(tokenize_function, batched=True)


def preprocess_logits(
    target_tokens: Tensor, logits: Tensor, _: Tensor, threshold: float | None
):
    logits = logits[..., target_tokens]
    if threshold is None:
        target_indices = argmax(logits, dim=-1)
    else:
        target_indices = where_torch(softmax(logits, dim=-1)[..., 1] > threshold, 1, 0)
    return target_tokens[target_indices]


def save_model(model: PreTrainedModel | PeftModel, path: str):
    if exists(path) and isdir(path):
        rmtree(path)
    model.save_pretrained(path)


def setup(
    tokenizer: PreTrainedTokenizerBase, classes: list[str], arguments: TrainingArguments
):
    token_ids = []
    id_to_token: OrderedDict[int, str] = OrderedDict()
    for token in classes:
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        token_ids.append(token_id)
        id_to_token[token_id] = token
    token_ids = tensor(token_ids, device=arguments.device, dtype=int32)

    def _metrics(prediction: EvalPrediction):
        return metrics(id_to_token, prediction)

    def _preprocess(
        dataset: DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        prompter: Callable[[str], str] | None,
    ):
        if "score" in dataset.columns:
            dataset["score"] = dataset["score"].astype(int).map(lambda x: classes[x])
        return preprocess(dataset, tokenizer, prompter)

    return (
        token_ids,
        LastTokenCollator(tokenizer, mlm=False),
        _preprocess,
        _metrics,
    )


class LastTokenCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for i in range(len(batch["labels"])):
            for j in range(len(batch["labels"][i]) - 1):
                if batch["labels"][i][j] == -100:
                    break
                if batch["labels"][i][j + 1] != -100:
                    batch["labels"][i][j] = -100
        return batch
