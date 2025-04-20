from datasets import Dataset
from numpy import where
from os.path import exists, isdir
from pandas import DataFrame
from peft import LoraConfig, PeftModel, get_peft_model
from shutil import rmtree
from sklearn.metrics import classification_report
from torch import Tensor, argmax, tensor
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
) -> tuple[PeftModel | PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name if tokenizer_name is None else tokenizer_name
    )
    model = AutoModelForCausalLM.from_pretrained(model_name)
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
        lora_configuration,
        tokenizer_name,
    )


def detection_model(
    model_name: str,
    lora_configuration: LoraConfig | None = None,
    tokenizer_name: str | None = None,
):
    return create_model(model_name, lora_configuration, tokenizer_name)


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
    prediction: PredictionOutput, tokenizer: PreTrainedTokenizerBase, tokens: list[int]
):
    return tokenizer.batch_decode(
        where(
            (prediction.label_ids == -100),
            tokenizer.all_special_ids[0],
            preprocess_logits(
                tokens, tensor(prediction.predictions), tensor([])
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
            examples["text"] = [prompter(text) for text in examples["text"]]
        input_ids = []
        attention_mask = []
        for text, score in zip(examples["text"], examples["score"]):
            score_tokens = tokenizer(str(score), add_special_tokens=False)
            text_tokens = tokenizer(
                text, truncation=True, max_length=tokenizer.model_max_length - len(score_tokens["input_ids"])  # type: ignore
            )
            input_ids.append(text_tokens["input_ids"] + score_tokens["input_ids"])  # type: ignore
            attention_mask.append(
                text_tokens["attention_mask"] + score_tokens["attention_mask"]  # type: ignore
            )
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return Dataset.from_pandas(dataset).map(tokenize_function, batched=True)


def preprocess_logits(target_tokens: list[int], logits: Tensor, _: Tensor):
    filtered_logits = logits[..., target_tokens]
    target_indices = argmax(filtered_logits, dim=-1)
    return tensor([target_tokens[i] for i in target_indices.flatten()]).reshape(
        target_indices.shape
    )


def save_model(model: PreTrainedModel | PeftModel, path: str):
    if exists(path) and isdir(path):
        rmtree(path)
    model.save_pretrained(path)


def setup(tokenizer: PreTrainedTokenizerBase, classes: list[str]):
    tokens: list[int] = []
    token_dictionary: dict[int, str] = {}
    for token in classes:
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        tokens.append(token_id)
        token_dictionary[token_id] = token

    def _metrics(prediction: EvalPrediction):
        return metrics(token_dictionary, prediction)

    return (
        tokens,
        LastTokenCollator(tokenizer, mlm=False),
        preprocess,
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
