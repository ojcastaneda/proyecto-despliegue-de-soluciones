from dataclasses import dataclass
from datasets import disable_progress_bars
from types import FunctionType
from typing import Callable
from humor_detection.predict import predict_classification, predict_detection
from humor_detection.llm import predict as predict_llm
from humor_detection.utils import relative_path
from importlib.util import module_from_spec, spec_from_file_location
from pandas import DataFrame
from peft import PeftModel
from streamlit import bar_chart, cache_resource, expander
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments
import torch


torch.classes.__path__ = []
disable_progress_bars()


@dataclass
class ModelDetails:
    decoder: bool
    file_name: str
    load_tuned: bool
    threshold: float | None


class PredictionModel:
    def __init__(
        self,
        model: PeftModel | PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        fix_tokenizer: Callable[[PreTrainedTokenizerBase], None] | None,
        prompter: Callable[[str], str] | None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter
        if fix_tokenizer is not None:
            fix_tokenizer(tokenizer)


MODELS = {
    "Bloom": ModelDetails(True, "bloom", False, None),
    "Bloom con LoRA": ModelDetails(True, "bloom", True, None),
    "Canine-c": ModelDetails(False, "canine", True, None),
    "DCCUChile": ModelDetails(False, "dccuchile", True, 0.65),
    "DeepSeek-Qwen": ModelDetails(True, "deepseek-qwen", False, 0.35),
    "DistilBert": ModelDetails(False, "distilbert", True, None),
    "DistilGPT2": ModelDetails(True, "distilgpt2", False, None),
    "DistilGPT2 con ajuste fino": ModelDetails(True, "distilgpt2", True, None),
    "Gemini 2.0 Flash": ModelDetails(True, "gemini", False, None),
    "LXYuan": ModelDetails(False, "lxyuan", True, None),
    "TextDetox": ModelDetails(False, "textdetox", True, None),
}

ARGUMENTS = TrainingArguments(disable_tqdm=True, per_device_eval_batch_size=1, use_cpu=True)


@cache_resource(max_entries=1, show_spinner="Cargando modelos")
def load_models(
    model_key: str,
) -> (
    tuple[PredictionModel, PredictionModel]
    | tuple[Callable[[str], str], Callable[[str], str]]
):
    cache_resource.clear()
    model = MODELS[model_key]
    spec = spec_from_file_location(
        model.file_name, relative_path(f"../experiments/{model.file_name}.py")
    )
    if spec is None or spec.loader is None:
        raise Exception("Incorrect configuration")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    model_name = getattr(module, "model_name", None)
    fix_tokenizer = getattr(module, "fix_tokenizer", None)
    classification_prompter = getattr(module, "classification_prompter", None)
    detection_prompter = getattr(module, "detection_prompter", None)
    if model_name is None:
        return classification_prompter, detection_prompter  # type: ignore
    if model.decoder:
        from humor_detection.decoder import (
            classification_model,
            detection_model,
            load_model,
        )

        if not model.load_tuned:
            return PredictionModel(
                *classification_model(model_name),
                fix_tokenizer,
                classification_prompter,
            ), PredictionModel(
                *detection_model(model_name),
                fix_tokenizer,
                detection_prompter,
            )
    else:
        from humor_detection.encoder import load_model
    path = f"../tuned_models/{model.file_name}"
    return PredictionModel(
        *load_model(model_name, relative_path(f"{path}/classification")),
        fix_tokenizer,
        classification_prompter,
    ), PredictionModel(
        *load_model(model_name, relative_path(f"{path}/detection")),
        fix_tokenizer,
        detection_prompter,
    )


def plot_predictions(detection: DataFrame, classification: DataFrame | None):
    if detection["score_0"].isna().any():
        return
    detection["No humor"] = detection["score_0"]
    detection["Humor"] = detection["score_1"]
    with expander("### Probabilidad de humor"):
        bar_chart(detection.iloc[0][["Humor", "No humor"]].to_frame())
    if classification is None:
        return
    scores = classification.iloc[0].filter(like="score_")
    classification = scores.to_frame()
    classification.index = [int(col.split("_")[1]) + 1 for col in scores.index]  # type: ignore
    with expander("### Probabilidad de puntuación"):
        bar_chart(classification)


def predict(
    classification_model: PredictionModel | Callable[[str], str],
    detection_model: PredictionModel | Callable[[str], str],
    prompt: str,
    threshold: float | None,
    api_key: str | None,
):
    if isinstance(detection_model, FunctionType):
        detection = predict_llm([prompt], ["0", "1"], detection_model, api_key=api_key)
    else:
        detection = predict_detection(
            detection_model.model,
            detection_model.tokenizer,
            [prompt],
            ARGUMENTS,
            detection_model.prompter,
            threshold,
        )
    if detection["labels"].iloc[0] != 1:
        return detection, None
    if isinstance(classification_model, FunctionType):
        classification = predict_llm(
            [prompt], ["1", "2", "3", "4", "5"], classification_model, api_key=api_key
        )
    else:
        classification = predict_classification(
            classification_model.model,
            classification_model.tokenizer,
            [prompt],
            ARGUMENTS,
            classification_model.prompter,
        )
    return detection, classification
