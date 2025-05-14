from copy import deepcopy
from os import environ
from typing import Callable, Literal
from numpy import array, max
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.metrics import classification_report
from datetime import datetime
from mlflow import (
    end_run,
    log_metric,
    log_params,
    set_experiment,
    set_tag,
    set_tracking_uri,
    start_run,
)
from os.path import abspath, dirname, join
from peft import PeftModel
from torch import Tensor, arange, full, tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import pad
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import sys


def set_random_seeds(seed=79):
    import random

    random.seed(seed)
    import numpy

    numpy.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


best_model_metrics = {
    "macro_f1": "macro_avg_f1-score",
    "weighted_f1": "weighted_avg_f1-score",
    "accuracy": "accuracy",
}


class CustomTrainer(Trainer):
    train_logs: dict[str, list[float]] | None = None
    eval_metrics: dict[str, float] | None = None
    best_model: dict | None = None
    best_model_metric: str | None
    token_ids: NDArray | None = None

    def __init__(
        self,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerBase,
        arguments: TrainingArguments,
        train_dataset: DataFrame | None = None,
        eval_dataset: DataFrame | None = None,
        prompter: Callable[[str], str] | None = None,
        best_model_metric: Literal["macro_f1", "weighted_f1", "accuracy"] | None = None,
        class_weights: list[float] | None = None,
        threshold: float | None = None,
        **kwargs,
    ):
        is_prediction = train_dataset is None and eval_dataset is None
        optimize_arguments(arguments, model, is_prediction)
        classes: list[str] | None = model.config.classes if hasattr(model.config, "classes") else None  # type: ignore
        if class_weights is not None:
            weights: list[float] = [0.0] * (
                model.config.num_labels if classes is None else len(classes)  # type: ignore
            )
            for i in range(len(class_weights)):
                weights[i] = float(class_weights[i])
            self.loss_function = CrossEntropyLoss(tensor(weights))
        else:
            self.loss_function = CrossEntropyLoss()
        self.best_model_metric = best_model_metrics.get(best_model_metric or "", None)
        if classes is None:
            from .encoder import compute_metrics, preprocess_dataset

            compute_metrics = compute_metrics(threshold)
            self.preprocess_dataset = preprocess_dataset(tokenizer, prompter)
        else:
            from .decoder import (
                LastTokenCollator,
                compute_metrics,
                preprocess_dataset,
                preprocess_logits,
            )

            if not is_prediction:
                kwargs["data_collator"] = LastTokenCollator(tokenizer, mlm=False)
            self.token_ids = array(
                [
                    tokenizer.encode(token, add_special_tokens=False)[0]
                    for token in classes
                ]
            )
            self.lookup_token = full((max(self.token_ids) + 1,), -100)  # type: ignore
            self.lookup_token[self.token_ids] = arange(len(self.token_ids))
            compute_metrics = compute_metrics(self.lookup_token.numpy(), threshold)
            model.loss_function = self.decoder_loss_function  # type: ignore
            self.preprocess_dataset = preprocess_dataset(
                tokenizer, prompter, classes, model.config  # type: ignore
            )
            kwargs["preprocess_logits_for_metrics"] = preprocess_logits(self.token_ids)
        if train_dataset is not None:
            kwargs["train_dataset"] = self.preprocess_dataset(train_dataset)
        if eval_dataset is not None:
            kwargs["eval_dataset"] = self.preprocess_dataset(eval_dataset)
        super().__init__(
            model,
            arguments,
            compute_metrics=None if is_prediction else compute_metrics,
            processing_class=tokenizer,
            **kwargs,
        )

    def train(self, *args, **kwargs):
        self.best_model = None
        self.train_logs = {
            "train_loss": [],
            "loss": [],
            "accuracy": [],
            "macro_f1": [],
            "weighted_f1": [],
        }
        self.eval_metrics = {}
        if self.loss_function.weight is not None:
            self.loss_function.weight = self.loss_function.weight.to(self.args.device)
        output = super().train(*args, **kwargs), self.train_logs, self.eval_metrics
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        return output

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Tensor],
        return_outputs=False,
        num_items_in_batch: int | None = None,
    ):
        if self.token_ids is None:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = self.loss_function(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

        if self.lookup_token.device != self.args.device:
            self.lookup_token = self.lookup_token.to(self.args.device)

        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def log(self, logs: dict[str, float], start_time: float | None = None):
        output = super().log(logs, start_time)
        if self.train_logs is None or self.eval_metrics is None:
            return output
        if "loss" in logs:
            self.train_logs["train_loss"].append(logs["loss"])
        if "eval_accuracy" in logs:
            self.train_logs["loss"].append(logs["eval_loss"])
            self.train_logs["accuracy"].append(logs["eval_accuracy"])
            self.train_logs["macro_f1"].append(logs["eval_macro_avg_f1-score"])
            self.train_logs["weighted_f1"].append(logs["eval_weighted_avg_f1-score"])
            if self.best_model_metric is None:
                self.eval_metrics = logs
                return output
            if (self.best_model_metric not in self.eval_metrics) or (
                logs[f"eval_{self.best_model_metric}"]
                > self.eval_metrics[self.best_model_metric]
            ):
                self.eval_metrics = {
                    key.replace("eval_", ""): value for key, value in logs.items()
                }
                self.model.cpu()
                self.best_model = deepcopy(self.model.state_dict())
                self.model.to(self.args.device)  # type: ignore
        return output

    def decoder_loss_function(
        self,
        logits: Tensor,
        labels: Tensor,
        vocab_size: int,
        ignore_index: int = -100,
        shift_labels: Tensor | None = None,
        **kwargs,
    ):
        if shift_labels is None:
            labels = pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
        logits = logits[..., self.token_ids]
        mask = shift_labels != ignore_index
        shift_labels[mask] = self.lookup_token[shift_labels[mask]]
        return self.loss_function(
            logits.view(-1, logits.shape[-1]), shift_labels.view(-1)
        )


def optimize_arguments(
    arguments: TrainingArguments, model: PreTrainedModel | PeftModel, predict=False
):
    arguments.set_save("no")
    arguments.set_logging("no" if predict else "epoch")
    arguments.set_evaluate(
        "no" if predict else "epoch",
        batch_size=arguments.per_device_eval_batch_size,
        accumulation_steps=1,
        delay=0,
    )
    arguments.logging_dir = None
    arguments.report_to = "none"
    arguments.gradient_accumulation_steps = 1
    arguments.torch_empty_cache_steps = 1
    arguments.gradient_checkpointing = not isinstance(model, PeftModel)


def log_metrics_mlflow(
    metrics: dict[str, float],
    parameters: dict,
    arguments: TrainingArguments,
    model: PreTrainedModel | PeftModel | tuple[str, list[str]],
    tokenizer: PreTrainedTokenizerBase | None,
    prefix: str,
):
    url = environ.get("ML_FLOW_URL")
    if url is not None and url.strip() != "":
        set_tracking_uri(url)
    if isinstance(model, tuple):
        model_name = model[0]
        model_type = "decoder"
        classes = len(model[1])
        parameters["classes"] = str(model[1])
    else:
        model_name = str(model.name_or_path)
        if hasattr(model, "classifier"):
            model_type = "encoder"
            classes = model.config.num_labels  # type: ignore
        else:
            model_type = "decoder"
            classes = len(model.config.classes)  # type: ignore
            parameters["classes"] = str(model.config.classes)  # type: ignore
    set_experiment(f"{prefix}_{model_name}")
    model_output = "classification" if classes == 5 else "detection"
    default_arguments = TrainingArguments()
    if not isinstance(model, tuple):
        optimize_arguments(default_arguments, model)
    default_arguments = default_arguments.to_dict()
    if tokenizer is not None and model_name != tokenizer.name_or_path:
        parameters["tokenizer_model"] = tokenizer.name_or_path
    time = datetime.now().strftime("%H:%M:%S-%d/%m/%Y")
    with start_run(run_name=f"{model_type}_{model_output}_{time}"):
        set_tag("model_type", model_type)
        set_tag("model_output", model_output)
        log_params(
            {f"configuration_{key}": str(value) for key, value in parameters.items()}
        )
        log_params(
            {
                f"trainer_{key}": str(value)
                for key, value in arguments.to_dict().items()
                if value != default_arguments[key]
            }
        )
        for key in metrics:
            if key.endswith(("accuracy", "f1-score", "loss", "precision", "recall")):
                log_metric(key.replace("eval_", ""), metrics[key])
        end_run()


def calculate_metrics(labels: list[int] | NDArray, predictions: list[int] | NDArray):
    metrics = classification_report(
        labels,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    flatten = {}
    if isinstance(metrics, str):
        return flatten
    for category in metrics:
        metric = metrics[category]
        if not isinstance(metric, dict):
            flatten[category.replace(" ", "_")] = metric
            continue
        for key in metric:
            flatten[f"{category}_{key}".replace(" ", "_")] = metric[key]
    return flatten


def relative_path(path: str):
    try:
        get_ipython  # type: ignore
        return path
    except NameError:
        return join(dirname(abspath(sys.argv[0])), path)
