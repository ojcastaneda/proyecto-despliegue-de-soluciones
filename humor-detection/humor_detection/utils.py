from .decoder import (
    preprocess_logits as preprocess_decoder_logits,
    setup as setup_decoder,
)
from .encoder import (
    preprocess_logits as preprocess_encoder_logits,
    setup as setup_encoder,
)
from datetime import datetime
from mlflow import (
    end_run,
    log_metric,
    log_params,
    set_experiment,
    set_tag,
    start_run,
)
from peft import PeftModel
from torch import Tensor, arange, full
from torch.nn import CrossEntropyLoss
from torch.nn.functional import pad
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


def set_random_seeds(seed=79):
    import random

    random.seed(seed)
    import numpy

    numpy.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seeds()


class CustomTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | PeftModel,
        *args,
        weights: Tensor | None = None,
        token_ids: Tensor | None,
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)
        self.loss_function = CrossEntropyLoss(weights)
        if token_ids is None:
            self.token_ids = None
        else:
            self.token_ids = token_ids
            self.lookup_token = full((max(self.token_ids) + 1,), -100)  # type: ignore
            self.lookup_token[self.token_ids] = arange(len(self.token_ids))
            model.loss_function = self.decoder_loss_function  # type: ignore

    def decoder_loss_function(
        self,
        logits: Tensor,
        labels: Tensor,
        vocab_size: int,
        shift_labels: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        if shift_labels is None:
            labels = pad(labels, (0, 1), value=-100)
            shift_labels = labels[..., 1:].contiguous()
        logits = logits[..., self.token_ids]
        mask = shift_labels != -100
        shift_labels[mask] = self.lookup_token[shift_labels[mask]]
        return self.loss_function(
            logits.view(-1, logits.shape[-1]), shift_labels.view(-1)
        )

    def train(self, *args, **kwargs):
        if self.loss_function.weight is not None:
            self.loss_function.weight = self.loss_function.weight.to(self.args.device)
        return super().train(*args, **kwargs)

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


def optimize_arguments(
    arguments: TrainingArguments, model: PreTrainedModel | PeftModel
):
    arguments.report_to = "none"
    arguments.eval_accumulation_steps = 1
    arguments.gradient_accumulation_steps = 1
    arguments.torch_empty_cache_steps = 1
    arguments.gradient_checkpointing = not isinstance(model, PeftModel)


def preprocess_logits(threshold: float | None, token_ids: Tensor | None):
    if token_ids is None:

        def preprocess(logits: Tensor, target: Tensor):
            return preprocess_encoder_logits(logits, target, threshold)

        return preprocess

    def preprocess(logits: Tensor, target: Tensor):
        return preprocess_decoder_logits(token_ids, logits, target, threshold)

    return preprocess


def setup(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    arguments: TrainingArguments,
):
    return (
        setup_encoder(tokenizer)
        if hasattr(model, "classifier")
        else setup_decoder(tokenizer, model.config.classes, arguments)  # type: ignore
    )


def log_metrics_mlflow(
    metrics: dict[str, float] | dict[str, dict[str, float]],
    parameters: dict,
    arguments: TrainingArguments,
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    train: bool,
):
    set_experiment("train" if train else "test")
    with start_run(run_name=f"{model.name_or_path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        set_tag("model_name", model.name_or_path)
        parameters["tokenizer_model"] = tokenizer.name_or_path
        if hasattr(model, "classifier"):
            set_tag("model_type", "encoder")
            classes = model.config.num_labels  # type: ignore
        else:
            set_tag("model_type", "decoder")
            classes = len(model.config.classes)  # type: ignore
            parameters["classes"] = str(model.config.classes)  # type: ignore
        set_tag("model_output", "classification" if classes == 5 else "detection")
        default_arguments = TrainingArguments()
        optimize_arguments(default_arguments, model)
        default_arguments = default_arguments.to_dict()
        log_params(
            {f"configuration__{key}": str(value) for key, value in parameters.items()}
        )
        log_params(
            {
                f"trainer__{key}": str(value)
                for key, value in arguments.to_dict().items()
                if value != default_arguments[key]
            }
        )
        for category in metrics:
            category_metrics = metrics[category]
            if not isinstance(category_metrics, dict):
                if category == "eval_loss":
                    log_metric(f"loss", category_metrics)
                continue
            for metric in category_metrics:
                if metric.endswith("support"):
                    continue
                log_metric(f"{category}__{metric}", category_metrics[metric])
        end_run()
