from .decoder import (
    preprocess_logits as preprocess_decoder_logits,
    setup as setup_decoder,
)
from .encoder import (
    preprocess_logits as preprocess_encoder_logits,
    setup as setup_encoder,
)
from peft import PeftModel
from torch import Tensor
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments


classification_classes = ["0", "1", "2", "3", "4"]
detection_classes = ["0", "1"]


def set_random_seeds(seed=79):
    import random

    random.seed(seed)
    import numpy

    numpy.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seeds()


def optimize_arguments(
    arguments: TrainingArguments, model: PreTrainedModel | PeftModel
):
    arguments.eval_accumulation_steps = 1
    arguments.gradient_accumulation_steps = 1
    arguments.torch_empty_cache_steps = 1
    arguments.gradient_checkpointing = not isinstance(model, PeftModel)


def preprocess_logits(tokens: list[int] | None = None):
    if tokens is None:
        return preprocess_encoder_logits

    def preprocess(logits: Tensor, target: Tensor):
        return preprocess_decoder_logits(tokens, logits, target)

    return preprocess


def setup(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    classes: list[str],
):
    return (
        setup_encoder(tokenizer)
        if hasattr(model, "classifier")
        else setup_decoder(tokenizer, classes)
    )
