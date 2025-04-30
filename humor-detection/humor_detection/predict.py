from .decoder import predict as predict_decoder
from .encoder import predict as predict_encoder
from .utils import CustomTrainer, optimize_arguments, setup
from pandas import DataFrame
from peft import PeftModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments
from typing import Callable


def predict(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None,
    threshold: float | None,
):
    optimize_arguments(arguments, model)
    token_ids, data_collator, preprocess_dataset, _ = setup(model, tokenizer, arguments)
    data = preprocess_dataset(
        DataFrame(
            {
                "text": prompts,
                "score": [0 if token_ids is None else ""] * len(prompts),
            }
        ),
        tokenizer,
        prompter,
    )
    prediction = CustomTrainer(
        model, arguments, data_collator, eval_dataset=data, token_ids=token_ids
    ).predict(
        data  # type: ignore
    )
    if token_ids is None:
        return predict_encoder(prediction, threshold)
    return predict_decoder(prediction, tokenizer, token_ids, threshold)


def predict_classification(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None = None,
):
    return predict(model, tokenizer, prompts, arguments, prompter, None)


def predict_detection(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None = None,
    threshold: float | None = None,
):
    return predict(model, tokenizer, prompts, arguments, prompter, threshold)
