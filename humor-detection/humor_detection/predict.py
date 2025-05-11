from .decoder import predict as predict_decoder
from .encoder import predict as predict_encoder
from .utils import CustomTrainer
from pandas import DataFrame
from peft import PeftModel
from scipy.special import softmax
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
    trainer = CustomTrainer(model, tokenizer, arguments, prompter=prompter)
    score = 0 if trainer.token_ids is None else None
    dataset = trainer.preprocess_dataset(
        DataFrame({"text": prompts, "score": [score] * len(prompts)})
    )
    prediction = trainer.predict(dataset)  # type: ignore
    if trainer.token_ids is None:
        logits = prediction.predictions
        labels = predict_encoder(prediction, threshold)
    else:
        labels, logits = predict_decoder(
            dataset, prediction, trainer.lookup_token.cpu().numpy(), threshold
        )
    output = {
        f"score_{i}": list(col)
        for i, col in enumerate(zip(*softmax(logits, -1).tolist()))
    }
    output["labels"] = labels.tolist()
    return DataFrame(output)


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
