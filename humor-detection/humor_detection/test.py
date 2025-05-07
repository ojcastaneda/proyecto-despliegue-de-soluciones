from .dataset import Test, load_csv
from .utils import CustomTrainer, log_metrics_mlflow
from pandas import DataFrame
from peft import PeftModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments
from typing import Callable


def test(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: DataFrame,
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None,
    threshold: float | None,
):
    trainer = CustomTrainer(
        model,
        tokenizer,
        arguments,
        eval_dataset=dataset,
        prompter=prompter,
        threshold=threshold,
    )
    output = trainer.evaluate()
    log_metrics_mlflow(
        output,
        {
            "threshold": threshold,
            "prompter": None if prompter is None else repr(prompter("<PLACEHOLDER>")),
        },
        arguments,
        model,
        tokenizer,
        False,
    )
    return output


def test_classification(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None = None,
):
    return test(
        model, tokenizer, load_csv(Test.classification_path), arguments, prompter, None
    )


def test_detection(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    arguments: TrainingArguments,
    prompter: Callable[[str], str] | None = None,
    threshold: float | None = None,
):
    return test(
        model, tokenizer, load_csv(Test.detection_path), arguments, prompter, threshold
    )
