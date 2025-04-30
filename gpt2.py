from humor_detection.decoder import (
    classification_model,
    detection_model,
    load_model,
    save_model,
)
from humor_detection.test import test_classification, test_detection
from humor_detection.train import train_classification, train_detection
from humor_detection.predict import predict_classification, predict_detection
from peft import LoraConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments


# model_name = "openai-community/gpt2-medium"
model_name = "openai-community/gpt2"
arguments = TrainingArguments(
    bf16=True,
    bf16_full_eval=True,
    eval_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=2,
    # weight_decay=5e-6,
    optim="adamw_8bit",
    lr_scheduler_type="cosine_with_restarts",
    per_device_eval_batch_size=15,
    per_device_train_batch_size=30,
    save_strategy="no",
)
# lora = LoraConfig(lora_alpha=8, lora_dropout=0.1, r=16, task_type="CAUSAL_LM")
prompts = [
    "¿Cuál es el último animal que subió al arca de Noé? El del-fin.",
    "El otro día unas chicas llamarón a mi puerta y me pidieron una pequeña donación para una piscina local.\nLes di un garrafa de agua.",
    "The brain surgeon changed my life. He really opened my mind.",
    "djasndoasndoa",
    "jajaja",
]


def fix_tokenizer(tokenizer: PreTrainedTokenizerBase):
    tokenizer.pad_token = tokenizer.eos_token


def run_classification():
    def prompter(input: str):
        return f"Give a humor rating 1 to 5 for the following text:\n{input}\nScore:\n"

    model, tokenizer = classification_model(model_name)
    fix_tokenizer(tokenizer)
    train_classification(
        model,
        tokenizer,
        arguments,
        prompter=prompter,
        full_dataset=True,
        class_weights=[1, 1.25, 1.25, 2.6, 8],
    )
    path = f"./models/gpt2/classification"
    save_model(model, path)
    model, _ = load_model(model_name, path)
    # print(predict_classification(model, tokenizer, prompts, arguments, prompter))


def run_detection():
    def prompter(input: str):
        return f"Detect if the following text is humor 1 or not 0:\n{input}\nScore:\n"

    model, tokenizer = detection_model(model_name)
    fix_tokenizer(tokenizer)
    train_detection(
        model,
        tokenizer,
        arguments,
        prompter=prompter,
        full_dataset=True,
        sample="under",
        threshold=0.85,
    )
    path = f"./models/gpt2/detection"
    save_model(model, path)
    model, _ = load_model(model_name, path)
    # print(predict_detection(model, tokenizer, prompts, arguments, prompter))


if __name__ == "__main__":
    # run_classification()
    run_detection()
