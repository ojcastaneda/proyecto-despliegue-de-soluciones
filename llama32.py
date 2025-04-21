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

import sys
import huggingface_hub as hf

# Ejecutar script y pasar access token de HuggingFace.
# `python llama32.py <HF Access Token>`
# args = sys.argv[1:]
# hf_access_token = args[0]

# hf.login(hf_access_token)

model_name = "meta-llama/Llama-3.2-1B"
evaluation_batch_size = 5
arguments = TrainingArguments(
    bf16=True,
    bf16_full_eval=True,
    eval_strategy="epoch",
    num_train_epochs=2,
    optim="adamw_8bit",
    per_device_eval_batch_size=25,
    per_device_train_batch_size=25,
    save_strategy="no"
)
# lora = LoraConfig("CAUSAL_LM", lora_alpha=16, lora_dropout=0.1, r=128)
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
  # print("train_classification:")
  # train_classification(model, tokenizer, arguments, prompter=prompter)
  arguments.per_device_eval_batch_size = evaluation_batch_size
  print("test_classification:")
  print(test_classification(model, tokenizer, arguments, prompter))
  path = f"./models/llama32/classification"
  save_model(model, path)
  # model, _ = load_model(model_name, path)
  # print("predict_classification:")
  # print(predict_classification(model, tokenizer, prompts, arguments, prompter))


def run_detection():
  def prompter(input: str):
    return f"Detect if the following text is humor 1 or not 0:\n{input}\nScore:\n"

  model, tokenizer = detection_model(model_name)
  fix_tokenizer(tokenizer)
  # print("train_detection:")
  # train_detection(model, tokenizer, arguments, prompter=prompter)
  arguments.per_device_eval_batch_size = evaluation_batch_size
  print("test_detection:")
  print(test_detection(model, tokenizer, arguments, prompter))
  path = f"./models/llama32/detection"
  save_model(model, path)
  # model, _ = load_model(model_name, path)
  # print("predict_detection:")
  # print(predict_detection(model, tokenizer, prompts, arguments, prompter))


if __name__ == "__main__":
  run_classification()
  run_detection()
