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

import os
import uuid
import mlflow

print(f"MLFLOW_SERVER_URI={os.environ.get("MLFLOW_SERVER_URI")}")
mlflow.set_tracking_uri(os.environ.get("MLFLOW_SERVER_URI"))


def fix_metrics_mlflow(metrics: dict):
    fixed_metrics = {}
    for key, value in metrics.items():
      if isinstance(value, dict):
        for subkey, subvalue in value.items():
          new_key = f"{key} {subkey}"
          fixed_metrics[new_key] = subvalue
      else:
        fixed_metrics[key] = value
    
    return fixed_metrics


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


def fix_tokenizer(tokenizer: PreTrainedTokenizerBase):
  tokenizer.pad_token = tokenizer.eos_token


def run_classification():
  def prompter(input: str):
    return f"Califica el nivel de humor de 0 a 4 del siguiente texto:\n{input}\nCalificación:\n"

  
  experiment_name = f"{model_name}-classification"
  mlflow.set_experiment(experiment_name)

  run_id = uuid.uuid4()
  run_name = f"{model_name}-classification-{run_id}"

  with mlflow.start_run(run_name=run_name):
    print(f"Model {model_name} Classification Run {run_id}:")
    mlflow.set_tags({
      'model_name': model_name,
      'model_type': 'decoder',
      'experiment_type': 'classification'
    })

    model, tokenizer = classification_model(model_name)

    fix_tokenizer(tokenizer)

    arguments.per_device_eval_batch_size = evaluation_batch_size

    mlflow.log_param("batch_size", evaluation_batch_size)
    mlflow.log_param("prompt_template", prompter("<INPUT>"))

    metrics = test_classification(model, tokenizer, arguments, prompter)
    print(metrics)

    mlflow.log_metrics(fix_metrics_mlflow(metrics))
    
    # path = f"./models/llama32/classification"
    # save_model(model, path)


def run_detection():
  def prompter(input: str):
    return f"Detecta si el siguiente texto es humor 1 o no 0:\n{input}\nCalificación:\n"
  
  experiment_name = f"{model_name}-detection"
  mlflow.set_experiment(experiment_name)

  run_id = uuid.uuid4()
  run_name = f"{model_name}-detection-{run_id}"

  with mlflow.start_run(run_name=run_name):
    print(f"Model {model_name} Detection Run {run_id}:")

    mlflow.set_tags({
        'model_name': model_name,
        'model_type': 'decoder',
        'experiment_type': 'detection'
    })
    
    model, tokenizer = detection_model(model_name)

    fix_tokenizer(tokenizer)

    arguments.per_device_eval_batch_size = evaluation_batch_size

    mlflow.log_param("batch_size", evaluation_batch_size)
    mlflow.log_param("prompt_template", prompter("<INPUT>"))

    metrics = test_detection(model, tokenizer, arguments, prompter)
    print(metrics)

    mlflow.log_metrics(fix_metrics_mlflow(metrics))
    
    # path = f"./models/llama32/detection"
    # save_model(model, path)


if __name__ == "__main__":
  run_classification()
  run_detection()
