# Databricks notebook source
# MAGIC %md
# MAGIC # Brian & Ajmal's code
# MAGIC
# MAGIC https://github.com/Data-drone/ANZ_LLM_Bootcamp/blob/master/3.2_Setting_Up_Inference_Model.py

# COMMAND ----------

# MAGIC %md
# MAGIC # Brian & Ajmal's code
# MAGIC
# MAGIC https://github.com/Data-drone/ANZ_LLM_Bootcamp/blob/master/3.2_Setting_Up_Inference_Model.py

# COMMAND ----------

# MAGIC %pip install -Uq bitsandbytes==0.40.0 transformers==4.31.0
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -Uq bitsandbytes==0.40.0 transformers==4.31.0
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   AutoConfig,
   pipeline
)

import huggingface_hub
import mlflow
import torch

HF_KEY = "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"

HF_MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
HF_MODEL_REVISION_ID = "0ba94ac9b9e1d5a0037780667e8b219adde1908c"

# COMMAND ----------

from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   AutoConfig,
   pipeline
)

import huggingface_hub
import mlflow
import torch

HF_KEY = "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"

HF_MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
HF_MODEL_REVISION_ID = "0ba94ac9b9e1d5a0037780667e8b219adde1908c"

# COMMAND ----------

huggingface_hub.login(token=HF_KEY)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

model_config = AutoConfig.from_pretrained(HF_MODEL_NAME,
                                          trust_remote_code=True, # this can be needed if we reload from cache
                                          revision=HF_MODEL_REVISION_ID
                                      )

model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME,
                                               revision=HF_MODEL_REVISION_ID,
                                               trust_remote_code=True, # this can be needed if we reload from cache
                                               config=model_config,
                                               device_map='auto',
                                               load_in_8bit=True
                                              )
  
pipe = pipeline(
  "text-generation", 
  model=model, 
  tokenizer=tokenizer,
  # Added-in by VV
  return_full_text=False,
)

inference_config = {
   "do_sample": True,
   "max_new_tokens": 512
}

example_sentences = ["<s>[INST]<<SYS>>Answer questions succintly<</SYS>> Who are you?[/INST]", 
                    "<s>[INST]<<SYS>>Answer questions succintly<</SYS>> How can you help me?[/INST]"]

# COMMAND ----------

huggingface_hub.login(token=HF_KEY)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

model_config = AutoConfig.from_pretrained(HF_MODEL_NAME,
                                          trust_remote_code=True, # this can be needed if we reload from cache
                                          revision=HF_MODEL_REVISION_ID
                                      )

model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME,
                                               revision=HF_MODEL_REVISION_ID,
                                               trust_remote_code=True, # this can be needed if we reload from cache
                                               config=model_config,
                                               device_map='auto',
                                               load_in_8bit=True
                                              )
  
pipe = pipeline(
  "text-generation", 
  model=model, 
  tokenizer=tokenizer,
  # Added-in by VV
  return_full_text=False,
)

inference_config = {
   "do_sample": True,
   "max_new_tokens": 512
}

example_sentences = ["<s>[INST]<<SYS>>Answer questions succintly<</SYS>> Who are you?[/INST]", 
                    "<s>[INST]<<SYS>>Answer questions succintly<</SYS>> How can you help me?[/INST]"]

# COMMAND ----------

generate_kwargs = {
  "max_new_tokens": 2048
}

print(pipe("<s>[INST]<<SYS>>Answer questions succintly<</SYS>> How can you help me?[/INST]", **generate_kwargs))

# COMMAND ----------

generate_kwargs = {
  "max_new_tokens": 2048
}

print(pipe("<s>[INST]<<SYS>>Answer questions succintly<</SYS>> How can you help me?[/INST]", **generate_kwargs))

# COMMAND ----------

MLFLOW_MODEL_NAME = "vv-transformers-llama2-13b"

# Works!
with mlflow.start_run() as run:  
    mlflow.transformers.log_model(
      pipe,
      artifact_path="model",
      registered_model_name=MLFLOW_MODEL_NAME,
      #signature=embedding_signature,
      input_example=example_sentences,
      inference_config=inference_config,
      pip_requirements={
        'bitsandbytes==0.39.1',
        'transformers==4.31.0'
      },
      await_registration_for=1000,
    )

# COMMAND ----------

MLFLOW_MODEL_NAME = "vv-transformers-llama2-13b"

# Works!
with mlflow.start_run() as run:  
    mlflow.transformers.log_model(
      pipe,
      artifact_path="model",
      registered_model_name=MLFLOW_MODEL_NAME,
      #signature=embedding_signature,
      input_example=example_sentences,
      inference_config=inference_config,
      pip_requirements={
        'bitsandbytes==0.39.1',
        'transformers==4.31.0'
      },
      await_registration_for=1000,
    )

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

model_info = client.get_registered_model(MLFLOW_MODEL_NAME)
print(model_info)
latest_version = model_info.latest_versions[0]
latest_run_id = latest_version.run_id
experiment_id = re.search(r"mlflow-tracking/([a-f0-9]{32})", latest_version.source).group(1)

link_to_model_registry = f"#mlflow/models/{MLFLOW_MODEL_NAME}"
link_to_latest_run = f"#mlflow/experiments/{experiment_id}/runs/{latest_run_id}/artifactPath/model"

displayHTML(f'''
  <a href="{link_to_model_registry}">View the model in Model Registry</a><br/>
  <a href="{link_to_latest_run}">View the latest run artifacts</a>
''')

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

model_info = client.get_registered_model(MLFLOW_MODEL_NAME)
print(model_info)
latest_version = model_info.latest_versions[0]
latest_run_id = latest_version.run_id
experiment_id = re.search(r"mlflow-tracking/([a-f0-9]{32})", latest_version.source).group(1)

link_to_model_registry = f"#mlflow/models/{MLFLOW_MODEL_NAME}"
link_to_latest_run = f"#mlflow/experiments/{experiment_id}/runs/{latest_run_id}/artifactPath/model"

displayHTML(f'''
  <a href="{link_to_model_registry}">View the model in Model Registry</a><br/>
  <a href="{link_to_latest_run}">View the latest run artifacts</a>
''')

# COMMAND ----------

# # DBTITLE 1,Setting Up the mlflow experiment
# #Enable Unity Catalog with mlflow registry
# mlflow.set_registry_uri('databricks-uc')

# try:
#   mlflow.create_experiment(experiment_name)
# except mlflow.exceptions.RestException:
#   print('experiment exists already')

# mlflow.set_experiment(experiment_name)

# client = mlflow.MlflowClient()

# # LLama 2 special type currently not supported
# # embedding_signature = mlflow.models.infer_signature(
# #     model_input=example_sentences,
# #     model_output=pipe(example_sentences)
# # )

# COMMAND ----------

# # DBTITLE 1,Setting Up the mlflow experiment
# #Enable Unity Catalog with mlflow registry
# mlflow.set_registry_uri('databricks-uc')

# try:
#   mlflow.create_experiment(experiment_name)
# except mlflow.exceptions.RestException:
#   print('experiment exists already')

# mlflow.set_experiment(experiment_name)

# client = mlflow.MlflowClient()

# # LLama 2 special type currently not supported
# # embedding_signature = mlflow.models.infer_signature(
# #     model_input=example_sentences,
# #     model_output=pipe(example_sentences)
# # )

# COMMAND ----------

# MAGIC %md
# MAGIC # VV Code

# COMMAND ----------

# MAGIC %md
# MAGIC # VV Code

# COMMAND ----------

# MAGIC %pip install -Uq \
# MAGIC   bitsandbytes==0.40.0 \
# MAGIC   transformers==4.31.0 \
# MAGIC   triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python \
# MAGIC   xformers
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -Uq \
# MAGIC   bitsandbytes==0.40.0 \
# MAGIC   transformers==4.31.0 \
# MAGIC   triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python \
# MAGIC   xformers
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   AutoConfig,
   pipeline
)

import huggingface_hub
import mlflow
import torch

import accelerate
import pandas as pd
import mlflow
import numpy as np
import re
import torch
import transformers

HF_KEY = "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"

HF_MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
HF_MODEL_REVISION_ID = "0ba94ac9b9e1d5a0037780667e8b219adde1908c"

# COMMAND ----------

from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   AutoConfig,
   pipeline
)

import huggingface_hub
import mlflow
import torch

import accelerate
import pandas as pd
import mlflow
import numpy as np
import re
import torch
import transformers

HF_KEY = "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"

HF_MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
HF_MODEL_REVISION_ID = "0ba94ac9b9e1d5a0037780667e8b219adde1908c"

# COMMAND ----------

huggingface_hub.login(token=HF_KEY)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, padding_side="left")

model_config = AutoConfig.from_pretrained(
  HF_MODEL_NAME,
  revision=HF_MODEL_REVISION_ID,
  trust_remote_code=True, # this can be needed if we reload from cache
)

model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME,
        revision=HF_MODEL_REVISION_ID,
        trust_remote_code=True,
        config=model_config,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        # pad_token_id=tokenizer.eos_token_id,
)
model.tie_weights()

pipe = pipeline(
  "text-generation", 
  model=model, 
  tokenizer=tokenizer,
  # device_map='auto',
  return_full_text=False,
  framework="pt",
  # clean_up_tokenization_spaces=True,
  # prefix=f"{SYSTEM_PROMPT}{USER_INSTRUCTION_PREFIX}"
)

# COMMAND ----------

huggingface_hub.login(token=HF_KEY)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, padding_side="left")

model_config = AutoConfig.from_pretrained(
  HF_MODEL_NAME,
  revision=HF_MODEL_REVISION_ID,
  trust_remote_code=True, # this can be needed if we reload from cache
)

model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME,
        revision=HF_MODEL_REVISION_ID,
        trust_remote_code=True,
        config=model_config,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        # pad_token_id=tokenizer.eos_token_id,
)
model.tie_weights()

pipe = pipeline(
  "text-generation", 
  model=model, 
  tokenizer=tokenizer,
  # device_map='auto',
  return_full_text=False,
  framework="pt",
  # clean_up_tokenization_spaces=True,
  # prefix=f"{SYSTEM_PROMPT}{USER_INSTRUCTION_PREFIX}"
)

# COMMAND ----------

MLFLOW_MODEL_NAME = "vv-transformers-llama2-13b"

with mlflow.start_run() as run:  
    mlflow.transformers.log_model(
        pipe,
        "model",
        task="text-generation",
        registered_model_name=MLFLOW_MODEL_NAME,
        inference_config = {
          "max_length": 2048,
          "include_prompt": False,
          "collapse_whitespace": True,
          'max_new_tokens': 2048,
          'temperature': 0.1,
          'top_p': 0.65,
          'top_k': 20,
          'repetition_penalty': 1.2,
          'no_repeat_ngram_size': 0,
          'use_cache': False,
          'do_sample': True,
          'eos_token_id': tokenizer.eos_token_id,
          'pad_token_id': tokenizer.eos_token_id,
        },
        pip_requirements={
          "accelerate",
          'bitsandbytes==0.39.1',
          "sentencepiece",
          "torch"
          'transformers==4.31.0',
        },
        await_registration_for=1000,
    )


# COMMAND ----------

MLFLOW_MODEL_NAME = "vv-transformers-llama2-13b"

with mlflow.start_run() as run:  
    mlflow.transformers.log_model(
        pipe,
        "model",
        task="text-generation",
        registered_model_name=MLFLOW_MODEL_NAME,
        inference_config = {
          "max_length": 2048,
          "include_prompt": False,
          "collapse_whitespace": True,
          'max_new_tokens': 2048,
          'temperature': 0.1,
          'top_p': 0.65,
          'top_k': 20,
          'repetition_penalty': 1.2,
          'no_repeat_ngram_size': 0,
          'use_cache': False,
          'do_sample': True,
          'eos_token_id': tokenizer.eos_token_id,
          'pad_token_id': tokenizer.eos_token_id,
        },
        pip_requirements={
          "accelerate",
          'bitsandbytes==0.39.1',
          "sentencepiece",
          "torch"
          'transformers==4.31.0',
        },
        await_registration_for=1000,
    )


# COMMAND ----------

client = mlflow.tracking.MlflowClient()

model_info = client.get_registered_model(MLFLOW_MODEL_NAME)
latest_version = model_info.latest_versions[0]
latest_run_id = latest_version.run_id
experiment_id = re.search(r"mlflow-tracking/([a-f0-9]{32})", latest_version.source).group(1)

link_to_model_registry = f"#mlflow/models/{MLFLOW_MODEL_NAME}"
link_to_latest_run = f"#mlflow/experiments/{experiment_id}/runs/{latest_run_id}/artifactPath/model"

displayHTML(f'''
  <a href="{link_to_model_registry}">View the model in Model Registry</a><br/>
  <a href="{link_to_latest_run}">View the latest run artifacts</a>
''')

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

model_info = client.get_registered_model(MLFLOW_MODEL_NAME)
latest_version = model_info.latest_versions[0]
latest_run_id = latest_version.run_id
experiment_id = re.search(r"mlflow-tracking/([a-f0-9]{32})", latest_version.source).group(1)

link_to_model_registry = f"#mlflow/models/{MLFLOW_MODEL_NAME}"
link_to_latest_run = f"#mlflow/experiments/{experiment_id}/runs/{latest_run_id}/artifactPath/model"

displayHTML(f'''
  <a href="{link_to_model_registry}">View the model in Model Registry</a><br/>
  <a href="{link_to_latest_run}">View the latest run artifacts</a>
''')

# COMMAND ----------

PROMPT_TEMPLATE = """
Blah blah
{{"x": [{{"y": 0}}]}}

{instruction}
"""

print(PROMPT_TEMPLATE.format(instruction="xyz"))

# COMMAND ----------

PROMPT_TEMPLATE = """
Blah blah
{{"x": [{{"y": 0}}]}}

{instruction}
"""

print(PROMPT_TEMPLATE.format(instruction="xyz"))

# COMMAND ----------



# COMMAND ----------


