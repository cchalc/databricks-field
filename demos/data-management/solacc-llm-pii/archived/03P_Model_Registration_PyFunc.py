# Databricks notebook source
# MAGIC %md
# MAGIC # Model Registration (PyFunc)
# MAGIC
# MAGIC Now that we have got a working model and have found a prompt that works for us, we want to do three things:
# MAGIC - Log the model with MLflow: packages up the model artifacts for easy retrieval and deployment
# MAGIC
# MAGIC - Register the model in Model Registry: centralises management of the full lifecycle of the model. This is the point from which we can confidently retrieve the latest working version for deployment in UDFs or model serving 
# MAGIC
# MAGIC - Deploy the model to GPU Model Serving

# COMMAND ----------

# DBTITLE 1,Install required libraries (covered in init script)
# If you don't want to repeat loading these libraries across notebooks, add the below to your cluster init script

# %pip install -Uq accelerate==0.23.0 bitsandbytes==0.41.1 mlflow transformers==4.33.2 xformers==0.0.21 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python 

# # Restart the Python kernel
# dbutils.library.restartPython()

# COMMAND ----------

import accelerate
import pandas as pd
import mlflow
import numpy as np
import re
import sentencepiece
import torch
import transformers

from huggingface_hub import snapshot_download, login, notebook_login
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from config import HF_MODEL_NAME, HF_MODEL_REVISION_ID, CACHE_PATH, DOWNLOAD_PATH, MLFLOW_MODEL_NAME_P, TEST_DATA, PROMPT_TEMPLATE_V2, SYSTEM_PROMPT, USER_INSTRUCTION

# Disable progress bars for cleaner output
transformers.utils.logging.disable_progress_bar()

# COMMAND ----------

# MAGIC %md
# MAGIC # Hugging Face Login
# MAGIC
# MAGIC Because we needed to first register to gain access to Llama 2 models, we need to authenticate with Hugging Face to verify we are able to access the model. 
# MAGIC
# MAGIC You need to provide your Hugging Face token. There are two ways to login:
# MAGIC - `notebook_login()`: UI-driven login
# MAGIC - `login()`: programatically login (it is recommended your token is saved in Databricks Secrets)

# COMMAND ----------

# notebook_login()
token = dbutils.secrets.get(scope="tokens", key="hf_token_vv")
login(token=token)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create LLM Pipeline
# MAGIC
# MAGIC We utilise our code from the previous Notebook to load our LLM pipeline
# MAGIC
# MAGIC - `AutoTokenizer` ([docs](https://huggingface.co/docs/transformers/v4.33.3/en/model_doc/auto#transformers.AutoTokenizer)) will retrieve the tokenizer used during the training of the model. It is recommended you utilise the same tokenizer that the model was trained on.
# MAGIC - `AutoConfig` ([docs](https://huggingface.co/docs/transformers/v4.33.3/en/model_doc/auto#transformers.AutoConfig)) XXXX
# MAGIC - `AutoModelForCausalLM` ([docs](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM)) will automatically load the configuration for the given model. Here, since we are using Llama 2 13B, it will retrieve a `LlamaForCausalLM` ([docs](https://huggingface.co/docs/transformers/v4.33.3/en/model_doc/llama2#transformers.LlamaForCausalLM)) object
# MAGIC - `transformers.pipeline` XXXX

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download model artificats
# MAGIC
# MAGIC We'll first download a snapshot of the model artifacts. By default, `snapshot_download` downloads the artifacts to a cache on local disk. However, if the cluster shuts down or we wish to run this Notebook on another cluster, the cache is lost and we need to download everything all over again. So we explicitly set the `cache_dir` to point to a Workspace location, so that the cache is persisted across sessions

# COMMAND ----------

CACHE_PATH = "/Users/vinny.vijeyakumaar@databricks.com/hf_cache"

snapshot_location = snapshot_download(
  repo_id=HF_MODEL_NAME, 
  revision=HF_MODEL_REVISION_ID, 
  ignore_patterns="*.bin",
  # ignore_patterns="*.safetensors",
  cache_dir=CACHE_PATH
)

print(f"Cache folder: {snapshot_location}")

# COMMAND ----------

# DBTITLE 1,Safetensors ON
tokenizer = AutoTokenizer.from_pretrained(snapshot_location, padding_side="left")

model_config = AutoConfig.from_pretrained(
  snapshot_location,
  # revision=HF_MODEL_REVISION_ID,
  # cache_dir=CACHE_PATH,
  trust_remote_code=True, # this can be needed if we reload from cache
)

model = AutoModelForCausalLM.from_pretrained(snapshot_location,
        # revision=HF_MODEL_REVISION_ID,
        use_safetensors=True,
        # use_safetensors=False,
        trust_remote_code=True,
        config=model_config,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        # torch_dtype=torch.bfloat16,
        # device_map='auto',
        # pad_token_id=tokenizer.eos_token_id,
)
model.tie_weights()

pipe = transformers.pipeline(
  "text-generation", 
  model=model, 
  tokenizer=tokenizer,
  device_map='auto',
  return_full_text=False,
  torch_dtype=torch.bfloat16,
)

# Required tokenizer setting for batch inference
pipe.tokenizer.pad_token_id = tokenizer.eos_token_id
pipe.tokenizer.eos_token_id = tokenizer.eos_token_id

print(f"EOS token ID: {tokenizer.eos_token_id}")


# COMMAND ----------

# MAGIC %md
# MAGIC # Test the standalone pipeline

# COMMAND ----------

# Define parameters to generate text
def gen_text(prompts, use_template=False, **kwargs):
    """
    Generates text based on the given prompts using a pre-trained language model.
    
    Parameters:
        prompts (list): A list of strings containing the prompts for text generation.
        use_template (bool): If True, formats each prompt with a pre-defined template before text generation. Default is False.
        
    Returns:
        A list of strings containing the generated texts corresponding to the input prompts.
    """
    
    if use_template:
        full_prompts = [
            # PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
            PROMPT_TEMPLATE_V2.format(text_to_analyse=prompt)
            for prompt in prompts
        ]
    else:
        full_prompts = prompts

    if "batch_size" not in kwargs:
        kwargs["batch_size"] = 1
    
    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    # configure other text generation arguments, see common configurable args here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,  # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    outputs = pipe(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]

    return outputs
  

# COMMAND ----------

# DBTITLE 1,Test the pipeline
generate_kwargs = {
  # 'max_new_tokens': 2048,
  'temperature': 0.1,
  'top_p': 0.65,
  'top_k': 20,
  'repetition_penalty': 1.2,
  'no_repeat_ngram_size': 0,
  'use_cache': False,
  'do_sample': True,
  # 'eos_token_id': tokenizer.eos_token_id,
  # 'pad_token_id': tokenizer.eos_token_id,
}

# Response time: ~120s
results = gen_text(["What is the definition of personally identifiable information?"], use_template=True, **generate_kwargs)
print(results[0])

# print(pipe("What is ML?"))

# COMMAND ----------

generate_kwargs = {
  "batch_size": 1,
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
}

# Response time: ~30s
results = gen_text([TEST_DATA[0]], use_template=True, **generate_kwargs)
print(results[0])

# COMMAND ----------

# DBTITLE 1,Call pipeline directly
PROMPT_TEMPLATE_V2 = """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer assistant. Always answer as helpfully as possible, while being succinct and safe. The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. 

If you don't know the answer to a question, please don't share false information.
<</SYS>>

In the text to analyse below detect instances of personally identifiable information (PII).
Personally identifiable information (PII) is information that can be used to identify, contact, or locate a single person, either alone or in combination with other pieces of information. 
You will label each found instance using format [PII_TYPE]. 
PII includes but not limited to: email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver's licence numbers (LICENCE).
Use the context of the text to verify whether proper nouns (e.g. June) in the text is a person's name.
Return JSON only. Do not provide explanations or commentary.
JSON:
{{"pii_detected": [{{"pii_type": "<PII type detected>", "value": "<PII value detected>"}}]}}

Text to analyse:
>>>
{text_to_analyse}
>>>[/INST]
"""

generate_kwargs = {
  "batch_size": 1,
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
}

# Call pipeline directly without using gen_text()
# Response time: ~30s
results = pipe([PROMPT_TEMPLATE_V2.format(text_to_analyse=TEST_DATA[0])], **generate_kwargs)
print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC # Log Model with mlflow.pyfunc

# COMMAND ----------

# from config import PROMPT_TEMPLATE_V2

class Llama2(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts['repository'], padding_side="left")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'],
            use_safetensors=True, 
            # use_safetensors=False, 
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            # return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        self.model.eval()


    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        # TODO: load prompt in context
        prompt = f"{SYSTEM_PROMPT}{USER_INSTRUCTION}"
        return prompt.format(text_to_analyse=instruction)

        # return PROMPT_TEMPLATE_V2.format(text_to_analyse=instruction)
        # return f"""{PROMPT_TEMPLATE}
        # {instruction}
        # [/INST]
        # """


    def _generate_response(self, prompt, temperature, max_new_tokens):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')

        generate_kwargs = {
          # "batch_size": 1,
          # 'max_new_tokens': 2048,
          # 'temperature': 0.1,
          'top_p': 0.65,
          'top_k': 20,
          'repetition_penalty': 1.2,
          'no_repeat_ngram_size': 0,
          'use_cache': False,
          # 'do_sample': True,
          # 'eos_token_id': tokenizer.eos_token_id,
          # 'pad_token_id': tokenizer.eos_token_id,
        }

        # Generate parameters 
        # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
        output = self.model.generate(encoded_input, 
                                     do_sample=True, 
                                     temperature=temperature, 
                                     max_new_tokens=max_new_tokens,
                                     **generate_kwargs,
                                     )
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return generated_response


    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """

        outputs = []

        for i in range(len(model_input)):
          prompt = model_input["prompt"][i]
          temperature = model_input.get("temperature", [0.1])[i]
          max_new_tokens = model_input.get("max_new_tokens", [2048])[i]

          outputs.append(self._generate_response(prompt, temperature, max_new_tokens))
      
        return outputs


# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

import pandas as pd

# Define input and output schema
input_schema = Schema([
      ColSpec(DataType.string, "prompt"), 
      # ColSpec(DataType.double, "temperature"), 
      # ColSpec(DataType.long, "max_new_tokens"),
    ]
)
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
                "prompt":["what is ML?"], 
                "temperature": [0.5],
                "max_new_tokens": [100]
              }
            )

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 2 minutes to complete
with mlflow.start_run() as run:
    # As the prompt template is key to getting us our best results, we'll also log it as an artifact along with our model
    mlflow.log_param("prompt_template", PROMPT_TEMPLATE_V2)

    mlflow.pyfunc.log_model(
        "model",
        python_model=Llama2(),
        artifacts={'repository' : snapshot_location},
        # pip_requirements=["torch", "transformers", "accelerate", "sentencepiece"],
        pip_requirements={
          f"accelerate=={accelerate.__version__}",
          "bitsandbytes==0.41.1", # package doesn't expose __version__
          f"mlflow=={mlflow.__version__}",
          f"sentencepiece=={sentencepiece.__version__}",
          f"torch=={torch.__version__}"
          f"transformers=={transformers.__version__}",
        },
        input_example=input_example,
        signature=signature,
    )


# COMMAND ----------

result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    MLFLOW_MODEL_NAME_P,
    await_registration_for=1000,
)


# COMMAND ----------

# MAGIC %md
# MAGIC # Load the model
# MAGIC
# MAGIC Now that we've logged and registered the model, let's load it from the model registry. Notice that we used `mlflow.transformers` to log the model, but now are loading it with `mlflow.pyfunc` as a PyFunc flavour model.
# MAGIC
# MAGIC This is because XXXX
# MAGIC
# MAGIC **If you receive an out of memory error**: 
# MAGIC You may receive an out of memory error trying to run the below code. This is because in the previous steps we've already loaded the model into memory. The easiest way to clear the memory is to `Detach & re-attach` the cluster from this Notebook
# MAGIC
# MAGIC <img src="https://github.com/vinoaj/databricks-resources/blob/main/assets/img/notebook-cluster-detach-reattach.png?raw=true" style="height: 75px; margin-right: 10px;">

# COMMAND ----------

# DBTITLE 1,Reinstall packages in case session has been cleared (handled in init script)
# %pip install -Uq accelerate==0.23.0 bitsandbytes==0.41.1 mlflow transformers==4.33.2 xformers==0.0.21 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python 

# # Restart the Python kernel
# dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd

from pyspark.sql.functions import col
from pyspark.sql.types import StringType

from config import MLFLOW_MODEL_NAME_P, PROMPT_TEMPLATE_V2, TEST_DATA, SYSTEM_PROMPT, USER_INSTRUCTION

client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model: mlflow.pyfunc

# COMMAND ----------

from pprint import pprint

model_uri = f"models:/{MLFLOW_MODEL_NAME_P}/latest"

# pprint(dict(client.search_model_versions(f"name='{MLFLOW_MODEL_NAME_P}'")[0]))

for model_version in client.search_model_versions(f"name='{MLFLOW_MODEL_NAME_P}'"):
    pprint(dict(model_version))

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME_P}/latest")
print(f"Loaded model: {MLFLOW_MODEL_NAME_P}")

# COMMAND ----------

# results = loaded_model.predict([data[0]])
results = loaded_model.predict({
  "prompt": [TEST_DATA[0]],
  "temperature": [0.1],
  "max_new_tokens": [2048],
})
print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy to Model Serving
# MAGIC
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves our model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. For more information on GPU model serving, contact the Databricks team or sign up [here](https://docs.google.com/forms/d/1-GWIlfjlIaclqDz6BPODI2j1Xg4f4WbFvBXyebBpN-Y/edit)
# MAGIC
# MAGIC At this point, model serving management is automated only via Databricks REST APIs

# COMMAND ----------

latest_version = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME_P}'")[0].version
print(latest_version)

# COMMAND ----------

import requests
import json

# Provide a name to the serving endpoint
ENDPOINT_NAME = 'vinnyv-llama2-13b-chat-hf'

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'
update_url = f"{databricks_url}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/config"

# model_version = latest_version
endpoint_config = {
  "name": ENDPOINT_NAME,
  "config": {
    "served_models": [{
      "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
      "model_name": MLFLOW_MODEL_NAME_P,
      "model_version": latest_version,
      "workload_type": "GPU_MEDIUM",
      "workload_size": "Small",
      # Currently, GPU serving models cannot be scaled down to zero
      "scale_to_zero_enabled": "False" 
    }]
  }
}
endpoint_json = json.dumps(endpoint_config, indent='  ')

# Send a POST request to the API
response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)

if response.status_code != 200:
  print(response.text)
  if "RESOURCE_ALREADY_EXISTS" in response.text:
    response = requests.request(method="PUT", headers=deploy_headers, url=update_url, data=endpoint_json)
    if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  else:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

# Show the response of the POST request
# When first creating the serving endpoint, it should show that the state 'ready' is 'NOT_READY'
# You can check the status on the Databricks model serving endpoint page, it is expected to take ~35 min for the serving endpoint to become ready
print(response.json())

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
