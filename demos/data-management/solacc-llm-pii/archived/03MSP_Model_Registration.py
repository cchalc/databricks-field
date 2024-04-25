# Databricks notebook source
# MAGIC %md
# MAGIC # Model Registration
# MAGIC
# MAGIC Now that we have got a working model and have found prompts that works for us, we want to do three things:
# MAGIC - **Log** the model with MLflow: packages up the model artifacts for easy retrieval and deployment
# MAGIC
# MAGIC - **Register** the model in Model Registry: centralises management of the full lifecycle of the model. This is the point from which we can confidently retrieve the latest working version for deployment in UDFs or model serving 
# MAGIC
# MAGIC - Deploy the model to GPU **Model Serving**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Install libraries
# MAGIC
# MAGIC We'll first install some required libraries to work with Llama 2 and Hugging Face Transformers. As we'll be doing this quite regularly through our exploration, it would be more efficient to load these as part of an init script for your cluster
# MAGIC
# MAGIC You can find a [sample init script here]($./init_script.sh)

# COMMAND ----------

# DBTITLE 1,Install required libraries (covered in init script)
# Uncomment the below if you're not using a cluster init script
# See init_script.sh for a sample init script
# Documentation on using init scripts: https://docs.databricks.com/en/init-scripts/index.html

# %pip install -Uq accelerate==0.23.0 bitsandbytes==0.41.1 mlflow transformers==4.33.2 xformers==0.0.21 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python 

# # Restart the Python kernel
# dbutils.library.restartPython()

# COMMAND ----------

import accelerate
import pandas as pd
import llm_utils
import mlflow
import numpy as np
import os
import re
import sentencepiece
import torch
import transformers

from huggingface_hub import snapshot_download, login, notebook_login
from mlflow.exceptions import MlflowException
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from config import (
    CACHE_PATH,
    DOWNLOAD_PATH,
    HF_MODEL_NAME,
    HF_MODEL_REVISION_ID,
    MODEL_SERVING_ENDPOINT_NAME,
    MLFLOW_MODEL_NAME_MS,
    # MLFLOW_MODEL_NAME_P,
    SYSTEM_PROMPT,
    TEST_DATA,
    USER_INSTRUCTION_PROFILE_PII,
    USER_INSTRUCTION_CATEGORISE_DEPARTMENT,
)

# Disable progress bars for cleaner output
transformers.utils.logging.disable_progress_bar()
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "False"

# Instantiate MLflow client
client = mlflow.tracking.MlflowClient()


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

# DBTITLE 1,Login to Hugging Face to get access to the model
# notebook_login()

token = dbutils.secrets.get(scope="tokens", key="hf_token_vv")
login(token=token)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create LLM Pipeline
# MAGIC
# MAGIC We utilise our code from the previous Notebook to load our LLM pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download model artificats
# MAGIC
# MAGIC We'll first download a snapshot of the model artifacts. By default, `snapshot_download` downloads the artifacts to a cache on local disk. However, if the cluster shuts down or we wish to run this Notebook on another cluster, the cache is lost and we need to download everything all over again. So we explicitly set the `cache_dir` to point to a Workspace location, so that the cache is persisted across sessions.
# MAGIC
# MAGIC When running this command, if the files are already present in the cache, those files will not be downloaded.

# COMMAND ----------

# TODO: automate generation of path
CACHE_PATH = "/Users/vinny.vijeyakumaar@databricks.com/hf_cache"

snapshot_location = snapshot_download(
  repo_id=HF_MODEL_NAME, 
  revision=HF_MODEL_REVISION_ID, 
  ignore_patterns="*.bin",
  # ignore_patterns="*.safetensors",
  cache_dir=CACHE_PATH
)


displayHTML(f"""
  Working with model & revision ID: <b>{HF_MODEL_NAME} : {HF_MODEL_REVISION_ID}</b><br/>
  <a href='{llm_utils.generate_model_revision_url(HF_MODEL_NAME, HF_MODEL_REVISION_ID)}'>View source files on Hugging Face</a><br/>
  Cache folder: {snapshot_location}
""")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Define pipeline

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(snapshot_location, padding_side="left")

model_config = AutoConfig.from_pretrained(
  snapshot_location,
  trust_remote_code=True, # this can be needed if we reload from cache
)

model = AutoModelForCausalLM.from_pretrained(snapshot_location,
        use_safetensors=True,
        trust_remote_code=True,
        config=model_config,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
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


# COMMAND ----------

# MAGIC %md
# MAGIC # Test the standalone pipeline

# COMMAND ----------

generate_kwargs = {
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
  "batch_size": 1,
}

# Take our test data and inject it into the prompt templates
test_prompts = [
    SYSTEM_PROMPT.format(instruction=USER_INSTRUCTION_PROFILE_PII.format(text_to_analyse=data_item))
    for data_item in TEST_DATA
]

# Call pipeline directly without using gen_text()
results = pipe(test_prompts, **generate_kwargs)
print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC # Log Model with `mlflow.pyfunc`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define our PyFunc model
# MAGIC
# MAGIC We'll now package up our model as an [MLflow PyFunc (python_function)](https://mlflow.org/docs/latest/models.html#python-function-python-function) model. The model is expressed as a class that extends `mlflow.pyfunc.PythonModel`. We give our class a descriptive name `Llama2_13B_PIIBot`.
# MAGIC
# MAGIC PyFunc models give us the flexibility of injecting custom logic into our model operations. Let's take a look at some of these:
# MAGIC
# MAGIC - `_build_prompt`: ensures the prompts utilise our already defined system prompt template. This is handy for prompt sanitisation and pre-processing prompts to handle other logic
# MAGIC - `_generate_response`: we can also implement our best determined `kwargs` and abstract away this responsibility from the user. Furthermore, we can also do some post-processing on the outputs if we desire.
# MAGIC
# MAGIC You'll note here we build a generic prompt utilising `SYSTEM_PROMPT` but not incorporating `USER_INSTRUCTION_PROFILE_PII` or the other user instruction prompt templates. This is because this model will have multiple potential uses, and we don't want to lock it down to a very specific use case. If we did, we'd need to house multiple models for various use cases, which in turn would consume unnecessary storage and GPUs. Instead, we'll utilise `USER_INSTRUCTION_PROFILE_PII` (and the other templates) when passing prompts to the model.
# MAGIC

# COMMAND ----------

print(f"System prompt we will package into the model: {SYSTEM_PROMPT}")

# COMMAND ----------

class Llama2_13B_PIIBot(mlflow.pyfunc.PythonModel):
    def __init__(self):
      self.input_params = {}

    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        self.default_system_prompt = SYSTEM_PROMPT
        self.input_params = {}

        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts['repository'], padding_side="left")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'],
            use_safetensors=True, 
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        # Put model in eval mode
        self.model.eval()


    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        
        sys_prompt = self.default_system_prompt
        if self.input_params is not None and "system_prompt" in self.input_params:
          sys_prompt = self.input_params["system_prompt"]

        prompt = sys_prompt.format(instruction=instruction)
        # print("Prompt: ", prompt)
        return prompt


    def _generate_response(self, prompt, temperature=0.1, max_new_tokens=2048):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')

        # Incorporate kwargs from our testing
        generate_kwargs = {
          # "batch_size": 1,
          'max_new_tokens': 2048,
          'temperature': 0.01,
          'top_p': 0.65,
          'top_k': 20,
          'repetition_penalty': 1.2,
          'no_repeat_ngram_size': 0,
          'use_cache': False,
          'do_sample': True,
        }

        # TODO: find out why not working in model serving
        # generate_kwargs = self._selective_merge(self.input_params, generate_kwargs)

        # Generate parameters 
        # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
        output = self.model.generate(encoded_input, 
                                    #  do_sample=True, 
                                    #  temperature=temperature, 
                                    #  max_new_tokens=max_new_tokens,
                                     **generate_kwargs,
                                     )
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        # print("Generated response: ", generated_response)
        return generated_response


    def _selective_merge(self, dict_A: dict, dict_B: dict) -> dict:
      """
      Dict A overrides common key values in Dict B.

      # Usage:
      dict_A = {'a': 1, 'b': 2, 'c': 3}
      dict_B = {'b': 20, 'd': 4}

      merged_dict = _selective_merge(dict_A, dict_B)
      print(merged_dict)  # Output: {'b': 2, 'd': 4}
      """
      return {k: dict_A.get(k, v) for k, v in dict_B.items()}


    def predict(self, context, model_input, params: dict = {}):
        """
        This method generates prediction for the given input.
        """
        self.input_params = params

        outputs = []

        for i in range(len(model_input)):
          prompt = model_input["prompt"][i]
          outputs.append(self._generate_response(prompt))
      
        # return outputs
        # {"candidates": [...]} is the required response format for MLflow AI gateway
        return {"candidates": outputs}


# COMMAND ----------

# MAGIC %md
# MAGIC ### Standalone testing of PyFunc model
# MAGIC
# MAGIC Now we can quickly test the PyFunc model on its own before registering it with MLflow

# COMMAND ----------

from mlflow.pyfunc import PythonModelContext
import transformers 

# try:
#   del loaded_model
# except:
#   pass

snapshot_location = "/Users/vinny.vijeyakumaar@databricks.com/hf_cache/models--meta-llama--Llama-2-13b-chat-hf/snapshots/0ba94ac9b9e1d5a0037780667e8b219adde1908c"
context = PythonModelContext({
  "repository": snapshot_location
})

loaded_model = Llama2_13B_PIIBot()
loaded_model.load_context(context)


# COMMAND ----------

# Take our test data and inject it into the prompt templates
test_prompts = [
    USER_INSTRUCTION_PROFILE_PII.format(text_to_analyse=data_item)
    for data_item in TEST_DATA
][-1:]

# df = pd.DataFrame(test_prompts, columns=["prompt"])
df = pd.DataFrame(test_prompts, columns=["prompt"])

# Run inference
# results = loaded_model.predict(context, {"prompt": test_prompts})
results = loaded_model.predict(context, df)
print(results)

# COMMAND ----------

system_prompt_override = """
<s>[INST] <<SYS>>
You speak like a pirate all the time
<</SYS>>

{instruction}
[/INST]
"""

# Take our test data and inject it into the prompt templates
test_prompts = [
    USER_INSTRUCTION_PROFILE_PII.format(text_to_analyse=data_item)
    for data_item in TEST_DATA
][-2:]

# df = pd.DataFrame(test_prompts, columns=["prompt"])
df = pd.DataFrame(test_prompts, columns=["prompt"])

# Run inference
# results = loaded_model.predict(context, {"prompt": test_prompts})
results = loaded_model.predict(context, df, params={"system_prompt": system_prompt_override})
print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model
# MAGIC
# MAGIC Next we log the model with MLflow.
# MAGIC
# MAGIC Note we are also using `log_param()` to package up our prompt templates with the model. The consumers of this model can retrieve the ideal prompt templates from the model's metadata. This ensures consistent usage of these templates across wherever the model is utilised.

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

import pandas as pd

# Define input and output schema
input_schema = Schema([ColSpec(DataType.string, "prompt")])
output_schema = Schema([ColSpec(DataType.string, "candidates")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({"prompt": ["What is MLflow?"]})

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 2 minutes to complete
with mlflow.start_run() as run:
    # As the prompt templates are key to getting us our best results, we'll also log it as an artifact along with our model
    mlflow.log_param("system_prompt", SYSTEM_PROMPT)
    mlflow.log_param("user_instruction_profile_pii", USER_INSTRUCTION_PROFILE_PII)
    mlflow.log_param("user_instruction_categorise_department", USER_INSTRUCTION_CATEGORISE_DEPARTMENT)
    # TODO: log kwargs
    # mlflow.log_XXX("kwargs", XXXX)

    mlflow.pyfunc.log_model(
        "model",
        python_model=Llama2_13B_PIIBot(),
        metadata={"task": "llm/v1/completions"},
        artifacts={'repository' : snapshot_location},
        pip_requirements={
          f"accelerate=={accelerate.__version__}",
          "bitsandbytes==0.41.1", # package doesn't expose __version__
          f"mlflow=={mlflow.__version__}",
          f"sentencepiece=={sentencepiece.__version__}",
          "torch",
          f"transformers=={transformers.__version__}",
        },
        input_example=input_example,
        signature=signature,
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Register model in Model Registry
# MAGIC
# MAGIC Now we'll register our model in the [MLflow Model Registry](https://docs.databricks.com/en/mlflow/model-registry.html). We'll then retrieve the model from Model Registry when utilising it in our ETL workflows and for deploying model serving endpoints.
# MAGIC
# MAGIC From the documentation:
# MAGIC > MLflow Model Registry is a centralized model repository and a UI and set of APIs that enable you to manage the full lifecycle of MLflow Models. Model Registry provides:
# MAGIC > - Chronological model lineage (which MLflow experiment and run produced the model at a given time).
# MAGIC > - Model Serving.
# MAGIC > - Model versioning.
# MAGIC > - Stage transitions (for example, from staging to production or archived).
# MAGIC > - Webhooks so you can automatically trigger actions based on registry events.
# MAGIC > - Email notifications of model events.
# MAGIC
# MAGIC This step can take more than 15 minutes to run. We've set `await_registration_for` to `1000` seconds (16.67 minutes). 
# MAGIC
# MAGIC *If you see a timeout error*: don't worry, the registration will still be taking place in the background. Use the link in the provided notification message to access the UI to view the progress of the registration. Once registration has been completed, you can proceed with the subsequent steps.

# COMMAND ----------

# DBTITLE 1,Register logged model to Model Registry
MLFLOW_MODEL_NAME_MS = "vv-pyfunc-llama2-13b-aig"

displayHTML(f"Check the registration status in the <a href='#mlflow/models/{MLFLOW_MODEL_NAME_MS}'>Model Registry page</a>")

try:
    result = mlflow.register_model(
        "runs:/"+run.info.run_id+"/model",
        MLFLOW_MODEL_NAME_MS,
        await_registration_for=1000,
    )
except MlflowException as e:
    if "Exceeded max wait time for model name" in str(e):
        displayHTML(f"Await registration timeout. Your model is still being registered in the background. Check the registration status in the <a href='#mlflow/models/{MLFLOW_MODEL_NAME_MS}'>Model Registry page</a>")
    else:
        # This will re-raise the caught exception if it's not due to exceeding the await timeout
        raise  


# COMMAND ----------

# DBTITLE 1,Get convenience links to the model registry and latest run pages
model_info = client.get_registered_model(MLFLOW_MODEL_NAME_MS)
latest_version = model_info.latest_versions[-1]
latest_run_id = latest_version.run_id
experiment_id = re.search(r"mlflow-tracking/([a-f0-9]{32})", latest_version.source).group(1)

displayHTML(f'''
  Latest version: {latest_version.version}<br />
  Model version status: {latest_version.status} <br />
  <a href="#mlflow/models/{MLFLOW_MODEL_NAME_MS}">View the model in Model Registry</a><br/>
  <a href="#mlflow/experiments/{experiment_id}/runs/{latest_run_id}/artifactPath/model">View the latest run artifacts</a>
''')

# COMMAND ----------

# MAGIC %md
# MAGIC # Load the model
# MAGIC
# MAGIC Now that we've logged and registered the model, let's load it from the model registry. 
# MAGIC
# MAGIC **If you receive an out of memory error**: 
# MAGIC You may receive an out of memory error trying to run the below code. This is because in the previous steps we've already loaded the model into memory. The easiest way to clear the memory is to `Detach & re-attach` the cluster from this Notebook
# MAGIC <br/>
# MAGIC <br/>
# MAGIC <img src="https://github.com/vinoaj/databricks-resources/blob/main/assets/img/notebook-cluster-detach-reattach.png?raw=true" width="600">

# COMMAND ----------

# DBTITLE 1,Reinstall packages in case session has been cleared
# Uncomment the below if you're not using a cluster init script
# See init_script.sh for a sample init script
# Documentation on using init scripts: https://docs.databricks.com/en/init-scripts/index.html

# %pip install -Uq accelerate==0.23.0 bitsandbytes==0.41.1 mlflow transformers==4.33.2 xformers==0.0.21 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python 

# # Restart the Python kernel
# dbutils.library.restartPython()


# COMMAND ----------

import mlflow
import pandas as pd

from pyspark.sql.functions import col
from pyspark.sql.types import StringType

from config import MLFLOW_MODEL_NAME_P, MODEL_SERVING_ENDPOINT_NAME, TEST_DATA, USER_INSTRUCTION_PROFILE_PII

client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# DBTITLE 1,Retrieve prompts from logged model parameters
model_run_id = loaded_model.metadata.run_id
run_data = client.get_run(model_run_id)
user_instruction_profile_pii = run_data.data.params["user_instruction_profile_pii"]
user_instruction_categorise_department = run_data.data.params["user_instruction_categorise_department"]

displayHTML(f"""<h1>Retrieved prompts</h1>
  <h2>Prompt: Categorise department</h2>
  <pre>{user_instruction_categorise_department}</pre>
  <hr />
  <h2>Prompt: Profile PII</h2>
  <pre>{user_instruction_profile_pii}</pre>""")

# COMMAND ----------

# DBTITLE 1,Load the latest logged model into the GPU
loaded_model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME_P}/latest")
print(f"Loaded model: {MLFLOW_MODEL_NAME_P}")

# COMMAND ----------

# Take our test data and inject it into the prompt templates
test_prompts = [
    user_instruction_profile_pii.format(text_to_analyse=data_item)
    for data_item in TEST_DATA
]

# df = pd.DataFrame(test_prompts, columns=["prompt"])
df = pd.DataFrame(test_prompts)


# Run inference
# results = loaded_model.predict(context, {"prompt": test_prompts})
results = loaded_model.predict(df)
print(results)

# COMMAND ----------

# DBTITLE 1,Retrieve prompts from logged model parameters
model_run_id = loaded_model.metadata.run_id
run_data = client.get_run(model_run_id)
user_instruction_profile_pii = run_data.data.params["user_instruction_profile_pii"]
user_instruction_categorise_department = run_data.data.params["user_instruction_categorise_department"]

displayHTML(f"""<h1>Retrieved prompts</h1>
  <h2>Prompt: Categorise department</h2>
  <pre>{user_instruction_categorise_department}</pre>
  <hr />
  <h2>Prompt: Profile PII</h2>
  <pre>{user_instruction_profile_pii}</pre>""")

# COMMAND ----------

# Take our test data and inject it into the prompt templates
test_prompts = [
    USER_INSTRUCTION_PROFILE_PII.format(text_to_analyse=data_item)
    for data_item in TEST_DATA
]

# Run inference
results = loaded_model.predict({"prompt": [test_prompts]})
print(results)

# COMMAND ----------

# Take our test data and inject it into the prompt templates
test_prompts = [
    user_instruction_profile_pii.format(text_to_analyse=data_item)
    for data_item in TEST_DATA
]

# df = pd.DataFrame(test_prompts, columns=["prompt"])
df = pd.DataFrame(test_prompts)


# Run inference
# results = loaded_model.predict(context, {"prompt": test_prompts})
results = loaded_model.predict(df)
print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC # Move model to Production stage
# MAGIC
# MAGIC Now that we're happy with the model and it's outputs, let's mark our latest version as production-ready. At the same time, we'll move the previous Production versions to stage `Archived` (TODO).
# MAGIC

# COMMAND ----------

# Take our test data and inject it into the prompt templates
test_prompts = [
    USER_INSTRUCTION_PROFILE_PII.format(text_to_analyse=data_item)
    for data_item in TEST_DATA
]

# Run inference
results = loaded_model.predict({"prompt": [test_prompts]})
print(results)

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = MLFLOW_MODEL_NAME_MS
model_versions = client.search_model_versions(filter_string=f"name='{model_name}'", order_by=["version DESC"])
latest_version = model_versions[0]

client.transition_model_version_stage(
  name=model_name,
  version=latest_version.version,
  stage="Production",
)


# COMMAND ----------

# MAGIC %md
# MAGIC # Move model to Production stage
# MAGIC
# MAGIC Now that we're happy with the model and it's outputs, let's mark our latest version as production-ready. At the same time, we'll move the previous Production versions to stage `Archived` (TODO).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy to Model Serving
# MAGIC
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves our model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. GPU model serving is currently in preview and available in a limited number of regions. For more information on GPU model serving, contact the Databricks team or sign up [here].(https://docs.google.com/forms/d/1-GWIlfjlIaclqDz6BPODI2j1Xg4f4WbFvBXyebBpN-Y/edit)
# MAGIC
# MAGIC At this point in time, model serving management is automated only via Databricks REST APIs

# COMMAND ----------

model_name = MLFLOW_MODEL_NAME_MS
model_versions = client.search_model_versions(filter_string=f"name='{model_name}'", order_by=["version DESC"])
latest_version = model_versions[0]

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = MLFLOW_MODEL_NAME_MS
model_versions = client.search_model_versions(filter_string=f"name='{model_name}'", order_by=["version DESC"])
latest_version = model_versions[0]

client.transition_model_version_stage(
  name=model_name,
  version=latest_version.version,
  stage="Production",
)


# COMMAND ----------

# Move previous Production versions to archived
# model_name = MLFLOW_MODEL_NAME_MS

# for mv in client.search_model_versions(f"name='{model_name}'"):
for mv in model_versions:
  if mv.current_stage == "Production" and mv.version < latest_version.version:
    print(f"Moving model {model_name} v.{mv.version} to stage Archived")
    client.transition_model_version_stage(
      name=model_name,
      version=mv.version,
      stage="Archived"
    )


# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy to Model Serving
# MAGIC
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves our model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. GPU model serving is currently in preview and available in a limited number of regions. For more information on GPU model serving, contact the Databricks team or sign up [here].(https://docs.google.com/forms/d/1-GWIlfjlIaclqDz6BPODI2j1Xg4f4WbFvBXyebBpN-Y/edit)
# MAGIC
# MAGIC At this point in time, model serving management is automated only via Databricks REST APIs

# COMMAND ----------

# DBTITLE 1,Helper functions
import requests
import json

class ModelServingHelper():
  def __init__(self):
    notebook_context    = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    self.databricks_url = notebook_context.apiUrl().getOrElse(None)
    self.token          = notebook_context.apiToken().getOrElse(None)
    self.headers        = {'Authorization': f'Bearer {self.token}', 'Content-Type': 'application/json'}
    self.api_url_base   = f"{self.databricks_url}/api/2.0/serving-endpoints"


  def check_endpoint_exists(self, endpoint_name: str) -> bool:
    endpoint_url = f"{self.api_url_base}/{endpoint_name}"
    response = requests.request(method="GET", headers=self.headers, url=endpoint_url)
    response_json = response.json()
    exists = ("name" in response_json and response_json["name"] == endpoint_name)
    print(f"Check endpoint exists: {endpoint_name} --> {exists}")
    return exists


  def create_endpoint(self, endpoint_config: dict):
    endpoint_config_json = json.dumps(endpoint_config)
    response = requests.request(method="POST", headers=self.headers, url=self.api_url_base, data=endpoint_config_json)
    print(f"Creating endpoint: {endpoint_config}")
    
    if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    else:
      return response.json()
    
  
  def update_endpoint(self, endpoint_name: str, endpoint_config: dict):
    endpoint_config_json = json.dumps(endpoint_config)
    config_url = f"{self.api_url_base}/{endpoint_name}/config"
    response = requests.request(method="PUT", headers=self.headers, url=config_url, data=endpoint_config_json)
    print(f"Updating endpoint: {endpoint_name} with config {endpoint_config}")
    
    if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    else:
      return response.json()


# helper = ModelServingHelper()
# helper.check_endpoint_exists("vinnyv-llama2-13b-chat-hf")

# COMMAND ----------

# For a 13B model, the endpoint creation time can take up to an hour

import requests
import json
from mlflow.tracking import MlflowClient

ENDPOINT_NAME = "vinnyv-llama2-13b-chat-hf-aig"

model_name = MLFLOW_MODEL_NAME_MS

client = MlflowClient()
ms_helper = ModelServingHelper()

model_info = client.get_registered_model(model_name)
latest_version = model_info.latest_versions[0]
model_version = latest_version

# https://docs.databricks.com/api/workspace/servingendpoints/create
# The workload size corresponds to a range of provisioned concurrency that the compute will autoscale between. 
# A single unit of provisioned concurrency can process one request at a time. 
# Valid workload sizes are "Small" (4 - 4 provisioned concurrency), "Medium" (8 - 16 provisioned concurrency), 
# and "Large" (16 - 64 provisioned concurrency)
endpoint_config = {
  "name": ENDPOINT_NAME,
  "config": {
    "served_models": [{
      "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
      "model_name": model_version.name,
      "model_version": model_version.version,
      # https://docs.databricks.com/en/machine-learning/model-serving/llm-optimized-model-serving.html#requirements
      "workload_type": "MULTIGPU_MEDIUM", #"GPU_MEDIUM",
      "workload_size": "Small",
      # Currently, GPU serving models cannot be scaled down to zero
      "scale_to_zero_enabled": "False" 
    }]
  }
}

endpoint_json = json.dumps(endpoint_config, indent=2)

if ms_helper.check_endpoint_exists(ENDPOINT_NAME) is False:
  print("Endpoint doesn't exist: creating new endpoint")
  ms_helper.create_endpoint(endpoint_config)
else:
  ms_helper.update_endpoint(ENDPOINT_NAME, endpoint_config)
  print("Endpoint exists: updating endpoint")


displayHTML(f"<a href='#mlflow/endpoints/{ENDPOINT_NAME}/events'>Link to Model Serving endpoint page</a>")

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

notebook_context  = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token             = notebook_context.apiToken().getOrElse(None)

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints/vinnyv-llama2-13b-chat-hf/invocations'
  headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  return response.json()

data = pd.DataFrame(["""You reply in valid JSON only. 
Personally identifiable information (PII) refers to any data that can be used to identify, contact, or locate a single person, either directly or indirectly.
PII includes but not limited to: valid email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver's licence numbers (LICENCE), etc.

Your task:
In the text to analyse below thoroughly detect all instances of PII. Do not ignore any potential PII.
Account for any spelling and grammatical errors.
Pay careful attention for person names. Use context clues, such as capitalisation, proper nouns (e.g. June), words following greetings (e.g. Hello), and surrounding text, to help identify person names. 
Brand and product names are not PII.
There can be multiple of the same PII type in the text. Return every instance detected.
Label each instance using format [PII_TYPE]. Do not return examples. Do not return empty or null values. 
Accuracy: Double-check your work to ensure that all PII is accurately detected and labeled.
Return valid JSON syntax only. Do not provide explanations. JSON:
{"pii_detected": [{"pii_type": "", "value": ""}]}
(e.g. {"pii_detected": [{"pii_type": "NAME", "value": "Harry},{"pii_type": "NAME", "value": "Susan"}]})


Text to analyse:
Hello Helena, your phone should be set up with the number 1-966-757-3629 x2936. If there are any issues with connectivity, please report them to IT by providing them with the phone's IMEI number, which is 27-544713-582252-7."""])

response = score_model(data)
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create MLflow AI Gateway Route
# MAGIC
# MAGIC The [MLflow AI Gateway service](https://mlflow.org/docs/latest/gateway/index.html) is a powerful tool designed to streamline the usage and management of various large language model (LLM) providers within an organization. It offers a high-level interface that simplifies the interaction with these services by providing a unified endpoint to handle specific LLM related requests.
# MAGIC
# MAGIC [Routes](https://mlflow.org/docs/latest/gateway/index.html#routes) specify which model a request should be routed to. Below we create a route pointing to the model we just deployed to model serving

# COMMAND ----------

# DBTITLE 1,Create AI Gateway route to model serving endpoint
import mlflow
from mlflow.gateway import set_gateway_uri, create_route, search_routes, query
from requests.exceptions import HTTPError
# from dbruntime.databricks_repl_context import get_context

set_gateway_uri("databricks")

notebook_context  = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
# databricks_url    = notebook_context.apiUrl().getOrElse(None)
workspace_url     = notebook_context.browserHostName().getOrElse(None)
print(f"Workspace URL: https://{databricks_url}")

# ctx = get_context()
# print(ctx.browserHostName)

pat = "xxxx"
# The route name should be the same as the model serving endpoint name
route_name = "vinnyv-llama2-13b-chat-hf-aig"

gateway_client = mlflow.gateway.MlflowGatewayClient("databricks")

try:
  route = mlflow.gateway.get_route(route_name)
  print("Deleting existing route")
  mlflow.gateway.delete_route(route_name)
except HTTPError:
  pass

route = gateway_client.create_route(
   name=route_name,
   route_type="llm/v1/completions",
   model={
       "name": ENDPOINT_NAME,
       "provider": "databricks-model-serving",
       "databricks_model_serving_config": {
          "databricks_workspace_url": f"https://{workspace_url}",
          "databricks_api_token": pat,
       },
   }
)

# https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints/vinnyv-llama2-13b-chat-hf-aig/invocations
print(route)

# COMMAND ----------

# DBTITLE 1,Helper functions
import mlflow
from typing import List

class AIGatewayHelper():
  def __init__(self, route:str):
    # mlflow.gateway.set_gateway_uri("databricks")
    self.gateway_client = mlflow.gateway.MlflowGatewayClient("databricks")
    self.route = route
  
  def query(self, model_inputs) -> List[str]:
    response = self.gateway_client.query(
      route=self.route,
      data=model_inputs
    )

    return response["candidates"]


# COMMAND ----------

route_name = "vinnyv-llama2-13b-chat-hf-aig"
# route_name = "mingyu-system-completions.5"
model_route = AIGatewayHelper(route_name)

print(model_route.query({"prompt": "What is MLflow?"}))

# COMMAND ----------

route_name = "vinnyv-llama2-13b-chat-hf-aig"
# route_name = "vinnyv-mosaicml-inference"
model_route = AIGatewayHelper(route_name)

print(model_route.query({"prompt": """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. The primary purpose of your job is to meticulously identify privacy risks in any given documentation. You are thorough. If you don't know the answer to a question, please don't share false information.
<</SYS>>

You speak in JSON and will return a single unnested array of strings. 
Here is a list of personally identifiable information (PII) types:
- FIRST_NAME
- SURNAME
- VEHICLE_ID
- ADDRESS
- IP_ADDRESS
- EMAIL
- WORK_EMAIL
- PHONE_NUMBER


Be meticulous in your analysis. Which items from only the above list are related to the following types of PII? 
- Person names
- Email addresses
- IP addresses

Return the identified items in a JSON object. Do not provide explanations or any other commentary.
[/INST]
"""}))

# COMMAND ----------

# route_name = "vinnyv-llama2-13b-chat-hf-aig"
route_name = "vinnyv-mosaicml-inference"
model_route = AIGatewayHelper(route_name)

print(model_route.query({"prompt": """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. The primary purpose of your job is to meticulously identify privacy risks in any given documentation. You are thorough. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Here is a list of personally identifiable information (PII) types:
- PHONE
- TIME
- URL
- DATE
- NAME

Be meticulous in your analysis. Only pick items from the above list and do not provide examples or any other non-existent items. 
Which items from the above list only are related to the following types of PII? If there are no related items, move on to the next type.
- Person names
- Email addresses
- Postal addresses

Return the identified items in only a JSON object. Do not provide commentary and explanations.
[/INST]
"""}))

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
