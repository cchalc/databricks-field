# Databricks notebook source
# MAGIC %md
# MAGIC # Model Registration & Deployment
# MAGIC
# MAGIC Now that we have got a working model and have found prompts that works for us, we want to do four things:
# MAGIC
# MAGIC <img src="https://github.com/vinoaj/databricks-resources/blob/main/assets/img/solacc-model-serving-flow.png?raw=true" width="800">
# MAGIC
# MAGIC - **Log** the model with MLflow: packages up the model artifacts for easy retrieval and deployment
# MAGIC
# MAGIC - **Register** the model in Model Registry: centralises management of the full lifecycle of the model. This is the point from which we can confidently retrieve the latest working version for deployment in UDFs or model serving. We'll save our model in Unity Catalog so we can effectively control access and usage
# MAGIC
# MAGIC - Deploy the model to GPU **Model Serving** [optimized for LLMs](https://docs.databricks.com/en/machine-learning/model-serving/llm-optimized-model-serving.html)
# MAGIC
# MAGIC - Create [**MLflow AI Gateway routes**](https://mlflow.org/docs/latest/gateway/index.html) to interface with
# MAGIC   - The served model
# MAGIC   - [MosaicML Inference](https://www.mosaicml.com/inference) service's Llama 2 70B model
# MAGIC
# MAGIC ___
# MAGIC
# MAGIC ## Cluster configuration
# MAGIC
# MAGIC We can utilise a single-node non-GPU cluster as this Notebook doesn't execute any inference or data transformations.
# MAGIC
# MAGIC - Single node
# MAGIC - Access mode: `Assigned` (can use Shared if not using cluster init script)
# MAGIC - DBR: `13.3+ ML`
# MAGIC - Node type: 32 GB memory, 8 cores
# MAGIC   - AWS: `m4.2xlarge`
# MAGIC   - Azure: XXXX
# MAGIC   - GCP: XXXX
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
import json
import llm_utils
import mlflow
import numpy as np
import pandas as pd
import re
import os
import requests
import sentencepiece
import torch
import transformers

from huggingface_hub import login, notebook_login

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from requests.exceptions import HTTPError
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    AI_GATEWAY_ROUTE_NAME_MODEL_SERVING,
    AI_GATEWAY_ROUTE_NAME_MOSAIC_70B,
    CATALOG_NAME,
    SCHEMA_NAME,
    HF_MODEL_NAME,
    HF_MODEL_REVISION_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MODEL_SERVING_ENDPOINT_NAME,
    SECRET_SCOPE,
    SECRET_KEY_HF,
    SECRET_KEY_MOSAICML,
    SECRET_KEY_PAT,
    SYSTEM_PROMPT,
    TEST_DATA,
    USE_UC,
    USER_INSTRUCTION_PROFILE_PII,
    USER_INSTRUCTION_CATEGORISE_DEPARTMENT,
    USER_INSTRUCTION_FILTER_PII_TYPES,
)

# Disable progress bars for cleaner output
transformers.utils.logging.disable_progress_bar()
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "False"

# Instantiate MLflow client
if USE_UC:
    # Set registry to Unity Catalog
    mlflow.set_registry_uri("databricks-uc")

client = mlflow.tracking.MlflowClient()

print(f"Working with LLM revision ID: {HF_MODEL_NAME} : {HF_MODEL_REVISION_ID}")


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
token = dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY_HF)
login(token=token)


# COMMAND ----------

# MAGIC %md
# MAGIC # Define LLM
# MAGIC
# MAGIC We'll now define our model and tokenizer. You'll notice here we're not configuring performance parameters for quantization, low CPU memory usage, and so on. That is because LLM optimised model serving will automatically configure those parameters for optimised performance.
# MAGIC

# COMMAND ----------

model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_NAME, revision=HF_MODEL_REVISION_ID, torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_NAME, revision=HF_MODEL_REVISION_ID, padding_side="left"
)

print(f"Loaded model and revision: {HF_MODEL_NAME}: {HF_MODEL_REVISION_ID}")


# COMMAND ----------

# MAGIC %md
# MAGIC # Log & Register Model
# MAGIC
# MAGIC We'll now log and register the model with MLflow using the `transformers` flavour. We want to log the model for several reasons:
# MAGIC
# MAGIC - It's the precursor to having the model registered in the Model Registry
# MAGIC - It allows us to log various iterations of the model and compare performance across iterations
# MAGIC
# MAGIC ## Optimising for LLMs
# MAGIC
# MAGIC In order to signal to model serving that we wish for the model to be optimised for LLMs, add `metadata={"task": "llm/v1/completions"}` to your `log_model()` call. 
# MAGIC
# MAGIC You'll notice this aligns with MLflow AI Gateway's definition of models served for [text/instruction completions](https://mlflow.org/docs/latest/gateway/index.html#supported-provider-models). This is useful, as it pre-defines the expected format for inputs and outputs. 
# MAGIC
# MAGIC ## Logging prompt templates
# MAGIC
# MAGIC We use `log_param()` to package up our prompt templates with the model. The consumers of this model can retrieve the ideal prompt templates from the model's metadata. This ensures consistent usage of these templates across wherever the model is utilised. However, the users also have the flexibility to override the templates with their own prompt structures.
# MAGIC
# MAGIC ## Combining logging and registration
# MAGIC
# MAGIC We register the model at the same time as logging it by setting the `registered_model_name` argument in the `log_model()` call. We can separate out the model logging and registration steps. However, for convenience, we combine them in a single step. You may wish to separate out the steps if you are not yet ready to register your model.
# MAGIC
# MAGIC ## Wait time
# MAGIC
# MAGIC This can take up to 10 minutes as MLflow retrieves all the relevant assets from Hugging Face, compiles the model artifacts, and registers it.
# MAGIC

# COMMAND ----------

# All possible inputs for llm/v1/completions are here: https://mlflow.org/docs/latest/gateway/index.html#completions
input_schema = Schema(
    [
        ColSpec("string",  "prompt"),
        ColSpec("double",  "temperature", optional=True),
        ColSpec("integer", "max_tokens", optional=True),
        ColSpec("string",  "stop", optional=True),
        ColSpec("integer",  "candidate_count", optional=True),
    ]
)

output_schema = Schema([ColSpec("string", "predictions")])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

input_example = {
    "prompt": [
        """<s>[INST] <<SYS>>\nYou are an expert, helpful, respectful and honest privacy officer. Always answer as helpfully as possible, while being safe. The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\nWhat is the definition of personally identifiable information (PII)?[/INST]""",

        """<s>[INST] <<SYS>>\nYou are an expert, helpful, respectful and honest privacy officer. Always answer as helpfully as possible, while being safe. The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\nYou reply in valid JSON only. \nPersonally identifiable information (PII) refers to any data that can be used to identify, contact, or locate a single person, either directly or indirectly.\nPII includes but not limited to: valid email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver\'s licence numbers (LICENCE), etc.\n\nYour task:\nIn the text to analyse below thoroughly detect all instances of PII. Do not ignore any potential PII.\nAccount for any spelling and grammatical errors.\nPay careful attention for person names. Use context clues, such as capitalisation, proper nouns (e.g. June), words following greetings (e.g. Hello), and surrounding text, to help identify person names. \nBrand and product names are not PII.\nThere can be multiple of the same PII type in the text. Return every instance detected.\nLabel each instance using format [PII_TYPE]. Do not return examples. Do not return empty or null values. \nAccuracy: Double-check your work to ensure that all PII is accurately detected and labeled.\nReturn valid JSON syntax only. Do not provide explanations. JSON:\n{"pii_detected": [{"pii_type": "", "value": ""}]}\n(e.g. {"pii_detected": [{"pii_type": "NAME", "value": "Harry},{"pii_type": "NAME", "value": "Susan"}]})\n\n\nText to analyse:\nAttn: August, we need you to conduct a comprehensive study analyzing the impact of industrial culture on employee motivation. Please use your account Savings Account for any necessary purchases and keep track of all transactions for auditing purposes. Kind regards, Vinny - vinny.vijeyakumaar@databricks.com\n\n[/INST]""",
    ],
    "temperature": [0.01, 0.01],
    "max_tokens": [2048, 2048],
    "candidate_count": [1, 1],
}

# e.g. main.solacc_llm_pii.solacc-llm-pii-llama2-13b
registered_model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MLFLOW_MODEL_NAME}"  

# Start MLflow run
with mlflow.start_run() as run:
    # As the prompt templates are key to getting us our best results, we'll also log it as an artifact alongside our model
    mlflow.log_param("system_prompt", SYSTEM_PROMPT)
    mlflow.log_param("user_instruction_categorise_department", USER_INSTRUCTION_CATEGORISE_DEPARTMENT)
    mlflow.log_param("user_instruction_filter_pii_types", USER_INSTRUCTION_FILTER_PII_TYPES)
    mlflow.log_param("user_instruction_profile_pii", USER_INSTRUCTION_PROFILE_PII)

    components = {"model": model, "tokenizer": tokenizer}

    print(f"Registering & logging model: {registered_model_name}")

    # Log and register model
    model_info = mlflow.transformers.log_model(
        transformers_model=components,
        task="text-generation",
        artifact_path="model",
        registered_model_name=registered_model_name,
        metadata={"task": "llm/v1/completions"},
        signature=signature,
        input_example=input_example,
        await_registration_for=1000,
    )

print(f"Registered & logged model: {registered_model_name}")
print(model_info)


# COMMAND ----------

# MAGIC %md
# MAGIC # Promote the Model to Production or Champion status
# MAGIC
# MAGIC Now we're ready to mark our model as production ready. The option varies depending on whether you're using Unity Catalog for your model registry.
# MAGIC
# MAGIC ### With Unity Catalog
# MAGIC
# MAGIC Unity Catalog Model Registry doesn't have the concept of stages (`None`, `Archived`, `Staging`, `Production`) that you may be used to with MLflow. Instead, we define an alias to point to our desired model version. For this exercise, we use the alias `Champion`.
# MAGIC
# MAGIC ### Without Unity Catalog
# MAGIC
# MAGIC - Update latest version to stage `Production`
# MAGIC - Move all previous production versions to stage `Archived`
# MAGIC

# COMMAND ----------

def set_model_version_champion(
    registered_model_name: str, latest_model_version: int = None
):
    if latest_model_version is None:
        model_version_infos = client.search_model_versions(
            f"name='{registered_model_name}'"
        )
        latest_model_version = max(
            [
                model_version_info.version
                for model_version_info in model_version_infos
            ]
        )

    # Set model alias
    client.set_registered_model_alias(
        registered_model_name, "Champion", latest_model_version
    )


def set_model_version_production(
    registered_model_name: str, latest_model_version: int = None
):
    model_versions = client.search_model_versions(
        filter_string=f"name='{model_name}', order_by=['version DESC']"
    )

    if latest_model_version is None:
        latest_model_version = model_versions[0].version

    # Move desired version to Production stage
    client.transition_model_version_stage(
        name=model_name,
        version=latest_model_version,
        stage="Production",
    )

    # Move all other Production versions to Archived stage
    for mv in model_versions:
        if (
            mv.current_stage == "Production"
            and mv.version != latest_model_version
        ):
            print(f"Moving model {model_name} v.{mv.version} to stage Archived")
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Archived"
            )


if USE_UC:
    set_model_version_champion(registered_model_name)
else:
    set_model_version_production(registered_model_name)


# COMMAND ----------

# DBTITLE 1,Get convenience links to the model registry and latest run pages
prefix = f"{CATALOG_NAME}.{SCHEMA_NAME}." if USE_UC else ""
registered_model_name = f"{prefix}{MLFLOW_MODEL_NAME}"
model_ui_url = f"/explore/data/models/{CATALOG_NAME}/{SCHEMA_NAME}/{MLFLOW_MODEL_NAME}" if USE_UC else f"#mlflow/models/{MLFLOW_MODEL_NAME}"

model_info = (client.get_model_version(registered_model_name, max(
    [info.version for info in client.search_model_versions(f"name='{registered_model_name}'")]))
    if USE_UC else client.get_registered_model(registered_model_name).latest_versions[-1])

latest_model_version, model_status, latest_run_id = (
    model_info.version, model_info.aliases, model_info.run_id) if USE_UC else (
    model_info.version, model_info.status, model_info.run_id)

experiment_id = re.search(r"mlflow-tracking/([a-f0-9]{32})", model_info.source).group(1)

displayHTML(f'''
  Latest version: {latest_model_version}<br />
  Model version status: {model_status} <br />
  <a href="{model_ui_url}">View the model in Model Registry</a><br/>
  <a href="#mlflow/experiments/{experiment_id}/runs/{latest_run_id}/artifactPath/model">View the latest run artifacts</a>
''')


# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy to Model Serving
# MAGIC
# MAGIC Once the model is registered, we can use the `serving-endpoints` API to create a Databricks GPU Model Serving Endpoint that serves our model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. GPU model serving is currently in public preview and available in a limited set of regions. Find out more [here](https://docs.databricks.com/en/machine-learning/model-serving/llm-optimized-model-serving.html).
# MAGIC
# MAGIC At this point in time, model serving management is automated only via Databricks REST APIs
# MAGIC

# COMMAND ----------

# DBTITLE 1,Helper functions
import requests
import json

class ModelServingHelper:
    def __init__(self):
        context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        self.databricks_url = context.apiUrl().getOrElse(None)
        self.token = context.apiToken().getOrElse(None)
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self.api_url_base = f"{self.databricks_url}/api/2.0/serving-endpoints"

    def send_request(self, method, endpoint, data=None):
        url = f"{self.api_url_base}/{endpoint}"
        if data is not None:
            response = requests.request(method, url, headers=self.headers, data=json.dumps(data))
        else:
            response = requests.request(method, url, headers=self.headers)
        if response.status_code == 404:
            return response.json()
        elif response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        return response.json()

    def check_endpoint_exists(self, endpoint_name: str) -> bool:
        response_json = self.send_request("GET", endpoint_name)
        exists = "name" in response_json and response_json["name"] == endpoint_name
        print(f"Endpoint {endpoint_name} exists: {exists}")
        return exists

    def create_endpoint(self, endpoint_config: dict):
        print(f"Creating endpoint with config: {endpoint_config}")
        return self.send_request("POST", "", endpoint_config)

    def delete_endpoint(self, endpoint_name: str):
        print(f"Deleting endpoint: {endpoint_name}")
        return self.send_request("DELETE", endpoint_name)

    def update_endpoint(self, endpoint_name: str, endpoint_config: dict):
        config = {"served_models": [endpoint_config["config"]["served_models"][0]]}
        print(f"Updating endpoint: {endpoint_name} with config {config}")
        return self.send_request("PUT", f"{endpoint_name}/config", config)


# COMMAND ----------

# For a 13B model, the first endpoint creation time can take up to an hour. 
# Subsequent udpates will be faster.
ms_helper = ModelServingHelper()

model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MLFLOW_MODEL_NAME}"
model_version = llm_utils.get_model_latest_version(model_name)

# https://docs.databricks.com/api/workspace/servingendpoints/create
# The workload size corresponds to a range of provisioned concurrency that the compute will autoscale between.
# A single unit of provisioned concurrency can process one request at a time.
# Valid workload sizes are "Small" (4 - 4 provisioned concurrency), "Medium" (8 - 16 provisioned concurrency),
# and "Large" (16 - 64 provisioned concurrency)
endpoint_config = {
    "name": MODEL_SERVING_ENDPOINT_NAME,
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": model_version,
                # https://docs.databricks.com/en/machine-learning/model-serving/llm-optimized-model-serving.html#requirements
                # 4 x A10G
                "workload_type": "MULTIGPU_MEDIUM",  
                "workload_size": "Small",
                # Currently, GPU serving models cannot be scaled down to zero
                "scale_to_zero_enabled": "False",
            }
        ]
    },
    "tags": [
        {"key": "owner", "value": "team_data_and_governance"},
        {"key": "purpose", "value": "pii_bot"},
    ]
}

if ms_helper.check_endpoint_exists(MODEL_SERVING_ENDPOINT_NAME) is False:
    print("Endpoint doesn't exist: creating new endpoint")
    ms_helper.create_endpoint(endpoint_config)
else:
    print("Endpoint exists: updating endpoint")
    ms_helper.update_endpoint(MODEL_SERVING_ENDPOINT_NAME, endpoint_config)


displayHTML(
    f"<a href='#mlflow/endpoints/{MODEL_SERVING_ENDPOINT_NAME}/events'>Link to Model Serving endpoint page</a>"
)


# COMMAND ----------

# DBTITLE 1,If you need to delete the endpoint to start from scratch
# Uncomment below lines if you wish to delete your endpoint
# ms_helper = ModelServingHelper()
# ms_helper.delete_endpoint(MODEL_SERVING_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create MLflow AI Gateway Route
# MAGIC
# MAGIC The [MLflow AI Gateway service](https://mlflow.org/docs/latest/gateway/index.html) is a powerful tool designed to streamline the usage and management of various large language model (LLM) providers within an organization. It offers a high-level interface that simplifies the interaction with these services by providing a unified endpoint to handle specific LLM related requests.
# MAGIC
# MAGIC <img src="https://github.com/vinoaj/databricks-resources/blob/main/assets/img/mlflow-ai-gateway.png?raw=true" width="600">
# MAGIC
# MAGIC [Routes](https://mlflow.org/docs/latest/gateway/index.html#routes) specify which model a request should be routed to. Below we'll create 2 routes:
# MAGIC 1. A route that points to the model we just deployed to model serving
# MAGIC 2. A route to [MosaicML's inference service](https://www.mosaicml.com/inference), specifically targeting the Llama2 70B model. We can use this route for generic prompts and without needing to incur the time and expense of hosting the 70B model in our own environment.

# COMMAND ----------

# Let the Gateway know we're operating inside of Databricks
gateway_client = mlflow.gateway.MlflowGatewayClient("databricks")

# COMMAND ----------

# DBTITLE 1,MLflow AI Gateway helper functions
import mlflow
from typing import List


class AIGatewayHelper:
    def __init__(self, route_name: str):
        self.gateway_client = mlflow.gateway.MlflowGatewayClient("databricks")
        self.route_name = route_name

    def create_route(self, route_config: dict):
        try:
            # At this time, the API doesn't support updating routes. 
            # So we delete and recreate the route instead.
            route = self.gateway_client.get_route(self.route_name)
            print(f"Deleting existing route: {self.route_name}")
            self.gateway_client.delete_route(self.route_name)
        except HTTPError:
            pass

        route = self.gateway_client.create_route(
            name=self.route_name,
            route_type=route_config["route_type"],
            model=route_config["model"],
        )

        print(f"Route created: {self.route_name}")
        print(route)
        return route

    def query(self, model_inputs) -> List[str]:
        response = self.gateway_client.query(
            route=self.route_name, data=model_inputs
        )

        return response["candidates"]


# COMMAND ----------

# DBTITLE 1,Create AI Gateway route to model serving endpoint
# Token to authenticate with the model serving endpoint. 
# This can be a personal access token, service principal token, etc.
api_token = dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY_PAT)

notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
workspace_url = notebook_context.browserHostName().getOrElse(None)

route_name = AI_GATEWAY_ROUTE_NAME_MODEL_SERVING

route_config = {
    "route_type": "llm/v1/completions",
    "model": {
        "name": MODEL_SERVING_ENDPOINT_NAME,
        "provider": "databricks-model-serving",
        "databricks_model_serving_config": {
            "databricks_workspace_url": f"https://{workspace_url}",
            "databricks_api_token": api_token,
        },
    },
}

aig = AIGatewayHelper(route_name)
route = aig.create_route(route_config)


# COMMAND ----------

# DBTITLE 1,Create route to MosaicML Inference (Llama 2 70B)
mosaicml_api_key = dbutils.secrets.get(
    scope=SECRET_SCOPE, key=SECRET_KEY_MOSAICML
)
route_name = AI_GATEWAY_ROUTE_NAME_MOSAIC_70B

route_config = {
    "route_type": "llm/v1/completions",
    "model": {
        "name": "llama2-70b-chat",
        "provider": "mosaicml",
        "mosaicml_config": {
            "mosaicml_api_key": mosaicml_api_key,
        },
    },
}

aig = AIGatewayHelper(route_name)
route = aig.create_route(route_config)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Test AI Gateway Routes

# COMMAND ----------

model_route = AIGatewayHelper(MODEL_SERVING_ENDPOINT_NAME)
print(model_route.query({"prompt": "What is MLflow?"}))


# COMMAND ----------

# DBTITLE 1,Route prompt to MosaicML Inference (Llama 2 70B)
route_name = AI_GATEWAY_ROUTE_NAME_MOSAIC_70B
model_route = AIGatewayHelper(route_name)

print(
    model_route.query(
        {
            "prompt": """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. The primary purpose of your job is to meticulously identify privacy risks in any given documentation. You are thorough. If you don't know the answer to a question, please don't share false information.
<</SYS>>

List A: Here is a list of personally identifiable information (PII) types:
- BROWSER_VERSION
- BROWSER_TYPE
- BROWSER_USER_AGENT
- OS_PLATFORM
- USER_AGENT

Be meticulous in your analysis. Only pick items from List A only. Do not include items not in List A.
Look at the below list of PII categories. For each category, answer this question: Which items in List A fit in this PII category? Double-check your answer to ensure it only contains items from List A. If there are no related items for a type, answer with an empty set.
- Person names
- Email addresses

Verify your answer. Eliminate any items that are not in List A.

Now return your answer in JSON only. Ignore categories where you didn't find matching items. Do not provide explanations. 
JSON format: {"Person names": ["NAME", "SURNAME",],} 
[/INST]
""",
            "temperature": 0.01,
            "max_tokens": 2048,
        }
    )
)


# COMMAND ----------

# DBTITLE 1,Route prompt to model serving endpoint
route_name = AI_GATEWAY_ROUTE_NAME_MODEL_SERVING
model_route = AIGatewayHelper(route_name)

print(
    model_route.query(
        {
            "prompt": """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. The primary purpose of your job is to meticulously identify privacy risks in any given documentation. You are thorough. If you don't know the answer to a question, please don't share false information.
<</SYS>>

You reply in valid JSON only. 
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
Respond only with valid JSON. Do not provide explanations. Do not provide commentary. JSON:
{"pii_detected": [{"pii_type": "", "value": ""}]}
(e.g. {"pii_detected": [{"pii_type": "NAME", "value": "Harry},{"pii_type": "NAME", "value": "Susan"}]})

<start of text to analyse>
Dear Dr. III, we are updating our email system and it appears your email (Sherwood_Zboncak@yahoo.com) has not been migrated yet. Please verify your password (xQoJ9X2HLmmT) and we will complete the migration faster.
<end of text>
[/INST]
""",
            "temperature": 0.01,
            "max_tokens": 2048,
        }
    )
)


# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
