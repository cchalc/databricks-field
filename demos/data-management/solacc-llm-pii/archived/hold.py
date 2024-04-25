# Databricks notebook source
# MAGIC %md
# MAGIC # Load & test the model
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

# DBTITLE 1,Import packages again if cluster has been detached and reattached
import html
import llm_utils
import mlflow
import os
import pandas as pd

from pyspark.sql.functions import col
from pyspark.sql.types import StringType

from config import (
    MLFLOW_MODEL_NAME,
    TEST_DATA,
    USE_UC,
    CATALOG_NAME,
    SCHEMA_NAME,
)


model_name = MLFLOW_MODEL_NAME
model_uri = f"models:/{model_name}/Production"

if USE_UC:
    # Set registry to Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MLFLOW_MODEL_NAME}"
    model_uri = f"models:/{model_name}@Champion"

client = mlflow.tracking.MlflowClient()

# Disable progress bars for cleaner outputs
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "False"


# COMMAND ----------

# DBTITLE 1,Retrieve prompts from logged model parameters
latest_model_version = llm_utils.get_model_latest_version(model_name)


def get_prompt_template(template_name: str) -> str:
    return llm_utils.get_model_param(
        model_name, template_name, latest_model_version
    )


system_prompt = get_prompt_template("system_prompt")
user_instruction_profile_pii = get_prompt_template(
    "user_instruction_profile_pii"
)
user_instruction_categorise_department = get_prompt_template(
    "user_instruction_categorise_department"
)
user_instruction_filter_pii_types = get_prompt_template(
    "user_instruction_filter_pii_types"
)

# View prompts
displayHTML(
    f"""<h1>Retrieved prompts</h1>
<h2>Prompt: System Prompt</h2><pre>{html.escape(system_prompt)}</pre><hr />
<h2>Prompt: Categorise department</h2><pre>{user_instruction_categorise_department}</pre><hr />
<h2>Prompt: Profile PII</h2><pre>{user_instruction_profile_pii}</pre><hr />
<h2>Prompt: Filter PII types</h2><pre>{user_instruction_filter_pii_types}</pre>
"""
)


# COMMAND ----------

# DBTITLE 1,Load the latest logged model into the GPU
loaded_model = mlflow.pyfunc.load_model(model_uri)
print(f"Loaded model: {loaded_model}")


# COMMAND ----------

print(loaded_model.predict({"prompt": "What is ML?"}))

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

df = pd.DataFrame(test_prompts, columns=["prompt"])
# df = pd.DataFrame(test_prompts)


# Run inference
# results = loaded_model.predict(context, {"prompt": test_prompts})
results = loaded_model.predict(df)
print(results)

