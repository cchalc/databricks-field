# Databricks notebook source
!pip install --upgrade mosaicml-cli einops==0.7.0 composer==0.16.4 boto3 pandas transformers awscli

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mcli 
from mcli import finetune
from mcli.objects.secrets import SecretType, MCLIEnvVarSecret, MCLIDatabricksSecret, MCLIS3Secret

# COMMAND ----------

secrets_scope = "fox_news_llm"

MOSAIC_API_KEY        = dbutils.secrets.get(scope=secrets_scope, key="mosaic_api_key") 
AWS_ACCESS_KEY        = dbutils.secrets.get(scope=secrets_scope, key="aws_access_key") # TODO
AWS_SECRET_ACCESS_KEY = dbutils.secrets.get(scope=secrets_scope, key="aws_secret_access_key") # TODO
# HF_TOKEN              = dbutils.secrets.get(scope=secrets_scope,key="hf_token")
DB_HOST               = dbutils.secrets.get(scope=secrets_scope, key="databricks_workspace_url")
DB_TOKEN              = dbutils.secrets.get(scope=secrets_scope, key="databricks_workspace_token")

# Name of the S3 bucket used to store data and model checkpoints
bucket_name = "data-science-ml-nonprod-auxfiles"

# COMMAND ----------

dbutils.secrets.list(secrets_scope)

# COMMAND ----------

mcli.set_api_key(MOSAIC_API_KEY)

# COMMAND ----------

mcli.get_secrets()

# COMMAND ----------

# # Add MosaicML API token to run on the Mosaic platform. 
# mcli_api_secret = MCLIEnvVarSecret(
#     name="fox_mosaic_api_key", # This can be any string you want
#     secret_type=SecretType.environment,
#     key="MOSAICML_API_KEY", # Do not change this key
#     value = MOSAIC_API_KEY,
# )
# mcli.create_secret(mcli_api_secret)

# COMMAND ----------

# # MLflow integration - need a token for the workspace you are running in. Create a secret for the token, and the workspace url. 
# databricks_secret = MCLIDatabricksSecret(
#     name="databricks_mlflow", 
#     secret_type=SecretType.databricks,
#     host=DB_HOST,
#     token=DB_TOKEN,
# )
# mcli.create_secret(databricks_secret)

# COMMAND ----------

# # AWS S3 - need to create a secret with AWS credentials. This example uses us-west-2 as the region.
# aws_s3_secret = MCLIS3Secret(
#     name="fox_news_llm", # This can be any string you want
#     secret_type=SecretType.s3,
#     profile="default",
#     config=f"[default]\nregion=us-west-2\noutput=json", # Do not change the config/creds info
#     credentials=f"[default]\naws_access_key_id={AWS_ACCESS_KEY}\naws_secret_access_key={AWS_SECRET_ACCESS_KEY}",# Do not change the config/creds info
# )
# mcli.create_secret(aws_s3_secret)

# COMMAND ----------

mcli.get_secrets()

# COMMAND ----------

import pandas as pd

def view_cluster_info(data):
    # Extracting relevant information and creating a dataframe
    rows = []
    for cluster in data:
        for instance in cluster.cluster_instances:
            rows.append({
                "Cluster Name": cluster.name,
                "Provider": cluster.provider,
                "Allow Fractional": cluster.allow_fractional,
                "Allow Multinode": cluster.allow_multinode,
                "Instance Name": instance.name,
                "GPU Type": instance.gpu_type,
                "GPUs": instance.gpus,
                "CPUs": instance.cpus,
                "Memory": instance.memory,
                "Storage": instance.storage,
                "Nodes": instance.nodes
            })
    df = pd.DataFrame(rows)
    return df

view_cluster_info(mcli.get_clusters())

# COMMAND ----------

# Helper function to see the latest logs for each of our runs.
def show_latest_log(run):
    print(f"{list(mcli.get_run_logs(run=run.name))[-1]}")

# COMMAND ----------

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

MLFLOW_EXPERIMENT = "/Users/niall.turbitt@fox.com/fox_news/experiments/fox_news_ift"
TRAIN_DATA_PATH = "s3://data-science-ml-nonprod-auxfiles/fox_news_llm/__unitystorage/schemas/91b66fd9-bf44-4b74-86ad-abef6ebaa1b8/volumes/f0ae7c00-b383-4b37-bc45-ef36923e41a4/headline_generation_instruction_dataset.jsonl"
EVAL_DATA_PATH = "s3://data-science-ml-nonprod-auxfiles/fox_news_llm/__unitystorage/schemas/91b66fd9-bf44-4b74-86ad-abef6ebaa1b8/volumes/f0ae7c00-b383-4b37-bc45-ef36923e41a4/headline_generation_instruction_dataset_sample.jsonl"

# Used for checkpoints
SAVE_FOLDER = "s3://data-science-ml-nonprod-auxfiles/fox_news_llm/models/instruction_fine_tune/headline_generation/"

# COMMAND ----------

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

ift_run = finetune(
    model=MODEL_NAME, 
    train_data_path=TRAIN_DATA_PATH,
    save_folder=SAVE_FOLDER,
    eval_data_path=EVAL_DATA_PATH,
    training_duration="1ep",
    task_type="INSTRUCTION_FINETUNE", 
    experiment_tracker = {
        "experiment_name": MLFLOW_EXPERIMENT
        }
    )

ift_run_name = ift_run.name
print(ift_run_name)

# COMMAND ----------


