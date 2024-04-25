# Databricks notebook source
# MAGIC %md
# MAGIC # Processing our Data
# MAGIC
# MAGIC Now that we have a model and set of verified prompts, let's look at processing our data
# MAGIC
# MAGIC Note that we iteratively apply our prompts to the data. We could apply all prompts to our data in one shot. However, for the purposes of this discussion, we'll progress step by step to explore how our LLM responds to our various requests.
# MAGIC
# MAGIC The steps we will take are illustrated below
# MAGIC
# MAGIC <img src="https://github.com/vinoaj/databricks-resources/blob/main/assets/img/solacc-data-flow.png?raw=true" width="1200">
# MAGIC
# MAGIC - Generate a prompt for each row of data to (a) categorise which department they belong to and (b) to profile the PII
# MAGIC - Run the prompts through our served model
# MAGIC - Parse the data
# MAGIC - Persist the profiled data in `bronze_documents_pii_handled`
# MAGIC - Quarantine false positives
# MAGIC - With the remaining detected PII types, determine which ones need to be redacted based on the department's redaction rules
# MAGIC - Redact the data
# MAGIC - Create the silver layer with per-department tables
# MAGIC

# COMMAND ----------

# Uncomment the below if you're not using a cluster init script
# See init_script.sh for a sample init script
# Documentation on using init scripts: https://docs.databricks.com/en/init-scripts/index.html

# Init script path: /Workspace/Users/vinny.vijeyakumaar@databricks.com/init_script.sh
# %pip install -Uq accelerate==0.23.0 bitsandbytes==0.41.1 mlflow transformers==4.33.2 xformers==0.0.21 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

# # Restart the Python kernel
# dbutils.library.restartPython()

# COMMAND ----------

import html
import json
import llm_utils
import mlflow
import os
import pandas as pd
import time
import requests

from pyspark.sql.functions import col, monotonically_increasing_id, expr, pandas_udf, to_json
from pyspark.sql.types import ArrayType, StringType

from config import (
    AI_GATEWAY_ROUTE_NAME_MODEL_SERVING,
    AI_GATEWAY_ROUTE_NAME_MOSAIC_70B,
    MLFLOW_MODEL_NAME,
    CATALOG_NAME,
    SCHEMA_NAME,
    MODEL_SERVING_ENDPOINT_NAME,
    USE_UC,
)


# COMMAND ----------

# Use "ai4privacy_pii_unmasked_text_sample" if you just want to work with a sample of the data
SOURCE_TABLE = "ai4privacy_pii_unmasked_text"

# Schemas of the expected outputs from our prompts
SCHEMA_DEPT_CATEGORY = "STRUCT<departments: ARRAY<STRING>>"
SCHEMA_PII_PROFILE = (
    "STRUCT<pii_detected:ARRAY<STRUCT<pii_type:STRING,value:STRING>>>"
)

SCHEMA_DEPT_CATEGORY_EXTRACTED = "ARRAY<STRING>"
SCHEMA_PII_PROFILE_EXTRACTED = "ARRAY<STRUCT<pii_type:STRING,value:STRING>>"

# Set default catalog and schema
_ = spark.sql(f"USE CATALOG {CATALOG_NAME}")
_ = spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

# spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)

if USE_UC:
    # Set model registry to Unity Catalog
    mlflow.set_registry_uri("databricks-uc")

# Instantiate MLflow client
client = mlflow.tracking.MlflowClient()

model_name = MLFLOW_MODEL_NAME
model_uri = f"models:/{model_name}/Production"
if USE_UC:
    model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MLFLOW_MODEL_NAME}"
    model_uri = f"models:/{model_name}@Champion"

# Display convenience links
displayHTML(
    f"""<a href='/explore/data/{CATALOG_NAME}/{SCHEMA_NAME}/'>Link to Unity Catalog Schema "{CATALOG_NAME}.{SCHEMA_NAME}"</a><br />
<br />
Utilising model: {model_name} at <br />
<a href='{llm_utils.get_model_serving_endpoint_ui_url(MODEL_SERVING_ENDPOINT_NAME)}'>Serving endpoint {MODEL_SERVING_ENDPOINT_NAME}</a>"""
)


# COMMAND ----------

# MAGIC %md
# MAGIC # Load Prompt Templates
# MAGIC
# MAGIC We'll also pull the user instruction template from the model's logged parameters. We could overwrite them with our own logic here. However, it's always best to utilise what was determined to be the best fit template at the time of the model's development.
# MAGIC
# MAGIC With each prompt we send to the model, we'll also wrap it around a **system prompt**. The system prompt allows us to define the "operating system" of the model. We can instruct it to be safe and to minimise hallucinations.
# MAGIC
# MAGIC <img src="https://github.com/vinoaj/databricks-resources/blob/main/assets/img/solacc-llama2-system-prompt.png?raw=true" width="1000">
# MAGIC
# MAGIC If you wish to iterate on these prompts, you can override them as you create the prompting UDFs below.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Retrieve prompts
latest_model_version = llm_utils.get_model_latest_version(model_name)

def get_template(template_name: str) -> str:
    return llm_utils.get_model_param(
        model_name, template_name, latest_model_version
    )


system_prompt = get_template("system_prompt")
user_instruction_profile_pii = get_template("user_instruction_profile_pii")
user_instruction_categorise_department = get_template("user_instruction_categorise_department")
user_instruction_filter_pii_types = get_template("user_instruction_filter_pii_types")

# View prompts
displayHTML(
    f"""<h1>Retrieved prompts</h1>
<h2>Prompt: System Prompt</h2><pre>{html.escape(system_prompt)}</pre><hr />
<h2>Prompt: Categorise department</h2><pre>{user_instruction_categorise_department}</pre><hr />
<h2>Prompt: Profile PII</h2><pre>{html.escape(user_instruction_profile_pii)}</pre><hr />
<h2>Prompt: Filter PII types</h2><pre>{user_instruction_filter_pii_types}</pre>
"""
)

# COMMAND ----------

# DBTITLE 1,You can override the prompt templates if you wish
# system_prompt = """<s>[INST] <<SYS>>
# You always speak like a pirate.
# <</SYS>>

# {instruction}[/INST]"""

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Serving endpoint interaction (UDF)
# MAGIC
# MAGIC We'll create a Pandas UDF called **`pii_bot`** to interact with our model serving endpoint.
# MAGIC
# MAGIC We could also interact with the served model via the MLflow AI Gateway route we created before. However, at the time of writing, the `mlflow.gateway.query()` function only takes a single prompt. This is handy for ad-hoc usage of the model. However, when we want to do efficient batch ETL, sending queries row-by-row is not efficient.
# MAGIC
# MAGIC So, instead, we call the model serving endpoint directly and send requests in batch.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Define pii_bot()
api_root = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
)
api_token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

MAX_RETRIES = 3
WAIT_SECONDS = 5


@pandas_udf("string")
def pii_bot(prompts: pd.Series) -> pd.Series:
    """
    Executes batch inference on a model with given prompts, formatted with a system prompt,
    and returns the primary text output for each prompt in a new pandas Series.

    Parameters:
    - prompts (pd.Series): Text prompts to be processed.

    Returns:
    - pd.Series: Primary text output from the model for each input prompt.

    Raises:
    - Exception: If the HTTP request to the model serving endpoint fails.
    """
    # Wrap system prompt around input prompts
    formatted_prompts = prompts.map(
        lambda prompt: system_prompt.format(instruction=prompt)
    )

    # Batch inference
    inference_data = {
        "inputs": {"prompt": formatted_prompts.to_list()},
        "params": {"max_tokens": 2048, "temperature": 0.0, "candidate_count": 1},
    }

    headers = {
        "Context-Type": "text/json",
        "Authorization": f"Bearer {api_token}",
    }

    # Introduce an N-second delay between API requests to prevent overwhelming the API endpoint
    time.sleep(WAIT_SECONDS)

    response = requests.post(
        url=f"{api_root}/serving-endpoints/{MODEL_SERVING_ENDPOINT_NAME}/invocations",
        json=inference_data,
        headers=headers,
    )

    # # Retry mechanism
    # for i in range(MAX_RETRIES):
    #     response = requests.post(
    #         url=f"{api_root}/serving-endpoints/{MODEL_SERVING_ENDPOINT_NAME}/invocations",
    #         json=inference_data,
    #         headers=headers,
    #     )

    #     if response.status_code != 504:  # If not a timeout, break out of the loop
    #         break

    #     if i < MAX_RETRIES - 1:  # Don't sleep after the last attempt
    #         time.sleep(WAIT_SECONDS)

    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    response_json = response.json()

    results = []
    for prediction in response_json["predictions"]:
        results.append(prediction["candidates"][0]["text"])

    return pd.Series(results)


# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC # Define helper UDFs
# MAGIC
# MAGIC We define a set of UDFs to help us generate the relevant prompts for each row in our dataset. The UDF will take each record, and inject it into the user instruction templates we loaded above.

# COMMAND ----------

# MAGIC %md
# MAGIC ## UDF: PII profiling prompt

# COMMAND ----------

print(user_instruction_profile_pii)

# COMMAND ----------

# You can override the prompt template here if you are experimenting with new prompts
# user_instruction_profile_pii = """...{text_to_analyse}..."""

@pandas_udf("string")
def create_profiling_prompt(data: pd.Series) -> pd.Series:
    """
    Generate a profiling prompt for each element in the data series.

    Uses a predefined user instruction template (`user_instruction_profile_pii`)
    to format the text entries in the data series.

    Parameters:
    - data (pd.Series): A pandas Series containing text entries to be formatted.

    Returns:
    - pd.Series: A pandas Series containing the formatted profiling prompts.
    """

    prompts = data.map(
        lambda text: user_instruction_profile_pii.format(text_to_analyse=text)
    )
    return prompts


# COMMAND ----------

# MAGIC %md
# MAGIC ## UDF: Department categorisation prompt creation

# COMMAND ----------

print(user_instruction_categorise_department)

# COMMAND ----------

# You can override the prompt template here if you are experimenting with new prompts
# user_instruction_categorise_department = """...{text_to_analyse}..."""

@pandas_udf("string")
def create_department_categorisation_prompt(data: pd.Series) -> pd.Series:
    prompts = data.map(
        lambda text: user_instruction_categorise_department.format(
            text_to_analyse=text
        )
    )
    return prompts


# COMMAND ----------

# MAGIC %md
# MAGIC # Define Bronze table to hold all LLM outputs
# MAGIC
# MAGIC We'll define a table in our bronze layer that will hold all the responses from our LLM for each of the messages we're processing

# COMMAND ----------

_ = spark.sql(
    f"""
      CREATE OR REPLACE TABLE {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled
      (
          message_id STRING COMMENT 'Unique identifier for the message',
          unmasked_text STRING COMMENT 'Raw unmasked text content',
          departments {SCHEMA_DEPT_CATEGORY_EXTRACTED} COMMENT 'List of departments associated with the message',
          pii_detected {SCHEMA_PII_PROFILE_EXTRACTED} COMMENT 'List of PII detected with type and value'
      )
      COMMENT "Messages with profiling information generated from PII Bot"
      TBLPROPERTIES
      (
          "sensitivity" = "high",
          "quality" = "bronze"
      )
"""
)


# COMMAND ----------

# MAGIC %md
# MAGIC # ETL: Generate prompts
# MAGIC
# MAGIC First, we'll generate a prompt for each record in our table. In subsequent steps we'll pass those prompts to our model.
# MAGIC
# MAGIC We could have had the prompt logic baked into our model. However, this locks our model's usefulness down to a very specific use case. That is why we load a generalisable model, and then feed it specific prompts depending on our needs.

# COMMAND ----------

# DBTITLE 1,View subset of source data
display(spark.read.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.{SOURCE_TABLE}").limit(3))

# COMMAND ----------

# DBTITLE 1,Generate prompts
df_unmasked = spark.sql(
    f"""
    SELECT message_id, unmasked_text
    FROM {CATALOG_NAME}.{SCHEMA_NAME}.{SOURCE_TABLE}
    -- LIMIT 300
    """
)

# Apply the prompt creation UDFs to the DataFrame
df_with_prompts = df_unmasked.withColumn(
    "prompt_categorisation",
    create_department_categorisation_prompt(col("unmasked_text")),
).withColumn("prompt_profiling", create_profiling_prompt(col("unmasked_text")))

# Write to delta table
target_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_prompts"
(
    df_with_prompts.write.format("delta")
    .mode("overwrite")
    .saveAsTable(target_table)
)

df_with_prompts.cache()


# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the prompts that were generated. 
# MAGIC - `prompt_categorisation`: holds the prompts that ask the LLM to categorise which departments the message likely belongs to
# MAGIC - `prompt_profiling`: holds the prompts that ask the LLM to identify and return PII in the `unmasked_text`
# MAGIC
# MAGIC If you expand any of these prompts, you'll note that it is simply the relevant prompt template, with the `unmasked_text` injected at the bottom of the template.
# MAGIC
# MAGIC We could generate these prompts on the fly. However, it's good to have a record of the prompt that was sent to the LLM. If we identify issues with the responses, we can easily conduct troubleshooting with the original prompt.
# MAGIC

# COMMAND ----------

# DBTITLE 1,View generated prompts
display(df_with_prompts)

# COMMAND ----------

# MAGIC %md
# MAGIC # ETL: Categorise messages
# MAGIC
# MAGIC Now that we have our prompts, let's categorise our messages by departments they may belong to. We pass the prompt in `prompt_categorisation` to the `pii_bot()` UDF for the model's response.
# MAGIC
# MAGIC In the below code, we batch our data into batches of 40 rows, and send each batch _sequentially_ to the model serving endpoint. This is to ensure the endpoint isn't overwhelmed with too many concurrent requests. 

# COMMAND ----------

# DBTITLE 1,Parallel processing with UDF
# # df_with_prompts.repartition(16)

# df_categorised = df_with_prompts.select(
#     col("message_id"),
#     col("unmasked_text"),
#     pii_bot(col("prompt_categorisation")).alias("category_raw"),
# ).limit(35)

# # Cache the DataFrame so that subsequent references to it don't trigger
# #   repeated calls to the model
# df_categorised.cache()

# display(df_categorised)


# COMMAND ----------

def batch_process_prompts(df_with_prompts, prompt_col_name:str, result_col_name:str):
    N_PROMPTS_PER_BATCH = 40
    WAIT_MINS = 2

    # 1. Add a monotonically increasing id to keep track of the order
    df_with_id = df_with_prompts.withColumn("mono_id", monotonically_increasing_id())

    # 2. Use floor division to create batch ids
    df_with_batches = df_with_id.withColumn("batch_id", (col("mono_id") % N_PROMPTS_PER_BATCH).cast("long"))

    # display(df_with_batches.limit(N_PROMPTS_PER_BATCH*2))

    results = []
    for i in range(N_PROMPTS_PER_BATCH):
        print(f"Processing batch {i} / {N_PROMPTS_PER_BATCH-1}")

        batch_df = df_with_batches.filter(col("batch_id") == i)
        processed_batch = batch_df.withColumn(result_col_name, pii_bot(col(prompt_col_name)))
        results.append(processed_batch)

        if i < N_PROMPTS_PER_BATCH - 1:
            # Wait n mins
            time.sleep(WAIT_MINS*60)

    # Union all results
    df_results = results[0]
    for result_df in results[1:]:
        df_results = df_results.union(result_df)

    return df_results


# COMMAND ----------

df_categorised = batch_process_prompts(df_with_prompts, "prompt_categorisation", "category_raw")
df_categorised.cache()
display(df_categorised)

# COMMAND ----------

# MAGIC %md
# MAGIC Great! We have some profiled data, and it's looking promising. However, the "JSON" responses the LLM has given us is really returned as a string. Now let's use the `FROM_JSON` function to convert those strings to proper struct types. Having it in a proper format will make further analysis and downstream work more straightforward.
# MAGIC
# MAGIC Because we were opinionated about the format the LLM had to speak to us in, we can confidently provide a schema to the `FROM_JSON()` function. Our schema defintion is
# MAGIC ```
# MAGIC STRUCT<
# MAGIC   departments: ARRAY<STRING>
# MAGIC >
# MAGIC ```
# MAGIC
# MAGIC [Read more about `FROM_JSON()` here](https://docs.databricks.com/en/sql/language-manual/functions/from_json.html)

# COMMAND ----------

df_categorised_parsed = df_categorised.selectExpr(
    "message_id",
    "unmasked_text",
    # Convert string to JSON struct data type
    "FROM_JSON(category_raw, 'STRUCT<departments: ARRAY<STRING>>') AS parsed_json",
    "category_raw",
).selectExpr(
    "message_id",
    "unmasked_text",
    # Pull out "departments" value from JSON
    "parsed_json.departments AS departments",
    "parsed_json",
    "category_raw",
)

df_categorised_parsed.cache()
display(df_categorised_parsed)


# COMMAND ----------

# DBTITLE 1,Persist categorisations
df_subset = df_categorised_parsed.select("message_id", "unmasked_text", "departments")

# Create temporary view, which allows us to use the contents of this 
#   DataFrame in the SQL query below
df_subset.createOrReplaceTempView("tmp_vw_categorised")

_ = spark.sql(
    f"""
  MERGE INTO {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled target 
    USING tmp_vw_categorised source
  ON target.message_id = source.message_id
  WHEN MATCHED THEN UPDATE SET target.departments = source.departments
  WHEN NOT MATCHED THEN INSERT (message_id, unmasked_text, departments) 
    VALUES (source.message_id, source.unmasked_text, source.departments)
  """
)


# COMMAND ----------

# DBTITLE 1,View subset of data
display(spark.read.table(f"bronze_documents_pii_handled").limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC # ETL: Profile PII for each message
# MAGIC
# MAGIC Now let's get to the core of what we are trying to achieve with this solution: identifying potential PII in unstructured messages

# COMMAND ----------

# DBTITLE 1,Parallel UDF method
# df_profiled = (
#     df_with_prompts.repartition(16)
#     .select(
#         "message_id",
#         "unmasked_text",
#         pii_bot(col("prompt_profiling")).alias("profile_raw")
#     )
# )

# df_profiled.cache()
# display(df_profiled)


# COMMAND ----------

# Not sure what's happening, but model serving always fails (504) on the last batch!
df_profiled = batch_process_prompts(df_with_prompts, "prompt_profiling", "profile_raw")
display(df_profiled)

# COMMAND ----------

# MAGIC %md
# MAGIC Like before, let's use the `FROM_JSON` function to convert those strings to proper struct types.
# MAGIC
# MAGIC Our schema defintion is
# MAGIC ```
# MAGIC STRUCT<
# MAGIC   pii_detected:ARRAY<
# MAGIC     STRUCT<
# MAGIC       pii_type:STRING,
# MAGIC       value:STRING
# MAGIC     >
# MAGIC   >
# MAGIC >
# MAGIC ```
# MAGIC
# MAGIC [Read more about `FROM_JSON()` here](https://docs.databricks.com/en/sql/language-manual/functions/from_json.html)

# COMMAND ----------

df_pii_parsed = df_profiled.selectExpr(
    "message_id",
    "unmasked_text",
    "FROM_JSON(profile_raw, 'STRUCT<pii_detected:ARRAY<STRUCT<pii_type:STRING,value:STRING>>>') AS parsed_json",
    "profile_raw",
).selectExpr(
    "message_id",
    "unmasked_text",
    "parsed_json.pii_detected AS pii_detected",
    "profile_raw",
)

df_pii_parsed.cache()
display(df_pii_parsed)


# COMMAND ----------

# MAGIC %md
# MAGIC You may notice some `null`s in the `pii_detected` column. This will occur if the model responded with content other than a JSON string representation.
# MAGIC
# MAGIC This is OK. Later on you'll see we quarantine these results. The quarantined prompts can be used to gather insights into why the model may have been confused by the instructions it received. We can utilise those insights to further refine our prompts.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Let's look at some of the PII types detected by our PII Bot. You'll notice by looking at these results, that we've detected types of PII that weren't explicitly provided as examples in our prompt instructions. Items such as `BITCOIN_ADDRESS`, `BIC`, and `IBAN`. This is great to see as it shows us that the model can "think" beyond it's explicit instructions.

# COMMAND ----------

# Select unique "pii_type" values in pii_arr column
df_unique_pii = (
    df_pii_parsed.selectExpr("explode(pii_detected)")
    .select("col.pii_type")
    .distinct()
    .orderBy("pii_type")
)

display(df_unique_pii)


# COMMAND ----------

# DBTITLE 1,Persist profiled information
df_pii_subset = (df_pii_parsed.select("message_id", "unmasked_text", "pii_detected"))

df_pii_subset.createOrReplaceTempView("tmp_vw_pii_profiled")

_ = spark.sql(
    f"""
  MERGE INTO {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled target 
    USING tmp_vw_pii_profiled source
  ON target.message_id = source.message_id
  WHEN MATCHED THEN UPDATE SET target.pii_detected = source.pii_detected
  WHEN NOT MATCHED THEN INSERT (message_id, unmasked_text, pii_detected) 
    VALUES (source.message_id, source.unmasked_text, source.pii_detected)
"""
)


# COMMAND ----------

# DBTITLE 1,OPTIMIZE table
_ = spark.sql(
    f"OPTIMIZE {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled"
)


# COMMAND ----------

display(
    spark.read.table(
        f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled"
    ).limit(3)
)


# COMMAND ----------

# MAGIC %md
# MAGIC # ETL: Quarantine poor results
# MAGIC
# MAGIC LLMs are imperfect. They can provide hallucinations, example content, or poorly-formatted results.
# MAGIC
# MAGIC ## Handling false positives
# MAGIC
# MAGIC It's possible that the LLM can have provided some false positives. Let's do two things with false positives:
# MAGIC - Remove them from the dataset we'll send downstream
# MAGIC - Quarantine the false positives in another table that we can periodically review to provide us with insights to fine-tune our prompts
# MAGIC
# MAGIC <img src="https://github.com/vinoaj/databricks-resources/blob/main/assets/img/solacc-false-positives.png?raw=true" width="600">
# MAGIC
# MAGIC To detect false positives, we:
# MAGIC - Explode the `pii_detected` column so that there is a row for each `pii_type <-> value` pairs
# MAGIC - Check if the `value` occurs in the `unmasked_text` using the `INSTR()` function. If it doesn't, quarantine it
# MAGIC - Check if `value` is non-empty. If it's empty, quarantine it
# MAGIC
# MAGIC ## Handling poorly formatted results
# MAGIC
# MAGIC If we were expecting a response in the form of a JSON string representation, but the LLM provides extraneous data (e.g. explanations or greetings), the `FROM_JSON()` function we use will return a `null`. We can also quarantine away these instances.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Explode: create a single row for each pii_type:value pairs
_ = spark.sql(
    f"""
      CREATE OR REPLACE TEMPORARY VIEW view_bronze_documents_pii_handled_expanded
      AS
      WITH exploded AS (
        SELECT message_id, unmasked_text, departments, 
          EXPLODE(pii_detected) AS pii_detected
        FROM {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled
      )
      SELECT e.message_id, e.unmasked_text, e.departments, e.pii_detected.*,
        pr.prompt_profiling, pr.prompt_categorisation
      FROM exploded e
      INNER JOIN {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_prompts pr
        ON e.message_id = pr.message_id
  """
)

df = spark.sql(f"SELECT * FROM view_bronze_documents_pii_handled_expanded")
df.cache()
display(df)


# COMMAND ----------

# DBTITLE 1,Create quarantine table
quarantined_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled_quarantined"

# Uncomment this if you want to recreate the table from scratch
# _ = spark.sql(f"DROP TABLE IF EXISTS {quarantined_table}")

_ = spark.sql(
    f"""
      CREATE TABLE IF NOT EXISTS {quarantined_table}
      (
          message_id STRING COMMENT 'Unique identifier for the message',
          unmasked_text STRING COMMENT 'Raw unmasked text content',
          departments {SCHEMA_DEPT_CATEGORY_EXTRACTED} COMMENT 'List of departments associated with the message',
          pii_type STRING COMMENT 'Type of PII detected',
          value STRING COMMENT 'Value of PII detected',
          prompt STRING COMMENT 'Prompt utilised'
      )
      COMMENT "Quarantined LLM responses. Includes false positives and empty values (indicating malformed prompt response)"
      TBLPROPERTIES
      (
          "sensitivity" = "high",
          "quality" = "bronze"
      )
"""
)


# COMMAND ----------

# DBTITLE 1,Quarantine inaccurate results
source_view = "view_bronze_documents_pii_handled_expanded"

# Quarantine false positives or empty values
_ = spark.sql(
    f"""
      INSERT INTO {quarantined_table}
      SELECT message_id, unmasked_text, departments, pii_type, value, prompt_profiling
      FROM {source_view}
      WHERE 
        -- Detected value is not in the original text
        INSTR(unmasked_text, value) <= 0
        -- Value is empty or contains placeholder text
        OR value IN ("", "...")
      """
)

# Quarantine null pii_detected (signifies LLM didn't return a valid JSON string)
_ = spark.sql(
    f"""
    INSERT INTO {quarantined_table}
    SELECT b.message_id, b.unmasked_text, b.departments, 
      NULL as pii_type, NULL as value, pr.prompt_profiling
    FROM {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled b
    INNER JOIN {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_prompts pr
      ON b.message_id = pr.message_id
    WHERE b.pii_detected IS NULL
"""
)

# Quarantine null departments (signifies LLM didn't return valid JSON string)
_ = spark.sql(
    f"""
    INSERT INTO {quarantined_table}
    SELECT b.message_id, b.unmasked_text, NULL AS departments, 
      NULL as pii_type, NULL AS value, pr.prompt_categorisation
    FROM {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled b
    INNER JOIN {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_prompts pr
      ON b.message_id = pr.message_id
    WHERE b.departments IS NULL
"""
)

display(spark.read.table(quarantined_table))


# COMMAND ----------

# MAGIC %md
# MAGIC # ETL: Materialise cleansed results
# MAGIC
# MAGIC Now we'll rebuild our data utilising the data points that were not quarantined.

# COMMAND ----------

# DBTITLE 1,Rebuild working table by returning non-quarantined PII types and values into a structured data column
target_table = (
    f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled_cleansed"
)

_ = spark.sql(
    f"""
    CREATE OR REPLACE TABLE {target_table}
    COMMENT "PII Bot profiled information, with false positives and null results removed"
    TBLPROPERTIES
    (
        "sensitivity" = "high",
        "quality" = "bronze"
    )
    AS
    WITH expanded_filtered (
      SELECT *
      FROM view_bronze_documents_pii_handled_expanded
      WHERE INSTR(unmasked_text, value) > 0
        AND value NOT IN ("", "...")
        AND departments IS NOT NULL
    )
    SELECT message_id, unmasked_text, departments,
      COLLECT_LIST(STRUCT(pii_type, value)) AS pii_detected
    FROM expanded_filtered
    GROUP BY message_id, unmasked_text, departments
  """
)

display(spark.read.table(target_table))


# COMMAND ----------

# MAGIC %md
# MAGIC # ETL: Selectively redact sensitive information
# MAGIC
# MAGIC Now that we have our profiled data, we can decide what we want to do with this data downstream. For the purposes of this example we'll assume analysts in the below departments can only see the following fields:
# MAGIC - **Customer Support**: redact everything except name, email, and addresses
# MAGIC - **Cybersecurity**: redact everything except IP and MAC addresses
# MAGIC - **People Ops**: redact everything except except name and email
# MAGIC - **Technical Support**: redact everything except name and email
# MAGIC - **All Others**: redact everything
# MAGIC
# MAGIC Next we'll perform selective replacements of PII in the unmasked text depending on the department's PII redaction rules.
# MAGIC
# MAGIC While an LLM could be used for this, we found in our testing, that it responded erraticaly when asked to perform selective redaction. So we'll go with simple replacements in text. This also has the added benefit of being more performant.
# MAGIC
# MAGIC ## Identifying PII types to exclude from redaction
# MAGIC But before we do that, we need to check which of the detected PII types fit into our exclusion rules. We'll not mask them in the original text. As the PII types detected can vary over time with our data, we'll use an LLM here to determine which PII types detected match our exclusion rules.
# MAGIC
# MAGIC We'll utilise the Llama 2 70B model, as it proved to be more accurate with the nuances of our prompt. The prompt asks the model to:
# MAGIC - Look at a list of PII types
# MAGIC - Identify which of those PII types fit within our exclusion categories (e.g. person names, email addresses)
# MAGIC
# MAGIC ## Utilising MosaicML Inference (Llama 2 70B) via MLflow AI Gateway
# MAGIC Serving a 70B model is unnecessarily expensive to handle this straightforward prompt. It's more efficient for our team to utilise an already deployed version, where someone else is taking care of administration, maintenance, and performance optimisation.
# MAGIC
# MAGIC Therefore, we opt to utilise [MosaicML's Inference service](https://www.mosaicml.com/inference) which is already hosting [Llama2-70B-Chat](https://www.mosaicml.com/blog/llama2-inference) which we can securely interact with.
# MAGIC
# MAGIC Why haven't we used this before? With previous prompts, we have been injecting our customer's messages into the prompt. Where we have potentially sensitive data in the prompt, there is less risk with utilising a model that is served **within our** environment.
# MAGIC
# MAGIC However, for the purposes of identifying which PII types to exclude from redaction, the prompt we use doesn't contain any sensitive values (see below). So it's safe to utilise a model outside our Workspace for this.
# MAGIC
# MAGIC Prompt example:
# MAGIC ```plaintext
# MAGIC List A: Here is a list of personally identifiable information (PII) types:
# MAGIC - OS
# MAGIC - BROWSER
# MAGIC - PHONE
# MAGIC - IMEI
# MAGIC - URL
# MAGIC - USER_AGENT
# MAGIC - NAME
# MAGIC - EMAIL
# MAGIC
# MAGIC Be meticulous in your analysis. Only pick items from List A only. Do not include items not in List A.
# MAGIC Look at the below list of PII categories. For each category, answer this question: Which items in List A
# MAGIC fit in this PII category? Double-check your answer to ensure it only contains items from List A. If there
# MAGIC are no related items for a type, answer with an empty set.
# MAGIC - IP addresses
# MAGIC - MAC addresses
# MAGIC - Network addresses
# MAGIC
# MAGIC Verify your answer. Eliminate any items that are not in List A.
# MAGIC
# MAGIC Now return your answer in JSON only. Ignore categories where you didn't find matching items.
# MAGIC Do not provide explanations.
# MAGIC JSON format: {"Person names": ["NAME", "SURNAME",],}
# MAGIC ```
# MAGIC

# COMMAND ----------

# Express categories in natural language so that it's easier for the LLM to understand
DEPARTMENT_REDACTION_EXEMPTIONS = {
    "CUSTOMER_SUPPORT": ["Person names", "Email addresses", "Postal addresses"],
    "CYBERSECURITY": ["IP addresses", "MAC addresses", "Network addresses"],
    "PEOPLE_OPS": ["Person names", "Email addresses"],
    "TECHNICAL_SUPPORT": ["Person names", "Email addresses"],
}


# COMMAND ----------

# DBTITLE 1,Explode: one row for each pii_type:value pairs
_ = spark.sql(
    f"""
    CREATE OR REPLACE TEMPORARY VIEW view_bronze_pii_handled_all_exploded
    AS
    WITH departments_exploded AS (
      SELECT message_id, unmasked_text,
        EXPLODE(departments) AS department,
        pii_detected
      FROM {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled_cleansed
    ),
    pii_exploded AS (
      SELECT message_id, unmasked_text, department,
        EXPLODE(pii_detected) AS pii_detected
      FROM departments_exploded
    )
    SELECT message_id, unmasked_text, department,
      pii_detected.pii_type,
      pii_detected.value AS pii_value
    FROM pii_exploded
  """
)

display(spark.sql("SELECT * FROM view_bronze_pii_handled_all_exploded"))


# COMMAND ----------

# DBTITLE 1,Coalesce detected PII types per department
_ = spark.sql(
    f"""
  CREATE OR REPLACE TEMPORARY VIEW view_bronze_department_pii_types_agg
  AS
  SELECT department, 
    COLLECT_SET(pii_type) AS pii_types_identified
  FROM view_bronze_pii_handled_all_exploded
  GROUP BY department                
"""
)

df_agg = spark.sql("SELECT * FROM view_bronze_department_pii_types_agg")
df_agg.cache()

display(df_agg)


# COMMAND ----------

# MAGIC %md
# MAGIC Now let's look at selectively redacting data by department according to our rules listed above. Like before, we'll utilise a UDF to first generate our prompts.

# COMMAND ----------

# DBTITLE 1,Prompt template
print(user_instruction_filter_pii_types)

# COMMAND ----------

# DBTITLE 1,UDF: Prompt creation
@pandas_udf("string")
def create_filter_pii_prompt(
    departments: pd.Series, pii_types_identified: pd.Series
) -> pd.Series:
    prompts = []

    for department, pii_identified in zip(departments, pii_types_identified):
        department_exclusions = DEPARTMENT_REDACTION_EXEMPTIONS.get(
            department, []
        )

        if not department_exclusions:
            prompts.append("")
            continue  # Skip to the next iteration if there are no exclusions

        pii_found_list = "\n".join(
            [f"- {pii_type}" for pii_type in pii_identified]
        )
        department_exclusions_str = "\n".join(
            [f"- {exclusion}" for exclusion in department_exclusions]
        )

        prompt = user_instruction_filter_pii_types.format(
            pii_identified_list=pii_found_list,
            department_exclusions=department_exclusions_str,
        )

        prompts.append(prompt)

    return pd.Series(prompts)


# COMMAND ----------

# DBTITLE 1,Create prompts
df_prompts = df_agg.withColumn(
    "prompts",
    create_filter_pii_prompt(col("department"), col("pii_types_identified")),
)

df_prompts.cache()
display(df_prompts)


# COMMAND ----------

# DBTITLE 1,UDF: Interact with MosaicML Inference (Llama 2 70B) via AI Gateway
ai_gateway = llm_utils.AIGatewayHelper(AI_GATEWAY_ROUTE_NAME_MOSAIC_70B)

@pandas_udf("string")
def llama_70b(prompts: pd.Series) -> pd.Series:
    
    def query_dict(prompt: str) -> dict:
        return {
            "prompt": system_prompt.format(instruction=prompt),
            "temperature": 0.1,
            "max_tokens": 2048,
            "candidate_count": 1,
        }

    results = prompts.map(
        lambda prompt: ai_gateway.query(query_dict(prompt))[0]["text"]
    )

    return results


# COMMAND ----------

# Send prompts to MosaicML Inference (Llama 2 70B) service
df_with_response = (df_prompts
                    # Exclude departments with no exclusion rules
                    .where(df_prompts["prompts"] != "")
                    # Execute model inference 
                    .withColumn("response", llama_70b(col("prompts")) 
))

df_with_response.cache()
display(df_with_response)


# COMMAND ----------

# MAGIC %md
# MAGIC The response of the LLM looks something like
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "IP addresses": ["IMEI"],
# MAGIC   "Network addresses": ["URL"],
# MAGIC   "Person names": ["NAME", "EMAIL"]
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC Next, we want to coalesce all the arrays into a single list
# MAGIC

# COMMAND ----------

# DBTITLE 1,Merge observations into a single list
@pandas_udf(ArrayType(StringType()))
def json_string_to_array(strings: pd.Series) -> pd.Series:
    def parse_string(string: str):
        j = json.loads(string)
        items = []
        for k, v in j.items():
            items.extend(v)

        # Eliminate empty values
        items = [item.strip() for item in items if len(item.strip()) > 0]

        # Use set to return a list of unique items
        return list(set(items))

    lists = strings.map(lambda s: parse_string(s))
    return lists


df_filter_list = df_with_response.withColumn(
    "merged_list", json_string_to_array(col("response"))
).cache()
df_filter_list.createOrReplaceTempView("view_bronze_dept_filters")

display(df_filter_list)


# COMMAND ----------

# DBTITLE 1,Create view with texts, PII detected, and redaction exclusion lists
_ = spark.sql(
    f"""
      CREATE OR REPLACE TEMPORARY VIEW view_bronze_redact_ready
      AS
      WITH redaction_exclusion_filters AS (
        SELECT department, 
          merged_list AS redaction_exclusion_filters
        FROM view_bronze_dept_filters
      ),
      depts_exploded AS (
        SELECT message_id, unmasked_text,
          EXPLODE(departments) AS department,
          pii_detected
        FROM {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled_cleansed
      )
      SELECT de.*,
        COALESCE(ref.redaction_exclusion_filters, ARRAY()) AS redaction_exclusion_filters
      FROM depts_exploded de
      LEFT OUTER JOIN redaction_exclusion_filters ref
        ON de.department = ref.department
"""
)


# COMMAND ----------

display(spark.sql("SELECT * FROM view_bronze_redact_ready ORDER BY message_id, department"))

# COMMAND ----------

# DBTITLE 1,UDF: Redact text
@pandas_udf("string")
def redact_text(
    unmasked_texts: pd.Series,
    pii_detected: pd.Series,
    excluded_flags: pd.Series,
) -> pd.Series:
    redacted_texts = []

    for i in range(len(unmasked_texts)):
        unmasked_text = unmasked_texts[i]
        pii_list = pii_detected[i]
        exclusions = excluded_flags[i]

        for item_pii in pii_list:
            if item_pii["pii_type"] not in exclusions:
                unmasked_text = unmasked_text.replace(
                    item_pii["value"], f"[{item_pii['pii_type']}]"
                )

        redacted_texts.append(unmasked_text)

    return pd.Series(redacted_texts)


# COMMAND ----------

df = spark.sql("SELECT * FROM view_bronze_redact_ready")
df_redacted = df.withColumn(
    "redacted_text",
    redact_text(
        df.unmasked_text, df.pii_detected, df.redaction_exclusion_filters
    ),
)

target_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_pii_redacted"
# spark.sql(f"DROP TABLE IF EXISTS {target_table}")
df_redacted.write.mode("overwrite").saveAsTable(target_table)


# COMMAND ----------

# DBTITLE 1,View redaction results
display(spark.sql(
            f"""
            SELECT message_id, unmasked_text, department, redaction_exclusion_filters, redacted_text 
            FROM {target_table}
            ORDER BY message_id, department
            """)
        )


# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at a subset of messages where the redactions differ based on a department's exclusion rules. For example, in this message, you see that the Cybersecurity analysts can view IP addresses (e.g. if they want to analyse number of incidents by IP address spaces), but Technical Support analysts cannot
# MAGIC
# MAGIC <img src="https://github.com/vinoaj/databricks-resources/blob/main/assets/img/solacc-selective-redaction.png?raw=true" width="800">
# MAGIC

# COMMAND ----------

display(spark.sql(
        f"""
        WITH diff_redactions AS (
            SELECT a.message_id, a.unmasked_text, a.department, a.redaction_exclusion_filters, a.redacted_text 
            FROM {target_table} a
            JOIN {target_table} b 
                ON a.message_id = b.message_id AND a.redacted_text != b.redacted_text
            LIMIT 30
        )
        SELECT DISTINCT *
        FROM diff_redactions
        ORDER BY message_id, department
        """)
)


# COMMAND ----------

# MAGIC %md
# MAGIC # ETL: Create table for each department

# COMMAND ----------

# DBTITLE 1,Get list of departments
# Let's build a list of all identified departments
df_unique_depts = (
    spark.read.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled")
    .selectExpr("explode(departments) AS department")
    .distinct()
    .orderBy("department")
)

df_unique_depts.cache()
display(df_unique_depts)


# COMMAND ----------

source_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_pii_redacted"

for row in df_unique_depts.collect():
    dept_name = row.department
    target_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.silver_pii_profiled_dept_{dept_name.lower()}"

    spark.sql(f"DROP TABLE IF EXISTS {target_table}")
    
    # Create empty table
    _ = spark.sql(
        f"""
      CREATE TABLE IF NOT EXISTS {target_table}
      (
          message_id STRING COMMENT 'Unique identifier for the message',
          redacted_text STRING COMMENT 'PII redacted text',
          pii_exclusions ARRAY<STRING> COMMENT 'PII exclusions for {dept_name.lower()} department'
      )
      COMMENT "PII redacted text"
      TBLPROPERTIES
      (
          "sensitivity" = "high",
          "quality" = "silver"
      )"""
    )

    # Create view of department's redacted messages
    _ = spark.sql(f"""
              CREATE OR REPLACE TEMPORARY VIEW vw_tmp_{dept_name}
              AS
              SELECT message_id, redacted_text, redaction_exclusion_filters as pii_exclusions
              FROM {source_table}
              WHERE department = "{dept_name}"
              """)

    # Populate table
    _ = spark.sql(
        f"""
      -- SELECT message_id, redacted_text, redaction_exclusion_filters as pii_exclusions
      -- FROM {source_table}
      -- WHERE department = "{dept_name}"

        MERGE INTO {target_table} target
        USING vw_tmp_{dept_name} source
        ON target.message_id = source.message_id
        WHEN MATCHED THEN UPDATE SET target.redacted_text = source.redacted_text, target.pii_exclusions = source.pii_exclusions
        WHEN NOT MATCHED THEN INSERT *
    """
    )

    # df_target.write.mode("overwrite").insertInto(target_table)


# COMMAND ----------

# DBTITLE 1,View subsets of data
dept_names = ["CYBERSECURITY", "LEGAL"]

for dept_name in dept_names:
    display(spark.sql(f"SELECT * FROM {CATALOG_NAME}.{SCHEMA_NAME}.silver_pii_profiled_dept_{dept_name} LIMIT 10"))


# COMMAND ----------

# MAGIC %md
# MAGIC #ETL: Publish to Gold Layer
# MAGIC
# MAGIC Finally, we can take our redacted data and publish them to the gold layer for our downstream analysts to consume.
# MAGIC
# MAGIC Since these tables sit in Unity Catalog, ensure only the appropriate groups (e.g. `cybersecurity-analysts`) are given permissions to query these tables.
# MAGIC

# COMMAND ----------

for row in df_unique_depts.collect():
    dept_name = row.department
    dept_str = dept_name.lower()

    source_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.silver_pii_profiled_dept_{dept_str}"
    target_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.gold_pii_profiled_dept_{dept_str}"

    # Uncomment if you want to create the table from scratch
    # _ = spark.sql(f"DROP TABLE IF EXISTS {target_table}")

    # Create gold table
    _ = spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {target_table}
            (
                message_id STRING COMMENT 'Unique identifier for the message',
                redacted_text STRING COMMENT 'PII redacted text',
                pii_exclusions ARRAY<STRING> COMMENT 'PII exclusions for {dept_str} department'
            )
            COMMENT "PII redacted customer messages belonging to {dept_str} analysts"
            TBLPROPERTIES
            (
                "sensitivity" = "high",
                "department" = "{dept_str}",
                "quality" = "gold"
            )"""
        )

    # Upsert redactions
    _ = spark.sql(
        f"""
        MERGE INTO {target_table} target
        USING {source_table} source
        ON target.message_id = source.message_id
        WHEN MATCHED THEN UPDATE SET target.redacted_text = source.redacted_text, target.pii_exclusions = source.pii_exclusions
        WHEN NOT MATCHED THEN INSERT *
        """
    )


# COMMAND ----------

# DBTITLE 1,View subsets of data
dept_names = ["CYBERSECURITY", "LEGAL"]

for dept_name in dept_names:
    display(spark.sql(f"SELECT * FROM {CATALOG_NAME}.{SCHEMA_NAME}.gold_pii_profiled_dept_{dept_name} LIMIT 10"))


# COMMAND ----------

# MAGIC %md
# MAGIC # Address quarantined results
# MAGIC
# MAGIC Now that we've materialised our gold layer, let's take a look at the data that didn't make it through. 
# MAGIC
# MAGIC Let's revisit our quarantined data. The affected messages have not made it into the silver or gold layers. By looking at the original `unmasked_text`, prompt `prompt_profiling`, and results, we can derive insights as to why the LLM may have faltered. 
# MAGIC
# MAGIC Using these insights, we can fine-tune our prompts. Strategies we may take:
# MAGIC
# MAGIC - Be prescriptive about what **not** to include in responses
# MAGIC - Provide more examples of true positives and true negatives
# MAGIC - Revisit the order of our instructions, and lay them out in a more logical manner
# MAGIC - Experiment with line breaks and spacing between logical task sets
# MAGIC

# COMMAND ----------

display(spark.read.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled_quarantined"))

# COMMAND ----------

# MAGIC %md
# MAGIC We can utilise the AI Gateway routes we created before to quickly iterate over prompt logic to identify prompts that may better work for us.
# MAGIC
# MAGIC Remember, we have two AI Gateway routes:
# MAGIC 1. Pointing to the **served model**: allows us to safely test prompts with sensitive data
# MAGIC 1. Pointing to MosaicML Inference for **Llama 2 70B**: allows us to (a) quickly iterate over prompts with non-sensitive data and (b) check if a 70B model may provide better results
# MAGIC
# MAGIC In fact, this is how we experimented with prompts that became the prompt templates for this solution accelerator.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Instantiate route connections and helper functions
aig_13B = llm_utils.AIGatewayHelper(AI_GATEWAY_ROUTE_NAME_MODEL_SERVING)
aig_70B = llm_utils.AIGatewayHelper(AI_GATEWAY_ROUTE_NAME_MOSAIC_70B)

def test_models(prompt:str):
  base_dict = {
    "prompt": prompt,
    "temperature": 0.01,
    "max_tokens": 2048,
  }

  response_13b = aig_13B.query(base_dict)
  response_70b = aig_70B.query(base_dict)

  print(f"13B: {response_13b}")
  print(f"70B: {response_70b}")

  print(f"""Model Serving (13B) response:
    {json.dumps(json.loads(response_13b[0]["text"]), indent=2)}

    Llama 2 70B response:
    {json.dumps(json.loads(response_70b[0]["text"]), indent=2)}""")
  

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can start testing our prompts. Be sure to include the system prompt along with user instruction prompt you're troubleshooting.
# MAGIC

# COMMAND ----------

print(system_prompt)

# COMMAND ----------

# DBTITLE 1,Problematic prompt hallucinating names Harry and Susan
prompt = """<s>[INST] <<SYS>>
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
Return valid JSON syntax only. Do not provide explanations. JSON:
{"pii_detected": [{"pii_type": "", "value": ""}]}
(e.g. {"pii_detected": [{"pii_type": "NAME", "value": "Harry},{"pii_type": "NAME", "value": "Susan"}]})

<start of text to analyse>
Can we also have a session on how to manage stress and cope in transition periods for our employees during these changes? Your involvement will certainly help address employee concerns.
<end of text>
[/INST]
"""

test_models(prompt)


# COMMAND ----------

# DBTITLE 1,Refined prompt that doesn't return hallucinations
prompt = """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. The primary purpose of your job is to meticulously identify privacy risks in any given documentation or message. You are thorough. If you don't know the answer, please don't share false information or examples.
<</SYS>>

You only consider what is in the below text to analyze. 
You reply in valid JSON only. Return a valid JSON object only. No other text. Do not explain your response. 

<start of text to analyze>
Can we also have a session on how to manage stress and cope in transition periods for our employees during these changes? Your involvement will certainly help address employee concerns.
<end of text to analyze>

Defining PII:
Personally identifiable information (PII) refers to any data that can be used to identify, contact, or locate a single person, either directly or indirectly.
PII includes but not limited to: valid email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver's licence numbers (LICENCE), MAC addresses (MAC), etc.
Pay careful attention for a person's name: use context clues, such as capitalisation, proper nouns (e.g. June), words following greetings, words following honorifics, and surrounding text, to help identify people's names. 
Brand names and product names are not PII.

Your task:
In the text to analyze meticulously and thoroughly detect all instances of PII.
Account for any spelling and grammatical errors.
Label each instance using format [PII_TYPE].
You do not provide examples or sample data. Do not provide instances that is not in the text to analyze.
There can be multiple of the same PII type in the text. Return every instance detected.

Instructions for your JSON response:
You return an empty set if there is no PII detected.
If you found PII, verify your answer. If a PII type and value is not in the text to analyze, remove it from your answer.
Return a valid JSON object only. No other text. Do not explain your response. 
JSON: 
{"pii_detected": [{"pii_type": "", "value": ""}]}
Example: 
{"pii_detected": [{"pii_type": "IP_ADDRESS", "value": "23.23.12.0"}]}
[/INST]"""

test_models(prompt)


# COMMAND ----------

# DBTITLE 1,Problematic prompt that identifies PHONE as a name or email address
prompt = """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. Always answer as helpfully as possible, while being safe. The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. If you don't know the answer to a question, please don't share false information.
<</SYS>>

List A: Here is a list of personally identifiable information (PII) types:
- PHONE

Be meticulous in your analysis. Only pick items from List A only. Do not include items not in List A.
Look at the below list of PII categories. For each category, answer this question: Which items in List A fit in this PII category? Double-check your answer to ensure it only contains items from List A. If there are no related items for a type, answer with an empty set.
- Person names
- Email addresses

Verify your answer. Eliminate any items that are not in List A.

Now return your answer in JSON only. Ignore categories where you didn't find matching items. Do not provide explanations. 
JSON format: {"Person names": ["NAME", "SURNAME",],}[/INST]"""

print(aig_70B.query({
  "prompt": prompt,
  "temperature": 0.01,
  "max_tokens": 2048,
}))


# COMMAND ----------

# DBTITLE 1,Refined prompt
prompt = """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. Always answer as helpfully as possible, while being safe. The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. If you don't know the answer to a question, please don't share false information.
<</SYS>>

<start of List A>
List A: Here is a list of personally identifiable information (PII) types:
- PHONE
<end of List A>

<start of List B>
List B: Here is a list of PII categories:
- Person names
- Email addresses
<end of List B>

Your task:
- Be meticulous in your analysis. It is OK if you end up with an empty set.
- Only pick items from List A. Do not include items not in List A.
- Step through the list of PII categories in List B. 
- For each PII category in List B, identify the items in List A that belong to that category.
- Only pick items that fit the category. If there are no related items, answer with an empty set.
- Double-check your answer and remove any items that don't belong to that category.
- Double-check your answer to ensure it only contains items from List A. 
- Eliminate List A items that don't match the List B PII category. 

Verify your answer: Step through each PII category and
- eliminate any items that are not in List A. 
- eliminate any items that don't belong to the PII category they are matched against.

Now return your answer in JSON only. Ignore PII categories where you didn't find matching items. Do not provide explanations. 
JSON format: {"PII category": ["", "",],}[/INST]"""

print(aig_70B.query({
  "prompt": prompt,
  "temperature": 0.0,
  "max_tokens": 2048,
}))


# COMMAND ----------

# DBTITLE 1,We can even ask the model for suggestions for improving the prompt
prompt = """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. Always answer as helpfully as possible, while being safe. The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Here is my previous instruction to you:
<start of previous instruction>
<start of List A>
List A: Here is a list of personally identifiable information (PII) types:
- PHONE
<end of List A>

<start of List B>
List B: Here is a list of PII categories:
- Person names
- Email addresses
<end of List B>

Be meticulous in your analysis. Only pick items from List A only. Do not include items not in List A.
Look at the list of PII categories in List B. 
For each PII category, answer this question: Which items in List A belong to this PII category? Only pick items that fit the category. If there are no related items, that is OK and answer with an empty set.
Double-check your answer to ensure it only contains items from List A. 

Verify your answer. Eliminate any items that are not in List A. Eliminate List A item not appropriate for the chosen List B PII category.

Now return your answer in JSON only. Ignore PII categories where you didn't find matching items. Do not provide explanations. 
JSON format: {"PII category": ["ITEM", "ITEM",],}
<end of previous instruction>

You responded with "{\n"Person names": ["PHONE"],\n"Email addresses": []\n}"

Phone isn't a type of person name. How can I refine my instruction so you don't repeat that mistake? I cannot change the format of List A and List B.[/INST]"""

print(aig_70B.query({
  "prompt": prompt,
  "temperature": 0.01,
  "max_tokens": 2048,
}))


# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
