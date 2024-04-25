# Databricks notebook source
# MAGIC %md
# MAGIC # Data Download and Preparation
# MAGIC
# MAGIC In this Notebook, we'll download a sample dataset that contains examples of PII. This data will act as our incoming set of customer messages that may contain sensitive information.
# MAGIC
# MAGIC For the purposes of this Solution Accelerator we'll utilise an open dataset, hosted on [Hugging Face](https://huggingface.co/), that contains unstructured text (e.g. customer reviews, customer requests) that contains PII
# MAGIC
# MAGIC We've selected [AI4Privacy](https://ai4privacy.com/)'s [PII Masking dataset](https://huggingface.co/datasets/ai4privacy/pii-masking-65k) used for fine-tuning LLMs to detect and handle PII. However, for the purposes of this exercise, we won't be fine-tuning an LLM as it is unnecessary for this use case.
# MAGIC
# MAGIC This is a multi-lingual dataset. For the purposes of this walkthrough, we'll focus on the English samples.
# MAGIC
# MAGIC You can swap out this dataset with other Hugging Face [datasets used for training PII detectors](https://huggingface.co/datasets?sort=trending&search=pii).
# MAGIC
# MAGIC ___
# MAGIC
# MAGIC ## Cluster configuration
# MAGIC
# MAGIC We can utilise a single-node non-GPU cluster as this Notebook doesn't execute any inference or complex data transformations.
# MAGIC
# MAGIC - Single node
# MAGIC - Access mode: `Assigned` (can use Shared if not using cluster init script)
# MAGIC - DBR: `13.3+ ML`
# MAGIC - Node type: 16GB memory, 4 cores
# MAGIC   - AWS: `m4.xlarge`
# MAGIC   - Azure: XXXX
# MAGIC   - GCP: XXXX
# MAGIC

# COMMAND ----------

# DBTITLE 1,Load libraries and configuration parameters
from datasets import load_dataset

from config import (
    CATALOG_NAME,
    SCHEMA_NAME,
    VOLUME_NAME,
    VOLUME_PATH,
)

DATASET = "ai4privacy/pii-masking-65k"
# Data file(s) to cherry pick. For this dataset, we'll pick the English samples
DATAFILE = "english_balanced_10k.jsonl"


# COMMAND ----------

# DBTITLE 1,Create our data stores if they don't already exist
_ = spark.sql(f"USE CATALOG {CATALOG_NAME}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
_ = spark.sql(f"USE SCHEMA {SCHEMA_NAME}")
_ = spark.sql(
    f"""
              CREATE VOLUME IF NOT EXISTS {VOLUME_NAME}
              COMMENT 'Location for Hugging Face Downloads'
              """
)

displayHTML(
    f"<a href='/explore/data/{CATALOG_NAME}/{SCHEMA_NAME}'>Link to Catalog Explorer</a>"
)


# COMMAND ----------

# MAGIC %md
# MAGIC We load the dataset directly from Hugging Face's data hub utilising their `datasets.load_dataset` function
# MAGIC
# MAGIC The `data_files` parameter allows us to cherrypick files from the dataset's repository. In this case we're only loading the English dataset.

# COMMAND ----------

dataset = load_dataset(DATASET, data_files=DATAFILE)
print(dataset)


# COMMAND ----------

# DBTITLE 1,Save dataset to Parquet
target_filename = "pii_en.parquet"
dataset["train"].to_parquet(f"{VOLUME_PATH}/{target_filename}")


# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at this data. What we're interested in, for the purposes of this Solution Accelerator, is the text in the `umasked_text` column. These are raw messages that mimic typical internal and external customer communications. Most of them have some form of PII in them.

# COMMAND ----------

# DBTITLE 1,View Parquet data
path = f"{VOLUME_PATH}/{target_filename}"
df = spark.sql(f"SELECT * FROM parquet.`{path}`")

print(f"Displaying Parquet tables at: {path}")
display(df)


# COMMAND ----------

# MAGIC %md
# MAGIC Next, we'll load this data into a Delta table to gain the performance benefits of utilising [Delta Lake](https://docs.databricks.com/en/delta/index.html)

# COMMAND ----------

# DBTITLE 1,Create Delta table
_ = spark.sql(
    f"""
  CREATE OR REPLACE TABLE {CATALOG_NAME}.{SCHEMA_NAME}.ai4privacy_pii_raw_en
  COMMENT "ai4privacy PII dataset - English"
  TBLPROPERTIES (
      "sensitivity" = "high",
      "quality" = "raw"
  )
  AS
  SELECT *
  FROM parquet.`{VOLUME_PATH}/{target_filename}`      
"""
)


# COMMAND ----------

# MAGIC %md
# MAGIC We'll now create another table with the same data but we also generate a unique message ID (`message_id`) for each row. This data shape reflects how messages would usually be stored in a real-world scenario.

# COMMAND ----------

_ = spark.sql(
    f"""
  CREATE OR REPLACE TABLE {CATALOG_NAME}.{SCHEMA_NAME}.ai4privacy_pii_unmasked_text
  COMMENT "Raw incoming messages"
  TBLPROPERTIES (
      "sensitivity" = "high",
      "quality" = "raw"
  )
  AS
  SELECT 
    UUID() AS message_id, -- Generated random message ID
    unmasked_text
  FROM {CATALOG_NAME}.{SCHEMA_NAME}.ai4privacy_pii_raw_en
"""
)

# Finally, optimise the table
_ = spark.sql(
    f"OPTIMIZE {CATALOG_NAME}.{SCHEMA_NAME}.ai4privacy_pii_unmasked_text"
)


# COMMAND ----------

# DBTITLE 1,Inspect the data
df = spark.sql(
    f"SELECT * FROM {CATALOG_NAME}.{SCHEMA_NAME}.ai4privacy_pii_unmasked_text"
)
display(df)


# COMMAND ----------

# MAGIC %md
# MAGIC Now let's create a smaller dataset (~50 rows) which we can use while experimenting with our LLM. We use the [`TABLESAMPLE()` clause](https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-qry-select-sampling.html) to retrieve a randomised set of rows. Since `TABLESAMPLE` takes a percentage as an input, we need to precalculate what percentage of the total table size will provide us with the desired number of rows.

# COMMAND ----------

DESIRED_NUM_ROWS = 50

df = spark.sql(
    f"SELECT * FROM {CATALOG_NAME}.{SCHEMA_NAME}.ai4privacy_pii_unmasked_text"
)
df_num_rows = df.count()

desired_rows_pct = round((DESIRED_NUM_ROWS / df_num_rows) * 100, 2)

print(f"Source table num rows: {df_num_rows}")
print(f"Sampling % required to get {DESIRED_NUM_ROWS}: {desired_rows_pct}%")

_ = spark.sql(
    f"""
  CREATE OR REPLACE TABLE {CATALOG_NAME}.{SCHEMA_NAME}.ai4privacy_pii_unmasked_text_sample
  COMMENT "Sample (approx. {DESIRED_NUM_ROWS} rows) of ai4privacy PII English dataset"
  TBLPROPERTIES (
      "sensitivity" = "high",
      "quality" = "raw"
  )
  AS
  SELECT *
  FROM {CATALOG_NAME}.{SCHEMA_NAME}.ai4privacy_pii_unmasked_text
  TABLESAMPLE({desired_rows_pct} PERCENT)
"""
)


# COMMAND ----------

df = spark.sql(
    f"SELECT * FROM {CATALOG_NAME}.{SCHEMA_NAME}.ai4privacy_pii_unmasked_text_sample"
)
display(df)


# COMMAND ----------

# MAGIC %md
# MAGIC # Next step: Select and experiment with a model
# MAGIC
# MAGIC Proceed to our [next Notebook]($./02_Model_Selection)

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
