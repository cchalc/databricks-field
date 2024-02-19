# Databricks notebook source
# MAGIC %md
# MAGIC # Loading Llama 2 7B Chat model from Marketplace
# MAGIC
# MAGIC This example notebook demonstrates how to load the Llama 2 7B Chat model from a Databricks Marketplace's Catalog ([see announcement blog](https://www.databricks.com/blog/llama-2-foundation-models-available-databricks-lakehouse-ai)).
# MAGIC
# MAGIC Environment:
# MAGIC - MLR: 13.3 ML
# MAGIC - Instance: `g5.xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure

# COMMAND ----------

# To access models in Unity Catalog, ensure that MLflow is up to date
%pip install --upgrade "mlflow-skinny[databricks]>=2.4.1"
dbutils.library.restartPython()

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

catalog_name = "cjc_models" # catalog name when installing the model from Databricks Marketplace
version = 1

# Create a Spark UDF to generate the response to a prompt
generate = mlflow.pyfunc.spark_udf(
    spark, f"models:/{catalog_name}.models.llama_2_7b_chat_hf/{version}", "string"
)

# COMMAND ----------

# MAGIC %md
# MAGIC The Spark UDF `generate` could inference on Spark DataFrames.

# COMMAND ----------

spark.sql("CREATE CATALOG IF NOT EXISTS cjc")
spark.sql("CREATE SCHEMA IF NOT EXISTS cjc.ml_serv")
spark.sql("CREATE VOLUME IF NOT EXISTS cjc.ml_serv.myc")
spark.sql("use cjc.ml_serv")

# COMMAND ----------

# Storage Params
STORAGE_PATH = "/Volumes/cjc/ml_serv/myc"

# COMMAND ----------

import pandas as pd

pii_sentences = [
    "Alice Smith's email is alice.smith@email.com and lives at 123 Maple St, Springfield.",
    "Bob Johnson's phone number is 555-1234, residing at 456 Pine Lane, Lakeside.",
    "Carol White mentioned her SSN is 123-45-6789, currently at 789 Oak Ave, Rivertown.",
    "David Brown's license plate is ABC1234 and his address is 101 Birch Rd, Hilltown.",
    "Eve Davis shared her passport number, G12345678, while living at 202 Cedar St, Coastcity.",
    "Frank Moore's credit card number is 1234 5678 9012 3456, billing to 303 Elm St, Greentown.",
    "Grace Lee's driver's license is L123-4567-8901, with a domicile at 404 Aspen Way, Frostville.",
    "Henry Wilson's bank account number is 123456789, banking at 505 Walnut St, Sunnyvale.",
    "Ivy Young disclosed her birthdate, 01/02/1990, alongside her residence at 606 Pinecone Rd, Raincity.",
    "Jack Taylor's employee ID is 7890, working at 707 Redwood Blvd, Cloudtown.",
    "Kathy Green's insurance policy is INS-123456, covered at 808 Maple Grove, Windyville.",
    "Leo Carter mentioned his membership number, MEM-789123, frequenting 909 Cherry Lane, Starcity.",
    "Mia Ward's patient ID is PAT-456789, consulting at 1010 Willow Path, Moonville.",
    "Nathan Ellis's booking reference is REF1234567, staying at 1111 Ivy Green Rd, Sunnyside.",
    "Olivia Sanchez's pet's name is Whiskers, living together at 1212 Magnolia St, Petville.",
    "Peter Gomez's library card number is 1234567, a patron of 1313 Lilac Lane, Booktown.",
    "Quinn Torres is registered under the voter ID VOT-7890123, residing at 1414 Oakdale St, Voteville.",
    "Rachel Kim mentioned her alumni number, ALU-123789, belonging to 1515 Pine St, Gradtown.",
    "Steve Adams's gym membership is GYM-456123, exercising at 1616 Fir Ave, Muscleville.",
    "Tina Nguyen's loyalty card is LOY-789456, shopping at 1717 Spruce Way, Marketcity."
]

# # Creating the DataFrame
# df = pd.DataFrame({
#     "text": pii_sentences
# })

# df


# COMMAND ----------

import pandas as pd

# To have more than 1 input sequences in the same batch for inference, more GPU memory would be needed; swap to more powerful GPUs (e.g. Standard_NC24ads_A100_v4 on Azure), or use Databricks Model Serving

df = spark.createDataFrame(
    pd.DataFrame(
        {
            "text": pii_sentences
        }
    )
)
display(df)


# COMMAND ----------

from pyspark.sql.functions import concat, lit

redact_instruction = "Given the following text, identify and replace any Personally Identifiable Information (PII) such as names, email addresses, phone numbers, physical addresses, and any other specific details that can identify an individual, with [REDACTED]. Ensure that the modified text maintains its original meaning as much as possible without revealing any PII: "

df = df.withColumn("text", concat(lit(redact_instruction), df["text"]))
display(df)

# COMMAND ----------


generated_df = df.select(generate(df.text).alias("redacted_text"))
display(generated_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We could also wrap the Spark UDF into a function that takes system prompts, and takes lists of text strings as input/output.

# COMMAND ----------

DEFAULT_SYSTEM_PROMPT = """\
You are only a bot in charge of redacting personally identifiable information (PII) such as names, addresses, phone numbers, social security numbers, etc., You only only take an instruction an repeat it back replacing the PII with [REDACTED] in the response. An example would be: Jimmy John lives in White Rock at 1234 Beach St and loves to play pickleball. Call for a game at 789-555-1234. This would be replaced with: [REDACTED] lives in [REDACTED] at [REDACTED] and loves to play pickleball. Call for a game at [REDACTED].
"""

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

PROMPT_FOR_GENERATION_FORMAT = """
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>

Repeat the following text back with the PII redacted: {instruction}
[/INST]
""".format(
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    instruction="{instruction}"
)

# COMMAND ----------

from typing import List
import pandas as pd


def gen_text(instructions: List[str]):
    prompts = [
        PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)
        for instruction in instructions
    ]

    # `generate` is a Spark UDF that takes a string column as input
    df = spark.createDataFrame(pd.DataFrame({"text": pd.Series(prompts)}))
    generated_df = df.select(generate(df.text).alias("generated_text"))

    # Get the rows of the 'generated_text' column in the dataframe 'generated_df' as a list, and truncate the instruction
    generated_text_list = [
        str(row.generated_text).split("[/INST]\n")[1] for row in generated_df.collect()
    ]

    return generated_text_list

# COMMAND ----------

# To have more than 1 input sequences in the same batch for inference, more GPU memory would be needed; swap to more powerful GPUs (e.g. Standard_NC24ads_A100_v4 on Azure), or use Databricks Model Serving
gen_text(
    [
        "Alice Smith's email is alice.smith@email.com and lives at 123 Maple St, Springfield.",
    ]
)
