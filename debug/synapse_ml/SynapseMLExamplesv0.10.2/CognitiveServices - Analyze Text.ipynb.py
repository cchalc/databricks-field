# Databricks notebook source
# MAGIC %md
# MAGIC # Cognitive Services - Analyze Text

# COMMAND ----------

import os
from pyspark.sql import SparkSession
from synapse.ml.core.platform import running_on_synapse, find_secret

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()
if running_on_synapse():
    from notebookutils.visualization import display

cognitive_key = find_secret("cognitive-api-key")
cognitive_location = "eastus"

# COMMAND ----------

df = spark.createDataFrame(
    data=[
        ["en", "Hello Seattle"],
        ["en", "There once was a dog who lived in London and thought she was a human"],
    ],
    schema=["language", "text"],
)

# COMMAND ----------

display(df)

# COMMAND ----------

from synapse.ml.cognitive import *

text_analyze = (
    TextAnalyze()
    .setLocation(cognitive_location)
    .setSubscriptionKey(cognitive_key)
    .setTextCol("text")
    .setOutputCol("textAnalysis")
    .setErrorCol("error")
    .setLanguageCol("language")
    .setEntityRecognitionParams(
        {"model-version": "latest"}
    )  # Can pass parameters to each model individually
    .setIncludePii(False)  # Users can manually exclude tasks to speed up analysis
    .setIncludeEntityLinking(False)
    .setIncludeSentimentAnalysis(False)
)

df_results = text_analyze.transform(df)

# COMMAND ----------

display(df_results)
