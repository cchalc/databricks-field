# Databricks notebook source
import os
from pyspark.sql import SparkSession
from synapse.ml.core.platform import running_on_synapse, find_secret

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()
if running_on_synapse():
    from notebookutils.visualization import display

cognitive_key = find_secret("cognitive-api-key")
cognitive_location = "eastus"

translator_key = find_secret("translator-key")
translator_location = "eastus"

search_key = find_secret("azure-search-key")
search_service = "mmlspark-azure-search"
search_index = "form-demo-index"

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


def blob_to_url(blob):
    [prefix, postfix] = blob.split("@")
    container = prefix.split("/")[-1]
    split_postfix = postfix.split("/")
    account = split_postfix[0]
    filepath = "/".join(split_postfix[1:])
    return "https://{}/{}/{}".format(account, container, filepath)


df2 = (
    spark.read.format("binaryFile")
    .load("wasbs://ignite2021@mmlsparkdemo.blob.core.windows.net/form_subset/*")
    .select("path")
    .limit(10)
    .select(udf(blob_to_url, StringType())("path").alias("url"))
    .cache()
)

# COMMAND ----------

display(df2)

# COMMAND ----------

# MAGIC %md
# MAGIC <embed src="https://mmlsparkdemo.blob.core.windows.net/ignite2021/form_svgs/Invoice11205.svg" width="40%"/>

# COMMAND ----------

from synapse.ml.cognitive import AnalyzeInvoices

analyzed_df = (
    AnalyzeInvoices()
    .setSubscriptionKey(cognitive_key)
    .setLocation(cognitive_location)
    .setImageUrlCol("url")
    .setOutputCol("invoices")
    .setErrorCol("errors")
    .setConcurrency(5)
    .transform(df2)
    .cache()
)

# COMMAND ----------

display(analyzed_df)

# COMMAND ----------

from synapse.ml.cognitive import FormOntologyLearner

organized_df = (
    FormOntologyLearner()
    .setInputCol("invoices")
    .setOutputCol("extracted")
    .fit(analyzed_df)
    .transform(analyzed_df)
    .select("url", "extracted.*")
    .cache()
)

# COMMAND ----------

display(organized_df)

# COMMAND ----------

from pyspark.sql.functions import explode, col

itemized_df = (
    organized_df.select("*", explode(col("Items")).alias("Item"))
    .drop("Items")
    .select("Item.*", "*")
    .drop("Item")
)

# COMMAND ----------

display(itemized_df)

# COMMAND ----------

display(itemized_df.where(col("ProductCode") == 48))

# COMMAND ----------

from synapse.ml.cognitive import Translate

translated_df = (
    Translate()
    .setSubscriptionKey(translator_key)
    .setLocation(translator_location)
    .setTextCol("Description")
    .setErrorCol("TranslationError")
    .setOutputCol("output")
    .setToLanguage(["zh-Hans", "fr", "ru", "cy"])
    .setConcurrency(5)
    .transform(itemized_df)
    .withColumn("Translations", col("output.translations")[0])
    .drop("output", "TranslationError")
    .cache()
)

# COMMAND ----------

display(translated_df)

# COMMAND ----------

from synapse.ml.cognitive import *
from pyspark.sql.functions import monotonically_increasing_id, lit

(
    translated_df.withColumn("DocID", monotonically_increasing_id().cast("string"))
    .withColumn("SearchAction", lit("upload"))
    .writeToAzureSearch(
        subscriptionKey=search_key,
        actionCol="SearchAction",
        serviceName=search_service,
        indexName=search_index,
        keyCol="DocID",
    )
)

# COMMAND ----------

import requests

url = "https://{}.search.windows.net/indexes/{}/docs/search?api-version=2019-05-06".format(
    search_service, search_index
)
requests.post(url, json={"search": "door"}, headers={"api-key": search_key}).json()

# COMMAND ----------


