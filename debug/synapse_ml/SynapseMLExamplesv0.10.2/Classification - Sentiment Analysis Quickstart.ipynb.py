# Databricks notebook source
# MAGIC %md
# MAGIC # A 5 minute tour of SynapseML

# COMMAND ----------

from pyspark.sql import SparkSession
from synapse.ml.core.platform import *

spark = SparkSession.builder.getOrCreate()

from synapse.ml.core.platform import materializing_display as display

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1: Load our Dataset

# COMMAND ----------

train, test = (
    spark.read.parquet(
        "wasbs://publicwasb@mmlspark.blob.core.windows.net/BookReviewsFromAmazon10K.parquet"
    )
    .limit(1000)
    .cache()
    .randomSplit([0.8, 0.2])
)

display(train)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2: Make our Model

# COMMAND ----------

from pyspark.ml import Pipeline
from synapse.ml.featurize.text import TextFeaturizer
from synapse.ml.lightgbm import LightGBMRegressor

model = Pipeline(
    stages=[
        TextFeaturizer(inputCol="text", outputCol="features"),
        LightGBMRegressor(featuresCol="features", labelCol="rating"),
    ]
).fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3: Predict!

# COMMAND ----------

display(model.transform(test))

# COMMAND ----------

# MAGIC %md
# MAGIC # Alternate route: Let the Cognitive Services handle it

# COMMAND ----------

from synapse.ml.cognitive import TextSentiment
from synapse.ml.core.platform import find_secret

model = TextSentiment(
    textCol="text",
    outputCol="sentiment",
    subscriptionKey=find_secret("cognitive-api-key"),
).setLocation("eastus")

display(model.transform(test))
