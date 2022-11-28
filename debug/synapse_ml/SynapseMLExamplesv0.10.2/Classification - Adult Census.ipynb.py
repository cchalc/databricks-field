# Databricks notebook source
# MAGIC %md
# MAGIC ## Classification - Adult Census
# MAGIC 
# MAGIC In this example, we try to predict incomes from the *Adult Census* dataset.
# MAGIC 
# MAGIC First, we import the packages (use `help(synapse)` to view contents),

# COMMAND ----------

from pyspark.sql import SparkSession

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's read the data and split it to train and test sets:

# COMMAND ----------

data = spark.read.parquet(
    "wasbs://publicwasb@mmlspark.blob.core.windows.net/AdultCensusIncome.parquet"
)
data = data.select(["education", "marital-status", "hours-per-week", "income"])
train, test = data.randomSplit([0.75, 0.25], seed=123)
train.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC `TrainClassifier` can be used to initialize and fit a model, it wraps SparkML classifiers.
# MAGIC You can use `help(synapse.ml.train.TrainClassifier)` to view the different parameters.
# MAGIC 
# MAGIC Note that it implicitly converts the data into the format expected by the algorithm: tokenize
# MAGIC and hash strings, one-hot encodes categorical variables, assembles the features into a vector
# MAGIC and so on.  The parameter `numFeatures` controls the number of hashed features.

# COMMAND ----------

from synapse.ml.train import TrainClassifier
from pyspark.ml.classification import LogisticRegression

model = TrainClassifier(
    model=LogisticRegression(), labelCol="income", numFeatures=256
).fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we save the model so it can be used in a scoring program.

# COMMAND ----------

from synapse.ml.core.platform import *

if running_on_synapse():
    model.write().overwrite().save(
        "abfss://synapse@mmlsparkeuap.dfs.core.windows.net/models/AdultCensus.mml"
    )
elif running_on_synapse_internal():
    model.write().overwrite().save("Files/models/AdultCensus.mml")
elif running_on_databricks():
    model.write().overwrite().save("dbfs:/AdultCensus.mml")
elif running_on_binder():
    model.write().overwrite().save("/tmp/AdultCensus.mml")
else:
    print(f"{current_platform()} platform not supported")
