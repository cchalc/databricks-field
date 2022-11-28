# Databricks notebook source
# MAGIC %md
# MAGIC ## Regression - Flight Delays
# MAGIC 
# MAGIC In this example, we run a linear regression on the *Flight Delay* dataset to predict the delay times.
# MAGIC 
# MAGIC We demonstrate how to use the `TrainRegressor` and the `ComputePerInstanceStatistics` APIs.
# MAGIC 
# MAGIC First, import the packages.

# COMMAND ----------

from pyspark.sql import SparkSession

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

import numpy as np
import pandas as pd
import synapse.ml

# COMMAND ----------

# MAGIC %md
# MAGIC Next, import the CSV dataset.

# COMMAND ----------

flightDelay = spark.read.parquet(
    "wasbs://publicwasb@mmlspark.blob.core.windows.net/On_Time_Performance_2012_9.parquet"
)
# print some basic info
print("records read: " + str(flightDelay.count()))
print("Schema: ")
flightDelay.printSchema()
flightDelay.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Split the dataset into train and test sets.

# COMMAND ----------

train, test = flightDelay.randomSplit([0.75, 0.25])

# COMMAND ----------

# MAGIC %md
# MAGIC Train a regressor on dataset with `l-bfgs`.

# COMMAND ----------

from synapse.ml.train import TrainRegressor, TrainedRegressorModel
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer

# Convert columns to categorical
catCols = ["Carrier", "DepTimeBlk", "ArrTimeBlk"]
trainCat = train
testCat = test
for catCol in catCols:
    simodel = StringIndexer(inputCol=catCol, outputCol=catCol + "Tmp").fit(train)
    trainCat = (
        simodel.transform(trainCat)
        .drop(catCol)
        .withColumnRenamed(catCol + "Tmp", catCol)
    )
    testCat = (
        simodel.transform(testCat)
        .drop(catCol)
        .withColumnRenamed(catCol + "Tmp", catCol)
    )
lr = LinearRegression().setRegParam(0.1).setElasticNetParam(0.3)
model = TrainRegressor(model=lr, labelCol="ArrDelay").fit(trainCat)

# COMMAND ----------

# MAGIC %md
# MAGIC Save, load, or Score the regressor on the test data.

# COMMAND ----------

from synapse.ml.core.platform import *

if running_on_synapse():
    model_name = "/models/flightDelayModel.mml"
elif running_on_synapse_internal():
    model_name = "Files/models/flightDelayModel.mml"
elif running_on_databricks():
    model_name = "dbfs:/flightDelayModel.mml"
else:
    model_name = "/tmp/flightDelayModel.mml"

model.write().overwrite().save(model_name)
flightDelayModel = TrainedRegressorModel.load(model_name)

scoredData = flightDelayModel.transform(testCat)
scoredData.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Compute model metrics against the entire scored dataset

# COMMAND ----------

from synapse.ml.train import ComputeModelStatistics

metrics = ComputeModelStatistics().transform(scoredData)
metrics.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, compute and show per-instance statistics, demonstrating the usage
# MAGIC of `ComputePerInstanceStatistics`.

# COMMAND ----------

from synapse.ml.train import ComputePerInstanceStatistics

evalPerInstance = ComputePerInstanceStatistics().transform(scoredData)
evalPerInstance.select("ArrDelay", "prediction", "L1_loss", "L2_loss").limit(
    10
).toPandas()
