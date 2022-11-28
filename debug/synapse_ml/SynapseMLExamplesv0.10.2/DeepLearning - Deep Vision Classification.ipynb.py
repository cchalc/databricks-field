# Databricks notebook source
# MAGIC %md
# MAGIC ## Deep Learning - Deep Vision Classifier

# COMMAND ----------

# MAGIC %md
# MAGIC ### Environment Setup -- reinstall horovod based on new version of pytorch

# COMMAND ----------

# install cloudpickle 2.0.0 to add synapse module for usage of horovod
%pip install cloudpickle==2.0.0 --force-reinstall --no-deps

# COMMAND ----------

import synapse
import cloudpickle

cloudpickle.register_pickle_by_value(synapse)

# COMMAND ----------

! horovodrun --check-build

# COMMAND ----------

from pyspark.sql.functions import udf, col, regexp_replace
from pyspark.sql.types import IntegerType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read Dataset

# COMMAND ----------

def assign_label(path):
    num = int(path.split("/")[-1].split(".")[0].split("_")[1])
    return num // 81


assign_label_udf = udf(assign_label, IntegerType())

# COMMAND ----------

# These files are already uploaded for build test machine
train_df = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.jpg")
    .load("/tmp/17flowers/train")
    .withColumn("image", regexp_replace("path", "dbfs:", "/dbfs"))
    .withColumn("label", assign_label_udf(col("path")))
    .select("image", "label")
)

display(train_df.limit(100))

# COMMAND ----------

test_df = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.jpg")
    .load("/tmp/17flowers/test")
    .withColumn("image", regexp_replace("path", "dbfs:", "/dbfs"))
    .withColumn("label", assign_label_udf(col("path")))
    .select("image", "label")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training

# COMMAND ----------

from horovod.spark.common.store import DBFSLocalStore
from pytorch_lightning.callbacks import ModelCheckpoint
from synapse.ml.dl import *

run_output_dir = "/dbfs/FileStore/test/resnet50"
store = DBFSLocalStore(run_output_dir)

epochs = 10

callbacks = [ModelCheckpoint(filename="{epoch}-{train_loss:.2f}")]

# COMMAND ----------

deep_vision_classifier = DeepVisionClassifier(
    backbone="resnet50",
    store=store,
    callbacks=callbacks,
    num_classes=17,
    batch_size=16,
    epochs=epochs,
    validation=0.1,
)

deep_vision_model = deep_vision_classifier.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction

# COMMAND ----------

pred_df = deep_vision_model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(
    predictionCol="prediction", labelCol="label", metricName="accuracy"
)
print("Test accuracy:", evaluator.evaluate(pred_df))
