# Databricks notebook source
# MAGIC %pip install dlt-meta
# MAGIC

# COMMAND ----------

layer = spark.conf.get("layer", None)
from src.dataflow_pipeline import DataflowPipeline
DataflowPipeline.invoke_dlt_pipeline(spark, layer)
