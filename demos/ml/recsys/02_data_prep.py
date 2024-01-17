# Databricks notebook source
# MAGIC %run ./01_setup

# COMMAND ----------

# MAGIC %pip install surprise

# COMMAND ----------

# Load the retail dataset from the databricks-datasets
retail_df = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("dbfs:/databricks-datasets/retail-data/by-day/*.csv")

display(retail_df)

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets
