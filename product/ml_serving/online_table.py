# Databricks notebook source
# MAGIC %pip install databricks-sdk --upgrade
# MAGIC %pip install mlflow>=2.9.0
# MAGIC %pip install --force-reinstall databricks-feature-store 
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog_name = "cjc"
schema_name = "ml_serv"


#spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
import mlflow 
from databricks import feature_engineering
import json
import requests
import time

fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------


