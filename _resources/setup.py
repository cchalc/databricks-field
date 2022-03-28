# Databricks notebook source
# MAGIC %md 
# MAGIC # Set up configuration
# MAGIC ## Libraries

# COMMAND ----------

#%pip install /dbfs/databricks/libraries/pyrasterframes-0.10.1.dev0+dbr7.3-py3-none-any.whl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Path and database configuration

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

project = "wwxf"
dbutils.widgets.text("projectid", project, "project") # name, value, label

# COMMAND ----------

from pyspark.sql.functions import rand, input_file_name, from_json, col
from pyspark.sql.types import *

# COMMAND ----------

# import mlflow
# import mlflow.spark
# from mlflow.utils.file_utils import TempDir

from time import sleep
import re
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

def clean_string(a: str) -> str:
  return re.sub('[^A-Za-z0-9]+', '', a).lower()

# COMMAND ----------

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
username = clean_string(user.partition('@')[0])

print("Created variables:")
print("user: {}".format(user))
print("username: {}".format(username))

# user settings
user_dbName = re.sub(r'\W+', '_', username) + "_" + project
user_path = f"/Users/{user}/{project}"
dbutils.widgets.text("user_dbName", user_dbName, "user_dbName")
dbutils.widgets.text("user_path", user_path, "user_path")
print(f"user path (user path): {user_path}")
print("dbName (using database): {}".format(user_dbName))
spark.sql("""create database if not exists {} LOCATION '{}/{}/tables' """.format(user_dbName, project, user_path))
spark.sql("""USE {}""".format(user_dbName))

print(f"By default using user database {user_dbName}. Please switch to project database if needed")


# project settings -- temporary
dbName = project
bronze_path = f"/mnt/bronze"
silver_path = f"/mnt/silver"
gold_path = f"/mnt/gold"
project_path = f"/{project}" # just temporary until we decide

dbutils.widgets.text("dbName", dbName, "dbName")
dbutils.widgets.text("project_path", project_path, "project_path")
print(f"project path (project path): {project_path}")

spark.sql("""create database if not exists {} LOCATION '{}/tables' """.format(dbName, project))
# spark.sql("""USE {}""".format(user_dbName))
# print("dbName (using database): {}".format(dbName))
