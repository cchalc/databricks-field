# Databricks notebook source
# MAGIC %md # Example of how to read in XLS files
# MAGIC Here we will use the pyspark.pandas dataframe and the `to_delta()` to write out data. We will also show a conversion `to_spark()` and and the write it out as normal.  

# COMMAND ----------

# MAGIC %md The only relevant cluster specifications to reporoduce this demo are:
# MAGIC   ```
# MAGIC     "spark_version": "10.4.x-scala2.12",
# MAGIC     "spark_conf": {
# MAGIC         "spark.sql.execution.arrow.pyspark.fallback.enabled": "true",
# MAGIC         "spark.sql.execution.arrow.pyspark.enabled": "true"
# MAGIC     },
# MAGIC     
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install openpyxl

# COMMAND ----------

import pyspark.pandas as ps
from pyspark.sql.types import *

# COMMAND ----------

db = "christopherchalcraft_scratch"
spark.sql(f"DROP DATABASE IF EXISTS {db} CASCADE") # comment out if you want
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")
spark.sql(f"USE {db}")

# COMMAND ----------

file_location = "/FileStore/tables/christopherchalcraft/data.xlsm"
sheet_name = "Cars"
df = (ps
      .read_excel(
        io=file_location,
        sheet_name=f"{sheet_name}",
        usecols=["Make","Model","Origin","Length"],
        dtype={"Make": str, "Model": str, "Origin": str, "Length": int}
        )
      )

# COMMAND ----------

type(df)

# COMMAND ----------

display(df)

# COMMAND ----------

# remove path of unmanaged table
table_name = "cars"
save_path = '/tmp/christopher.chalcraft'

# table for pyspark.pandas (pspd)
pspd_table_name = "cars_pspd"
pspd_table_path = save_path+f"/{table_name}_pspd"
spark.sql(f"drop table if exists {table_name}_pspd")
dbutils.fs.rm(pspd_table_path, True)

# table for spark df
spark_table_name = "cars_spark"
spark_table_path = save_path+f"/{table_name}_spark"
spark.sql(f"drop table if exists {table_name}_spark")
dbutils.fs.rm(spark_table_path, True)

# COMMAND ----------

# write out the pyspark.pandas dataframe
# https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/api/pyspark.pandas.DataFrame.to_delta.html#pyspark.pandas.DataFrame.to_delta

(df
 .to_delta(
   path=pspd_table_path,
   mode="overwrite"
 )
)

# COMMAND ----------

# MAGIC %fs ls /tmp/christopher.chalcraft/cars_pspd

# COMMAND ----------

# create a table in metastore
spark.sql("CREATE TABLE " + pspd_table_name + " USING DELTA LOCATION '" + pspd_table_path + "'")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from cars_pspd

# COMMAND ----------

# convert to spark df
sdf = (df
      .to_spark()
      )

# COMMAND ----------

display(sdf)

# COMMAND ----------

sdf.printSchema()

# COMMAND ----------

## You can create a managed table with .saveAsTable
# (df
#  .write
#  .format("delta")
#  .mode("overwrite")
#  .saveAsTable(f"{db}.cars")
# )

# COMMAND ----------

(sdf
 .write
 .format("delta")
 .save(spark_table_path)
)
spark.sql("CREATE TABLE " + spark_table_name + " USING DELTA LOCATION '" + spark_table_path + "'")

# COMMAND ----------

# MAGIC %fs ls /tmp/christopher.chalcraft/cars_spark

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from cars_spark
