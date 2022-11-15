# Databricks notebook source
dbutils.fs.head("./test_dbutils.py")

# COMMAND ----------

# MAGIC %fs ls /Users/christopher.chalcraft@databricks.com/sparksubmit/

# COMMAND ----------

filepath = "/Users/christopher.chalcraft@databricks.com/sparksubmit/test_dbutils.py"

# COMMAND ----------

mystring = "print(dbutils.help())"

# COMMAND ----------

dbutils.fs.put(f"{filepath}", mystring)

# COMMAND ----------

import os
os.listdir(f"/dbfs{filepath}")

# COMMAND ----------


