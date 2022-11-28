# Databricks notebook source
filepath = "/Users/christopher.chalcraft@databricks.com/sparksubmit/test_dbutils.py"

# COMMAND ----------

mystring = "print(dbutils.help())"

# COMMAND ----------

dbutils.fs.put(f"{filepath}", mystring)

# COMMAND ----------

# MAGIC %fs ls /Users/christopher.chalcraft@databricks.com/sparksubmit

# COMMAND ----------


dbutils.fs.head("dbfs:/Users/christopher.chalcraft@databricks.com/sparksubmit/test_dbutils.py")

# COMMAND ----------

# dbutils.fs.rm("/Users/christopher.chalcraft@databricks.com/sparksubmit", True)

# COMMAND ----------


