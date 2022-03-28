# Databricks notebook source
def get_farmers_market_data():
  return (
    spark.read.format('csv').option("header", "true")
      .load('/databricks-datasets/data.gov/farmers_markets_geographic_data/data-001/')
  )

# COMMAND ----------

df = get_farmers_market_data()
display(df)

# COMMAND ----------

# MAGIC %fs ls /Users/christopher.chalcraft@databricks.com/field_demos/dltpoc/tables

# COMMAND ----------

# MAGIC %fs ls /mnt

# COMMAND ----------

# MAGIC %run ../../_resources/setup

# COMMAND ----------

df.write.format("delta").mode("overwrite").option("path", f"{cloud_storage_path}/tables").saveAsTable("farmers_market")
