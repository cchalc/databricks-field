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

# File location and type (uploaded manually)
file_location = "/FileStore/tables/christopher_chalcraft/rules.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
rules = (spark.read.format(file_type)
         .option("inferSchema", infer_schema)
         .option("header", first_row_is_header)
         .option("sep", delimiter)
         .load(file_location)
        )

display(rules)

# COMMAND ----------

rules.show()

# COMMAND ----------

assert " " not in ''.join(rules.columns)  

# COMMAND ----------

# MAGIC %run ../../_resources/setup

# COMMAND ----------

df.write.format("delta").mode("overwrite").option("path", f"{cloud_storage_path}/tables/farmers_market").saveAsTable("farmers_market")

# COMMAND ----------


