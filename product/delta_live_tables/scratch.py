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

rules.filter(col("tag").contains("validity"))

# COMMAND ----------

for row in rules.filter(col("tag") == "validity").collect():
  print(row)

# COMMAND ----------

# dbutils.fs.rm(cloud_storage_path, True)

# COMMAND ----------

# MAGIC %run ../../_resources/setup

# COMMAND ----------

def get_rules(tag):
  """
    loads data quality rules from csv file
    :param tag: tag to match
    :return: dictionary of rules that matched the tag
  """
  rules = {}
  df = spark.read.format("csv").option("header", "true").load("/FileStore/tables/christopher_chalcraft/rules.csv")
  for row in df.filter(col("tag") == tag).collect():
    rules[row['name']] = row['constraint']
  return rules

# COMMAND ----------

expectation_rules = get_rules("validity")

# COMMAND ----------

(rules
 .coalesce(1)
 .write
 .format("csv")
 .mode("overwrite")
 .option("header", True)
 .option("path", f"{cloud_storage_path}/rules")
 .saveAsTable("rules")
)

# COMMAND ----------

# MAGIC %fs ls /Users/christopher.chalcraft@databricks.com/field_demos/dltpoc/tables

# COMMAND ----------

df.write.format("delta").mode("overwrite").option("path", f"{cloud_storage_path}/tables/farmers_marke").saveAsTable("farmers_market")

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/christopher_chalcraft
