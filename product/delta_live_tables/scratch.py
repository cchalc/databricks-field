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

# checking the quarantine notebook

rules = {}
quarantine_rules = {}

rules["valid_website"] = "(Website IS NOT NULL)"
rules["valid_location"] = "(Location IS NOT NULL)"

# concatenate inverse rules
quarantine_rules["invalid_record"] = "NOT({0})".format(" AND ".join(rules.values()))

# COMMAND ----------

rules.values()

# COMMAND ----------

quarantine_rules

# COMMAND ----------

" AND ".join(rules.values())

# COMMAND ----------

def get_raw_fire_department():
  return (
    spark.read.format('csv')
      .option('header', 'true')
      .option('multiline', 'true')
      .load('/databricks-datasets/timeseries/Fires/Fire_Department_Calls_for_Service.csv')
      .withColumnRenamed('Call Type', 'call_type')
      .withColumnRenamed('Received DtTm', 'received')
      .withColumnRenamed('Response DtTm', 'responded')
      .withColumnRenamed('Neighborhooods - Analysis Boundaries', 'neighborhood')
    .select('call_type', 'received', 'responded', 'neighborhood')
  )

# COMMAND ----------

firedep = get_raw_fire_department()
display(firedep)

# COMMAND ----------

firedep.write.format("delta").mode("overwrite").option("path", f"{cloud_storage_path}/tables/tmp_firedepartment").saveAsTable("tmp_firedepartment")

# COMMAND ----------

all_tables = []

def generate_tables(call_table, response_table, filter):

  def create_call_table():
    return (
      spark.sql("""
        SELECT
          unix_timestamp(received,'M/d/yyyy h:m:s a') as ts_received,
          unix_timestamp(responded,'M/d/yyyy h:m:s a') as ts_responded,
          neighborhood
        FROM christopher_chalcraft_dltpoc.tmp_firedepartment
        WHERE call_type = '{filter}'
      """.format(filter=filter))
    )

  def create_response_table():
    return (
      spark.sql("""
        SELECT
          neighborhood,
          AVG((ts_received - ts_responded)) as response_time
        FROM christopher_chalcraft_dltpoc.{call_table}
        GROUP BY 1
        ORDER BY response_time
        LIMIT 10
      """.format(call_table=call_table))
    )
    
  ct = create_call_table()
  rt = create_response_table()
    
  return ct, rt

  all_tables.append(response_table)



# COMMAND ----------

# generate_tables("alarms_table", "alarms_response", "Alarms")
# generate_tables("fire_table", "fire_response", "Structure Fire")
# generate_tables("medical_table", "medical_response", "Medical Incident")

# COMMAND ----------

# ct, rt = generate_tables("alarms_table", "alarms_response", "Alarms")

# COMMAND ----------

print(all_tables)

# COMMAND ----------

tfd = spark.sql("""
SELECT
  unix_timestamp(received, 'M/d/yyyy h:m:s a') as ts_received,
  unix_timestamp(responded, 'M/d/yyyy h:m:s a') as ts_responded,
  neighborhood
FROM
  christopher_chalcraft_dltpoc.tmp_firedepartment
WHERE
  call_type = 'Medical Incident'
"""
               )

# COMMAND ----------

display(tfd)
