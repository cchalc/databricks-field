# Databricks notebook source
# DBTITLE 1,Constants
# If you want to run this notebook yourself, you need to create a Databricks personal access token,
# store it using our secrets API, and pass it in through the Spark config, such as this:
# spark.pat_token {{secrets/query_history_etl/user}}, or Azure Keyvault.

WORKSPACE_HOST = 'https://eastus2.azuredatabricks.net'
ENDPOINTS_URL = "{0}/api/2.0/sql/endpoints".format(WORKSPACE_HOST)
QUERIES_URL = "{0}/api/2.0/sql/history/queries".format(WORKSPACE_HOST)

MAX_RESULTS_PER_PAGE = 25000
MAX_PAGES_PER_RUN = 500

# We will fetch all queries that were started between this number of hours ago, and now()
# Queries that are running for longer than this will not be updated.
# Can be set to a much higher number when backfilling data, for example when this Job didn't
# run for a while.
NUM_HOURS_TO_UPDATE = 3

DATABASE_NAME = "query_history_etl"
ENDPOINTS_TABLE_NAME = "endpoints"
QUERIES_TABLE_NAME = "queries"

#Databricks secrets API
#auth_header = {"Authorization" : "Bearer " + spark.conf.get("spark.pat_token")}
#Azure KeyVault
auth_header = {"Authorization" : "Bearer " + dbutils.secrets.get(scope = "<scope-name>", key = "<key-name>")}

# COMMAND ----------

# DBTITLE 1,Imports
import requests
from datetime import date, datetime, timedelta
from pyspark.sql.functions import from_unixtime, lit, json_tuple
from delta.tables import *
import time

# COMMAND ----------

# DBTITLE 1,Functions definition
def check_table_exist(db_tbl_name):
    table_exist = False
    try:
        spark.read.table(db_tbl_name) # Check if spark can read the table
        table_exist = True        
    except:
        pass
    return table_exist
  
def current_time_in_millis():
    return round(time.time() * 1000)
  
def get_boolean_keys(arrays):
  # A quirk in Python's and Spark's handling of JSON booleans requires us to converting True and False to true and false
  boolean_keys_to_convert = []
  for array in arrays:
    for key in array.keys():
      if type(array[key]) is bool:
        boolean_keys_to_convert.append(key)
  #print(boolean_keys_to_convert)
  return boolean_keys_to_convert

# COMMAND ----------

# DBTITLE 1,Initialization
notebook_start_execution_time = current_time_in_millis()

spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(DATABASE_NAME))

# COMMAND ----------

# DBTITLE 1,Fetch from SQL Endpoint API
response = requests.get(ENDPOINTS_URL, headers=auth_header)

if response.status_code != 200:
  raise Exception(response.text)
response_json = response.json()

endpoints_json = response_json["endpoints"]

# A quirk in Python's and Spark's handling of JSON booleans requires us to converting True and False to true and false
boolean_keys_to_convert = set(get_boolean_keys(endpoints_json))

for endpoint_json in endpoints_json:
  for key in boolean_keys_to_convert:
    endpoint_json[key] = str(endpoint_json[key]).lower()

endpoints = spark.read.json(sc.parallelize(endpoints_json))
display(endpoints)

endpoints.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable(DATABASE_NAME + "." + ENDPOINTS_TABLE_NAME)

# COMMAND ----------

# DBTITLE 1,Fetch from Query History API
start_date = datetime.now() - timedelta(hours=NUM_HOURS_TO_UPDATE)
start_time_ms = start_date.timestamp() * 1000
end_time_ms = datetime.now().timestamp() * 1000

next_page_token = None
has_next_page = True
pages_fetched = 0

while (has_next_page and pages_fetched < MAX_PAGES_PER_RUN):
  print("Starting to fetch page " + str(pages_fetched))
  pages_fetched += 1
  if next_page_token:
    # Can not set filters after the first page
    request_parameters = {
      "max_results": MAX_RESULTS_PER_PAGE,
      "page_token": next_page_token
    }
  else:
    request_parameters = {
      "max_results": MAX_RESULTS_PER_PAGE,
      "filter_by": {"query_start_time_range": {"start_time_ms": start_time_ms, "end_time_ms": end_time_ms}}
    }

  print ("Request parameters: " + str(request_parameters))
  
  response = requests.get(QUERIES_URL, headers=auth_header, json=request_parameters)
  if response.status_code != 200:
    raise Exception(response.text)
  response_json = response.json()
  next_page_token = response_json["next_page_token"]
  has_next_page = response_json["has_next_page"]
  
  boolean_keys_to_convert = set(get_boolean_keys(response_json["res"]))
  for array_to_process in response_json["res"]:
    for key in boolean_keys_to_convert:
      array_to_process[key] = str(array_to_process[key]).lower()
    
  query_results = spark.read.json(sc.parallelize(response_json["res"]))
  
  # For querying convience, add columns with the time in seconds instead of milliseconds
  query_results_clean = query_results \
    .withColumn("query_start_time", from_unixtime(query_results.query_start_time_ms / 1000)) \
    .withColumn("query_end_time", from_unixtime(query_results.query_end_time_ms / 1000))

  # The error_message column is not present in the REST API response when none of the queries failed.
  # In that case we add it as an empty column, since otherwise the Delta merge would fail in schema
  # validation
  if "error_message" not in query_results_clean.columns:
    query_results_clean = query_results_clean.withColumn("error_message", lit(""))
  
  if not check_table_exist(db_tbl_name="{0}.{1}".format(DATABASE_NAME, QUERIES_TABLE_NAME)):
    # TODO: Probably makes sense to partition and/or Z-ORDER this table.
    query_results_clean.write.format("delta").saveAsTable("{0}.{1}".format(DATABASE_NAME, QUERIES_TABLE_NAME)) 
  else:
    # Merge this page of results into the Delta table. Existing records that match on query_id have
    # all their fields updated (needed because the status, end time, and error may change), and new
    # records are inserted.
    queries_table = DeltaTable.forName(spark, "{0}.{1}".format(DATABASE_NAME, QUERIES_TABLE_NAME))
    queries_table.alias("queryResults").merge(
        query_results_clean.alias("newQueryResults"),
        "queryResults.query_id = newQueryResults.query_id") \
      .whenMatchedUpdateAll() \
      .whenNotMatchedInsertAll() \
      .execute()
    # TODO: Add more merge conditions to make it more efficient.

# COMMAND ----------

print("Time to execute: {}s".format((current_time_in_millis() - notebook_start_execution_time) / 1000))
