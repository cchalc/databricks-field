# Databricks notebook source
# MAGIC %md # Feature Serving example notebook
# MAGIC
# MAGIC Feature Serving lets you serve pre-materialized features and run on-demand computation for features. 
# MAGIC
# MAGIC This notebook illustrates how to:
# MAGIC 1. Create a `FeatureSpec`. A `FeatureSpec` defines a set of features (prematerialized and on-demand) that are served together. 
# MAGIC 2. Create an `Online Table` from a Delta Table.
# MAGIC 2. Serve the features. To serve features, you create a Feature Serving endpoint with the `FeatureSpec`.
# MAGIC
# MAGIC ### Requirements
# MAGIC This notebook requires Databricks Runtime for Machine Learning 14.2 or above.

# COMMAND ----------

# MAGIC %md ## Set up the Feature Table

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC %pip install mlflow>=2.9.0
# MAGIC %pip install --force-reinstall databricks-feature-store 
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Specify the catalog and schema to use. You must have USE_CATALOG privilege on the catalog and USE_SCHEMA and CREATE_TABLE privileges on the schema.
# Change the catalog and schema here if necessary.

catalog_name = "cjc"
schema_name = "feature_serving"


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

feature_table_name = f"{catalog_name}.{schema_name}.location_features"
online_table_name=f"{catalog_name}.{schema_name}.location_features_online"
function_name = f"{catalog_name}.{schema_name}.distance"

# COMMAND ----------

# MAGIC %md ### To access the feature table from Feature Serving, you must create an Online Table from the feature table.  
# MAGIC Feature table is used for offline training of models, and online table is used in online inference

# COMMAND ----------

# Read in the dataset
destination_location_df = spark.read.option("inferSchema", "true").load("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_destination-locations/", format="csv", header="true")

# Create the feature table
fe.create_table(
  name = feature_table_name,
  primary_keys="destination_id",
  df = destination_location_df,
  description = "Destination location features."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up a Databricks Online Table
# MAGIC
# MAGIC You can create an online table from the Catalog Explorer UI, Databricks SDK or Rest API. The steps to use Databricks python SDK are described below. For more details, see the Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#create)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/online-tables#create)). For information about required permissions, see Permissions ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#user-permissions)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/online-tables#user-permissions)).

# COMMAND ----------

# DBTITLE 1,Databricks Online Table Creation
from pprint import pprint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *
import mlflow

workspace = WorkspaceClient()

# Create an online table
spec = OnlineTableSpec(
  primary_key_columns = ["destination_id"],
  source_table_full_name = feature_table_name,
  run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'}),
  perform_full_copy=True)

try:
  online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

pprint(workspace.online_tables.get(online_table_name))

# COMMAND ----------

# MAGIC %md ## Create the function

# COMMAND ----------

# MAGIC %md The next cell defines a function that calculates the distance between the destination and the user's current location.

# COMMAND ----------

# DBTITLE 1,Haversine Distance Calculator Function
# Define the function. This function calculates the distance between two locations. 
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(latitude DOUBLE, longitude DOUBLE, user_latitude DOUBLE, user_longitude DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON AS
$$
import math
lat1 = math.radians(latitude)
lon1 = math.radians(longitude)
lat2 = math.radians(user_latitude)
lon2 = math.radians(user_longitude)

# Earth's radius in kilometers
radius = 6371

# Haversine formula
dlat = lat2 - lat1
dlon = lon2 - lon1
a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
distance = radius * c

return distance
$$""")

# COMMAND ----------

# DBTITLE 1,Databricks Travel Feature Spec Generator
from databricks.feature_engineering import FeatureLookup, FeatureFunction

features=[
  FeatureLookup(
    table_name=feature_table_name,
    lookup_key="destination_id"
  ),
  FeatureFunction(
    udf_name=function_name, 
    output_name="distance",
    input_bindings={
      "latitude": "latitude", 
      "longitude": "longitude", 
      "user_latitude": "user_latitude", 
      "user_longitude": "user_longitude"
    },
  ),
]

feature_spec_name = f"{catalog_name}.{schema_name}.travel_spec"
try: 
  fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# COMMAND ----------

# MAGIC %md ## Create a Feature Serving endpoint

# COMMAND ----------

# DBTITLE 1,Databricks Endpoint Creation Script
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# Create endpoint
endpoint_name = "cjc-location"

try:
  status = workspace.serving_endpoints.create_and_wait(
  name=endpoint_name,
  config = EndpointCoreConfigInput(
    served_entities=[
    ServedEntityInput(
        entity_name=feature_spec_name,
        scale_to_zero_enabled=True,
        workload_size="Small"
    )
    ]
  )
  )
  print(status)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# COMMAND ----------

# Get the status of the endpoint
status = workspace.serving_endpoints.get(name=endpoint_name)
print(status)

# COMMAND ----------

# MAGIC %md You can now view the status of the Feature Serving Endpoint in the table on the **Serving endpoints** page. Click **Serving** in the sidebar to display the page.

# COMMAND ----------

# MAGIC %md ## Query

# COMMAND ----------

# DBTITLE 1,MLflow Databricks Prediction Client
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
response = client.predict(
    endpoint=endpoint_name,
    inputs={
        "dataframe_records": [
            {"destination_id": 1, "user_latitude": 37, "user_longitude": -122},
            {"destination_id": 2, "user_latitude": 37, "user_longitude": -122},
        ]
    },
)

pprint(response)

# COMMAND ----------

# MAGIC %md ## Clean up
# MAGIC
# MAGIC When you are finished, delete the `FeatureSpec`, feature endpoint, and online table. 
# MAGIC
# MAGIC The online table can only be deleted from Catalog Explorer, as follows:   
# MAGIC
# MAGIC 1. In the left sidebar, click **Catalog**.   
# MAGIC 2. Navigate to the online table.  
# MAGIC 3. From the kebab menu, select **Delete**.   
# MAGIC
# MAGIC Run the following commands to delete the `FeatureSpec` and feature endpoint.

# COMMAND ----------

# fe.delete_feature_spec(name=feature_spec_name)

# COMMAND ----------

# workspace.serving_endpoints.delete(name=endpoint_name)

# COMMAND ----------

# workspace.online_tables.delete(name=online_table_name)
