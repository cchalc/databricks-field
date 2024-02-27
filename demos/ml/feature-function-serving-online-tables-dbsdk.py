# Databricks notebook source
# MAGIC %md # Feature and Function Serving Example Notebook
# MAGIC
# MAGIC Feature Serving lets you serve pre-materialized features and run on-demand computation for features. 
# MAGIC
# MAGIC This notebook illustrates how to:
# MAGIC 1. Create a `FeatureSpec`. A `FeatureSpec` defines a set of features (prematerialized and on-demand) that are served together. 
# MAGIC 2. Serve the features. To serve features, you create a Feature and Function Serving endpoint with the `FeatureSpec`.
# MAGIC
# MAGIC ### Requirements
# MAGIC This notebook requires Databricks Runtime for Machine Learning 14.2 or above.

# COMMAND ----------

# MAGIC %md ## Set up the Feature Table

# COMMAND ----------

# MAGIC %pip install -U databricks-sdk>=0.18.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Specify the catalog and schema to use. You must have USE_CATALOG privilege on the catalog and USE_SCHEMA and CREATE_TABLE privileges on the schema.
# Change the catalog and schema here if necessary.

catalog_name = "cjc"
schema_name = "default"

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
feature_table_name = f"{catalog_name}.{schema_name}.location_features"
function_name = f"{catalog_name}.{schema_name}.distance"

# COMMAND ----------

# MAGIC %md To access the feature table from Feature and Function Serving, you must publish the table.  
# MAGIC This notebook uses Databricks Online Tables to publish the table.

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
# MAGIC You create an online table from the Catalog Explorer. The steps are described below. For more details, see the Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#create)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/online-tables#create)). For information about required permissions, see Permissions ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#user-permissions)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/online-tables#user-permissions)).
# MAGIC
# MAGIC
# MAGIC In Catalog Explorer, navigate to the source table that you want to sync to an online table. From the kebab menu, select **Create online table**.
# MAGIC
# MAGIC * Use the selectors in the dialog to configure the online table.
# MAGIC   * Name: Name to use for the online table in Unity Catalog.
# MAGIC   * Primary Key: Column(s) in the source table to use as primary key(s) in the online table.
# MAGIC   * Timeseries Key: (Optional). Column in the source table to use as timeseries key. When specified, the online table includes only the row with the latest timeseries key value for each primary key.
# MAGIC   * Sync mode: Specifies how the synchronization pipeline updates the online table. Select one of Snapshot, Triggered, or Continuous.
# MAGIC   * Policy
# MAGIC     * Snapshot - The pipeline runs once to take a snapshot of the source table and copy it to the online table. Subsequent changes to the source table are automatically reflected in the online table by taking a new snapshot of the source and creating a new copy. The content of the online table is updated atomically.
# MAGIC     * Triggered - The pipeline runs once to create an initial snapshot copy of the source table in the online table. Unlike the Snapshot sync mode, when the online table is refreshed, only changes since the last pipeline execution are retrieved and applied to the online table. The incremental refresh can be manually triggered or automatically triggered according to a schedule.
# MAGIC     * Continuous - The pipeline runs continuously. Subsequent changes to the source table are incrementally applied to the online table in real time streaming mode. No manual refresh is necessary.
# MAGIC * When you are done, click Confirm. The online table page appears.
# MAGIC
# MAGIC The new online table is created under the catalog, schema, and name specified in the creation dialog. In Catalog Explorer, the online table is indicated by online table icon.

# COMMAND ----------

# MAGIC %md ## Create the function

# COMMAND ----------

# MAGIC %md The next cell defines a function that calculates the distance between the destination and the user's current location.

# COMMAND ----------

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
fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)

# COMMAND ----------

# MAGIC %md You can now view the `FeatureSpec` (`travel_spec`) and the distance function (`distance`) in Catalog Explorer. Click **Catalog** in the sidebar. In the Catalog Explorer, navigate to your schema in the **main** catalog. The `FeatureSpec` and the function appear under **Functions**. 
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/catalog-explorer.png"/>

# COMMAND ----------

# MAGIC %md ## Create a Feature & Function Serving endpoint

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

# Create endpoint
endpoint_name = "cjc-location"

workspace.serving_endpoints.create(
  name=endpoint_name,
  config=EndpointCoreConfigInput(
    served_entities=[
      ServedEntityInput(
        entity_name=feature_spec_name,
        scale_to_zero_enabled=True,
        workload_size="Small"
      )
    ]
  )
)

# COMMAND ----------

# Get the status of the endpoint
endpoint_status = workspace.serving_endpoints.get(name=endpoint_name)
print(endpoint_status)

# COMMAND ----------

# MAGIC %md You can now view the status of the Feature Serving Endpoint in the table on the **Serving endpoints** page. Click **Serving** in the sidebar to display the page.

# COMMAND ----------

# MAGIC %md ## Query the endpoint

# COMMAND ----------

# Wait until the endpoint is ready. This can take about 15 minutes.
import time
while workspace.serving_endpoints.get(name=endpoint_name).pending_config:
  time.sleep(20)

# COMMAND ----------

import requests
import json


def query_endpoint(url, token, lookup_keys):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    data_json = json.dumps({
      'dataframe_records': lookup_keys
    })
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    return response.json()
  
notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host_name = notebook_context.browserHostName().get()
url = f"https://{host_name}/serving-endpoints/{endpoint_name}/invocations"
token = notebook_context.apiToken().get()

query_endpoint(
  url, 
  token, 
  [{
    "destination_id": 5,
    "user_latitude": 31.2,
    "user_longitude": -96.7
  }]
)

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

workspace.serving_endpoints.delete(name=endpoint_name)

# COMMAND ----------

fe.delete_feature_spec(name=feature_spec_name)
