# Databricks notebook source
# MAGIC %md # Travel Recommendation Example Notebook
# MAGIC 
# MAGIC This notebook illustrates the use of different feature computation modes: Batch, Streaming and On-Demand. It has been shown that Machine learning models degrade in performance as the features become stale. This is true more so for certain type of features than others. If the data being generated updates quickly and factors heavily into the outcome of the model, it should be updated regularly. However, updating static data often would lead to increased costs with no perceived benefits. This notebook illustrates various feature computation modes available in Databricks using Databricks Feature Store based on the feature freshness requirements for a travel recommendation website. 
# MAGIC 
# MAGIC ![Feature Computation Options](files/shared_uploads/aakrati.talati@databricks.com/freshness.png)
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/realtime/freshness.png"/>
# MAGIC 
# MAGIC This notebook builds a ranking model to predict likelihood of a user purchasing a destination package.
# MAGIC 
# MAGIC The notebook is structured as follows:
# MAGIC 
# MAGIC 1. Explore the dataset
# MAGIC 1. Compute the features in three computation modes
# MAGIC    * Batch features
# MAGIC    * Streaming features
# MAGIC    * On-demand features
# MAGIC 1. Publishes the features to the online store, based on the freshness requirements using streaming or batch mode (This notebook uses DynamoDB. For a list of supported online stores, see the Databricks documentation (AWS|Azure)) 
# MAGIC 1. Train and deploy the model
# MAGIC 1. Serve realtime queries with automatic feature lookup
# MAGIC 1. Clean up
# MAGIC 
# MAGIC ### Requirements
# MAGIC * Databricks Runtime 11.3 LTS for Machine Learning or above. 
# MAGIC * Access to AWS DynamoDB. This notebook uses DynamoDB as the online store.

# COMMAND ----------

# MAGIC %pip install geopy

# COMMAND ----------

# MAGIC %md ## Data Set
# MAGIC 
# MAGIC For the Travel recommendation model, we have different types of data available: 
# MAGIC 
# MAGIC * __Destination location__ data - is a static dataset of destinations for which my website serves vacation packages for. The destination location dataset consists of `latitude`, `longitude`, `name` and `price`. This dataset only changes when a new destination is added. The update frequency for this data is once a month and I compute these features in __batch-mode__. 
# MAGIC * __Destination popularity__ data - My website gathers the popularity information from the website usage logs based on number of impressions (e.g. `mean_impressions`, `mean_clicks_7d`) and user activity on those impressions. I use __batch-mode__ since my data sees shifts in patterns over longer periods of time. 
# MAGIC * __Destination availability__ data - Whenever a user books a room for the hotel, my destination availability and price (e.g. `destination_availability`, `destination_price`) gets affected. Because price and availability are a big driver for users booking vacation destinations, I want to keep this data fairly up-to-date, especially around holiday time. Batch-mode computation with hours of latency would not work, so I use spark structured streaming to update my data in __streaming-mode__.
# MAGIC * __User preferences__ - I have seen some of my users prefer to book closer to their current location whereas some prefer to go global and far-off. Because user location can only be determined on the booking time, I want to use __on-demand feature computation__ to calculate the `distance` between a context feature such as user location (`user_longitude`, `user_latitude`) and static feature destination location. This way I can keep my data in offline training and online model serving in sync. 

# COMMAND ----------

# MAGIC %md ![Datasets](files/shared_uploads/aakrati.talati@databricks.com/schema.png)
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/realtime/schema.png"/>

# COMMAND ----------

# MAGIC %md # Compute Features

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE DATABASE IF NOT EXISTS travel_recommendations;

# COMMAND ----------

# MAGIC %md ### Setup Helper Functions

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

# Get the user's username
def getUsername() -> str:
  return (
    spark
      .sql("SELECT current_user()")
      .first()[0]
      .lower()
      .split("@")[0]
      .replace(".", "_"))
    
# Cleanup and set dbfs dir 
def cleanup_dir(dir_name):
   dbutils.fs.rm(dir_name, True)
    
def get_latest_model_version(model_name: str):
  client = MlflowClient()
  models = client.get_latest_versions(model_name, stages=["None"])
  for m in models:
    new_model_version = m.version
  return new_model_version

# COMMAND ----------

# MAGIC %md ## Compute Batch Features
# MAGIC 
# MAGIC Calculate the aggregated features from the vacation purchase logs for destination and users. The destination features include popularity features such as impressions and clicks and pricing features such as price at the time of booking. The user features capture the user profile information such as past purchased price. Because the booking data does not change very often, it can be computed once per day in batch.

# COMMAND ----------

import pyspark.sql.functions as F

vacation_purchase_df = spark.read.option("inferSchema", "true").load("s3a://databricks-datasets-oregon/travel_recommendations_realtime/raw_travel_data/fs-demo_vacation-purchase_logs/", format="csv", header="true")
vacation_purchase_df = vacation_purchase_df.withColumn("booking_date", F.to_date("booking_date"))
display(vacation_purchase_df)

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark.sql.window as w

def user_features_fn(vacation_purchase_df):
    """
    Computes the user_features feature group.
    """
    return (
        vacation_purchase_df.withColumn(
            "lookedup_price_7d_rolling_sum",
            F.sum("price").over(
                w.Window.partitionBy("user_id")
                .orderBy(F.col("ts").cast("long"))
                .rangeBetween(start=-(7 * 86400), end=0)
            ),
        )
        .withColumn(
            "lookups_7d_rolling_sum",
            F.count("*").over(
                w.Window.partitionBy("user_id")
                .orderBy(F.col("ts").cast("long"))
                .rangeBetween(start=-(7 * 86400), end=0)
            ),
        )
        .withColumn(
            "mean_price_7d",
            F.col("lookedup_price_7d_rolling_sum") / F.col("lookups_7d_rolling_sum"),
        )
        .withColumn(
            "tickets_purchased",
            F.when(F.col("purchased") == True, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "last_6m_purchases",
            F.sum("tickets_purchased").over(
                w.Window.partitionBy("user_id")
                .orderBy(F.col("ts").cast("long"))
                .rangeBetween(start=-(6 * 30 * 86400), end=0)
            ),
        )
        .withColumn("day_of_week", F.dayofweek("ts"))
        .select("user_id", "ts", "mean_price_7d", "last_6m_purchases", "day_of_week")
    )

def destination_features_fn(vacation_purchase_df):
    """
    Computes the destination_features feature group.
    """
    return (
        vacation_purchase_df.withColumn(
            "clicked", F.when(F.col("clicked") == True, 1).otherwise(0)
        )
        .withColumn(
            "sum_clicks_7d",
            F.sum("clicked").over(
                w.Window.partitionBy("destination_id")
                .orderBy(F.col("ts").cast("long"))
                .rangeBetween(start=-(7 * 86400), end=0)
            ),
        )
        .withColumn(
            "sum_impressions_7d",
            F.count("*").over(
                w.Window.partitionBy("destination_id")
                .orderBy(F.col("ts").cast("long"))
                .rangeBetween(start=-(7 * 86400), end=0)
            ),
        )
        .select("destination_id", "ts", "sum_clicks_7d", "sum_impressions_7d")
    )
    return destination_df

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

fs.create_table(
    name="travel_recommendations.user_features",
    primary_keys=["user_id"],
    timestamp_keys="ts",
    df=user_features_fn(vacation_purchase_df),
    description="User Features",
)

fs.create_table(
    name="travel_recommendations.popularity_features",
    primary_keys=["destination_id"],
    timestamp_keys="ts",
    df=destination_features_fn(vacation_purchase_df),
    description="Destination Popularity Features",
)

# COMMAND ----------

# MAGIC %md Another static dataset is destination location feature which only updates every month because it need only be refreshed when a new destination package is offered. 

# COMMAND ----------

destination_location_df = spark.read.option("inferSchema", "true").load("s3a://databricks-datasets-oregon/travel_recommendations_realtime/raw_travel_data/fs-demo_destination-locations/", format="csv", header="true")

fs.create_table(
  name = "travel_recommendations.location_features",
  primary_keys="destination_id",
  df = destination_location_df,
  description = "Destination location features."
)

# COMMAND ----------

# MAGIC %md ## Compute Streaming Features
# MAGIC 
# MAGIC Availability of the destination can hugely affect the prices. Availability can change frequently especially around the holidays or long weekends during busy season. This data has a freshness requirement of every few minutes, so we use spark structured streaming to ensure data is fresh when doing model prediction. 

# COMMAND ----------

# MAGIC %md ![Hotel Availability](files/shared_uploads/aakrati.talati@databricks.com/streaming-1.png)
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/realtime/streaming.png"/>

# COMMAND ----------

from pyspark.sql.types import IntegerType, DoubleType, TimestampType, DateType, StringType, StructType, StructField
from pyspark.sql.functions import col

# Setup the delta checkpoint directory
fs_destination_availability_features_delta_checkpoint = "/Shared/fs_realtime/checkpoints/destination_availability_features_delta/"
cleanup_dir(fs_destination_availability_features_delta_checkpoint)

# Create schema 
destination_availability_schema = StructType([StructField("event_ts", TimestampType(), True),
                                             StructField("destination_id", IntegerType(), True),
                                             StructField("name", StringType(), True),
                                             StructField("booking_date", DateType(), True),
                                             StructField("price", DoubleType(), True),
                                             StructField("availability", IntegerType(), True),
                                             ])
destination_availability_log = spark.readStream.format("delta").option("maxFilesPerTrigger", 1000).option("inferSchema", "true").schema(destination_availability_schema).json("s3a://databricks-datasets-oregon/travel_recommendations_realtime/raw_travel_data/fs-demo_destination-availability_logs/json/*")
destination_availability_df = destination_availability_log.select(
  col("event_ts"),
  col("destination_id"),
  col("name"),
  col("booking_date"),
  col("price"),
  col("availability")
)
display(destination_availability_df)

# COMMAND ----------

fs.create_table(
    name="travel_recommendations.availability_features",
    primary_keys=["destination_id", "booking_date"],
    timestamp_keys=["event_ts"],
    schema=destination_availability_schema,
    description="Destination Availability Features",
)

# Now write the data to the feature table in "merge" mode
fs.write_table(
    name="travel_recommendations.availability_features",
    df=destination_availability_df,
    mode="merge",
    checkpoint_location=fs_destination_availability_features_delta_checkpoint
)

# COMMAND ----------

# MAGIC %md ## Compute Realtime/On-Demand features
# MAGIC 
# MAGIC User location is a context feature that is captured at the time of the query. This data is not known in advance hence the derived feature i.e., user distance from destination can only be computed in realtime at the prediction time. MLflow pyfunc captures this feature transformation using a preprocessing code that manipulates the input data frame before passing to model at training and serving time. 

# COMMAND ----------

# MAGIC %md ![MLflow pyfunc](files/shared_uploads/aakrati.talati@databricks.com/pyfunc.png)
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/realtime/pyfunc.png"/>

# COMMAND ----------

import geopy
import mlflow
import logging
import lightgbm as lgb
import pandas as pd
import geopy.distance as geopy_distance

from typing import Tuple


# Define the model class with on-demand computation model wrapper
class OnDemandComputationModelWrapper(mlflow.pyfunc.PythonModel):
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        try: 
            new_model_input = self._compute_ondemand_features(X_train)
            self.model = lgb.train(
              {"num_leaves": 32, "objective": "binary"}, 
              lgb.Dataset(new_model_input, label=y_train.values),
              5)
        except Exception as e:
            logging.error(e)
            
    def _distance(
        self,
        lon_lat_user: Tuple[float, float],
        lon_lat_destination: Tuple[float, float],
    ) -> float:
        """
        Wrapper call to calculate pair distance in miles
        ::lon_lat_user (longitude, latitude) tuple of user location
        ::lon_lat_destination (longitude, latitude) tuple of destination location
        """
        return geopy_distance.distance(
            geopy_distance.lonlat(*lon_lat_user),
            geopy_distance.lonlat(*lon_lat_destination),
        ).miles
        
    def _compute_ondemand_features(self, model_input: pd.DataFrame)->pd.DataFrame:
      try:
        # Fill NAs first
        loc_cols = ["user_longitude","user_latitude","longitude","latitude"]
        location_noNAs_pdf = model_input[loc_cols].fillna(model_input[loc_cols].median().to_dict())
        
        # Calculate distances
        model_input["distance"] = location_noNAs_pdf.apply(lambda x: self._distance((x[0], x[1]), (x[2], x[3])), axis=1)
        
        # Drop columns
        model_input.drop(columns=loc_cols)
        
      except Exception as e:
        logging.error(e)
        raise e
      return model_input

    def predict(self, context, model_input):
        new_model_input = self._compute_ondemand_features(model_input)
        return  self.model.predict(new_model_input)

# COMMAND ----------

# MAGIC %md # Train a custom model with batch + on-demand + streaming features
# MAGIC 
# MAGIC We will now use all the features created above to train a ranking model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get ground-truth labels and on-demand input features

# COMMAND ----------

# Random split to define a training and inference set
training_labels_df = (
  vacation_purchase_df
    .where("ts < '2022-11-23'")
)

test_labels_df = (
  vacation_purchase_df
    .where("ts >= '2022-11-23'")
)
display(training_labels_df.limit(5))

# COMMAND ----------

# MAGIC %md ## Create a training set

# COMMAND ----------

# DBTITLE 1,Define Feature Lookups (for batch and streaming input features)
from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store.entities.feature_lookup import FeatureLookup

fs = FeatureStoreClient()

feature_lookups = [
    FeatureLookup(
        table_name=f"travel_recommendations.popularity_features",
        lookup_key="destination_id",
        timestamp_lookup_key="ts"
    ),
    FeatureLookup(
        table_name=f"travel_recommendations.location_features",
        lookup_key="destination_id",
        feature_names=["latitude", "longitude"]
    ),
    FeatureLookup(
        table_name=f"travel_recommendations.user_features",
        lookup_key="user_id",
        timestamp_lookup_key="ts",
        feature_names=["mean_price_7d"]
    ),
      FeatureLookup(
        table_name=f"travel_recommendations.availability_features",
        lookup_key=["destination_id", "booking_date"],
        timestamp_lookup_key="ts",
        feature_names=["availability"]
    )
]

# COMMAND ----------

training_set = fs.create_training_set(
    training_labels_df,
    feature_lookups=feature_lookups,
#     exclude_columns=['user_id', 'destination_id', 'ts', 'booking_date', 'clicked', 'price'],
    label='purchased',
)

# COMMAND ----------

# DBTITLE 1,Load as spark dataframe
training_df = training_set.load_df()
display(training_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and log model to MLflow

# COMMAND ----------

# Record specific additional dependencies required by model serving
def get_conda_env():
  model_env = mlflow.pyfunc.get_default_conda_env()
  model_env["dependencies"][-1]["pip"] += [
    f"geopy=={geopy.__version__}",
    f"lightgbm=={lgb.__version__}",
    f"pandas=={pd.__version__}"
  ]
  return model_env

# COMMAND ----------

from sklearn.model_selection import train_test_split

with mlflow.start_run():
  
  # Split features and labels
  features_and_label = training_df.columns
 
  # Collect data into a Pandas array for training and testing
  data = training_df.toPandas()[features_and_label]
  train, test = train_test_split(data, random_state=123)
  X_train = train.drop(["purchased"], axis=1)
  X_test = test.drop(["purchased"], axis=1)
  y_train = train.purchased
  y_test = test.purchased

  # Fit
  pyfunc_model = OnDemandComputationModelWrapper()
  pyfunc_model.fit(X_train, y_train)
  
  # Log custom model to MLflow
  model_name = "realtime_destination_recommendations"
  fs.log_model(
    artifact_path="model",
    model=pyfunc_model,
    flavor = mlflow.pyfunc,
    training_set=training_set,
    registered_model_name=model_name,
    conda_env=get_conda_env()
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch score test set

# COMMAND ----------

scored_df = fs.score_batch(
  f"models:/{model_name}/{get_latest_model_version(model_name)}",
  vacation_purchase_df,
  result_type="float"
)

# COMMAND ----------

display(scored_df)

# COMMAND ----------

test_df = (
  scored_df
    .where("destination_id = 16")
)
display(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Accuracy calculation

# COMMAND ----------

from pyspark.sql import functions as F

scored_df2 = scored_df.withColumnRenamed("prediction", "original_prediction")
scored_df2 = scored_df2.withColumn("prediction", (F.when(F.col("original_prediction") >= 0.2, True).otherwise(False))) # simply convert the original probability predictions to true or false
pd_scoring = scored_df2.select("purchased", "prediction").toPandas()

from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(pd_scoring["purchased"], pd_scoring["prediction"]))

# COMMAND ----------

# MAGIC %md # Publish feature tables to online store
# MAGIC 
# MAGIC In order to use the above models in a realtime scenario, we will publish the table to a online store. This will allow the model to serve prediction queries with low-latency.
# MAGIC Follow the instructions in https://docs.databricks.com/machine-learning/feature-store/online-feature-stores.html#provide-online-store-credentials-using-databricks-secrets to store secrets in the Databricks secret manager with the below scope. 

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
from databricks.feature_store.online_store_spec import AmazonDynamoDBSpec

fs = FeatureStoreClient()

destination_location_online_store_spec = AmazonDynamoDBSpec(
  region="us-west-2",
  write_secret_prefix="feature-store-example-write/dynamo",
  read_secret_prefix="feature-store-example-read/dynamo",
  table_name = "feature_store_travel_recommendations_location_features"
)

destination_online_store_spec = AmazonDynamoDBSpec(
  region="us-west-2",
  write_secret_prefix="feature-store-example-write/dynamo",
  read_secret_prefix="feature-store-example-read/dynamo",
  table_name = "feature_store_travel_recommendations_popularity_features"
)

destination_availability_online_store_spec = AmazonDynamoDBSpec(
  region="us-west-2",
  write_secret_prefix="feature-store-example-write/dynamo",
  read_secret_prefix="feature-store-example-read/dynamo",
  table_name = "feature_store_travel_recommendations_availability_features"
)

user_online_store_spec = AmazonDynamoDBSpec(
  region="us-west-2",
  write_secret_prefix="feature-store-example-write/dynamo",
  read_secret_prefix="feature-store-example-read/dynamo",
  table_name = "feature_store_travel_recommendations_user_features"
)

# Setup the delta checkpoint directory
fs_destination_availability_features_online_checkpoint = "/Shared/fs_realtime/checkpoints/destination_availability_features_online/"
cleanup_dir(fs_destination_availability_features_online_checkpoint)

# COMMAND ----------

fs.publish_table(f"travel_recommendations.user_features", user_online_store_spec)

fs.publish_table(f"travel_recommendations.location_features", destination_location_online_store_spec)

fs.publish_table(f"travel_recommendations.popularity_features", destination_online_store_spec)

# Push features to Online Store through Spark Structured streaming
fs.publish_table(f"travel_recommendations.availability_features", 
                 destination_availability_online_store_spec,
                 streaming = True,
                 checkpoint_location=fs_destination_availability_features_online_checkpoint)

# COMMAND ----------

# MAGIC %md # Realtime Model Inference

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enable model inferencevia API call
# MAGIC 
# MAGIC After calling `log_model`, a new version of the model is saved. To provision a serving endpoint, follow the steps below.
# MAGIC 
# MAGIC 1. Click **Models** in the left sidebar. If you don't see it, switch to the Machine Learning Persona ([AWS](https://docs.databricks.com/workspace/index.html#use-the-sidebar)|[Azure](https://docs.microsoft.com/azure/databricks//workspace/index#use-the-sidebar)).
# MAGIC 2. Enable serving for the model named "realtime_destination_recommendations". See the Databricks documentation for details ([AWS](https://docs.databricks.com/applications/mlflow/model-serving.html#model-serving-from-model-registry)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/model-serving#model-serving-from-model-registry)).

# COMMAND ----------

# We need both a token for the API, which we can get from the notebook.
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
# With the token, we can create our authorization header for our subsequent REST calls
headers = {"Authorization": f"Bearer {token}"}

# Next we need an enpoint at which to execute our request which we can get from the Notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
# This ojbect comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
# Lastly, extract the databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]

# COMMAND ----------

import requests

url = f"https://{instance}/api/2.0/mlflow/endpoints/enable"
r = requests.post(url, headers=headers, json={"registered_model_name": model_name})
assert r.status_code == 200, f"Expected an HTTP 200 response, received {r.status_code}"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Send payloads via REST call
# MAGIC 
# MAGIC With Databricks Serverless Real-Time Inference, the endpoint takes a different body format:
# MAGIC You can see the Users in New York, see high scores for Florida whereas Users in California, see high scores for Hawaii.
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "dataframe_records": [
# MAGIC     {"user_id": 4, "booking_date": "2022-12-22", "destination_id": 16, "user_latitude": 40.71277, "user_longitude": -74.005974}, 
# MAGIC     {"user_id": 39, "booking_date": "2022-12-22", "destination_id": 1, "user_latitude": 37.77493, "user_longitude": -122.41942}
# MAGIC   ]
# MAGIC }
# MAGIC ```
# MAGIC 
# MAGIC Databricks Serverless Real-Time Inference is in preview; to enroll, follow the instructions ([AWS](https://docs.databricks.com/applications/mlflow/migrate-and-enable-serverless-real-time-inference.html#enable-serverless-real-time-inference-for-your-workspace)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/migrate-and-enable-serverless-real-time-inference#enable-serverless-real-time-inference-for-your-workspace)).

# COMMAND ----------

# DBTITLE 1,Create wrapper function
import requests

def score_model(data_json: dict):
    url = f"https://{instance}/model/{model_name}/{get_latest_model_version(model_name)}/invocations"
    response = requests.request(method="POST", headers=headers, url=url, json=data_json)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()

# COMMAND ----------

payload_json = {
  "dataframe_records": [
    # Users in New York, see high scores for Florida 
    {"user_id": 4, "booking_date": "2022-12-22", "destination_id": 16, "user_latitude": 40.71277, "user_longitude": -74.005974}, 
    # Users in California, see high scores for Hawaii 
    {"user_id": 39, "booking_date": "2022-12-22", "destination_id": 1, "user_latitude": 37.77493, "user_longitude": -122.41942} 
  ]
}

# COMMAND ----------

# MAGIC %md Wait for 5 mins before running the next command. The serving cluster needs to come up before sending the request.

# COMMAND ----------

print(score_model(payload_json))

# COMMAND ----------

# MAGIC %md ## Cleanup 
# MAGIC 
# MAGIC 1. Stop the serving endpoint by visiting models tab 
# MAGIC 2. Cleanup secrets in Databricks secret manager and the online feature table 
# MAGIC 3. Stop the streaming writes to feature table and online store
