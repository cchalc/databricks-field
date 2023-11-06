# Databricks notebook source
# MAGIC %md # Feature Store Time Series Feature Table
# MAGIC
# MAGIC In this notebook, you create time series feature tables based on simulated Internet of Things (IoT) sensor data. You then:
# MAGIC - Generate a training set by performing a point-in-time lookup on the time series feature tables. 
# MAGIC - Use the training set to train a model.
# MAGIC - Register the model.
# MAGIC - Perform batch inference on new sensor data.
# MAGIC
# MAGIC ## Requirements
# MAGIC - Databricks Runtime 10.4 LTS for Machine Learning or above.
# MAGIC
# MAGIC **Note:** Starting with Databricks Runtime 13.2 ML, a change was made to the `create_table` API. Timestamp key columns must now be specified in the `primary_keys` argument. If you are using this notebook with Databricks Runtime 13.1 ML or below, use the commented-out code for the `create_table` call in Cmd 9.

# COMMAND ----------

# MAGIC %md ## Background
# MAGIC
# MAGIC The data used in this notebook is simulated to represent this situation: you have a series of readings from a set of IoT sensors installed in different rooms of a warehouse. You want to use this data to train a model that can detect when a person has entered a room. Each room has a temperature sensor, a light sensor, and a CO2 sensor, each of which records data at a different frequency.

# COMMAND ----------

# MAGIC %sql USE CATALOG cjc

# COMMAND ----------

import uuid
# run_id = str(uuid.uuid4()).replace('-', '')

# cjc.point_in_time_demo_155bfca5e41144bb9b84db35bc574705
run_id = '155bfca5e41144bb9b84db35bc574705'

database_name = f"point_in_time_demo_{run_id}"
model_name = f"pit_demo_model_{run_id}"

print(f"Database name: {database_name}")
print(f"Model name: {model_name}")

# Create the database
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")

# COMMAND ----------

# MAGIC %md ## Generate the simulated dataset
# MAGIC
# MAGIC In this step, you generate the simulated dataset and then create four Spark DataFrames, one each for the light sensors, the temperature sensors, the CO2 sensors, and the ground truth.

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql.functions import *

wavelength_lo, wavelength_hi = 209.291, 213.111
ppm_lo, ppm_hi = 35, 623.99
temp_lo, temp_hi = 15.01, 25.99
humidity_lo, humidity_hi = 35.16, 43.07

def is_person_in_the_room(wavelength, ppm, temp, humidity):
  return (
    (wavelength < (wavelength_lo + (wavelength_hi - wavelength_lo) * .45)) &
    (ppm > (.9 * ppm_hi)) &
    (temp > ((temp_hi + temp_lo) / 2)) &
    (humidity > (humidity_hi * .6))
  )

def generate_dataset(start, end):
  def generate_sensor_df(features):
    return pd.DataFrame({
      'room': np.random.choice(3, end-start),
      'ts': start + np.random.choice(end-start, end-start, replace=False) + np.random.uniform(-0.99, 0.99, end-start),
      **features
    }).sort_values(by=['ts'])    
  
  wavelength_df = generate_sensor_df({
    'wavelength': np.random.normal(np.mean([wavelength_lo, wavelength_hi]), 2, end-start),
  })
  temp_df = generate_sensor_df({
    'temp': np.random.normal(np.mean([temp_lo, temp_hi]), 4, end-start),
    'humidity': np.random.normal(np.mean([humidity_lo, humidity_hi]), 2, end-start), 
  })
  
  ppm_bern = np.random.binomial(1, 0.3, end-start)
  ppm_normal_1 = np.random.normal(ppm_lo, 8, end-start)
  ppm_normal_2 = np.random.normal(ppm_hi, 3, end-start)
  ppm_df = generate_sensor_df({
    'ppm': ppm_bern*ppm_normal_1+(1-ppm_bern)*ppm_normal_2
  })
  
  df = pd.DataFrame({
    'room': np.random.choice(3, end-start),    
    'ts': np.random.uniform(start, end, end-start)
  }).sort_values(by=['ts'])
  for right_df in [wavelength_df, ppm_df, temp_df]:
    df = pd.merge_asof(
      df, 
      right_df, 
      on='ts', 
      by='room'
    )
  df['person'] = is_person_in_the_room(df['wavelength'], df['ppm'], df['temp'], df['humidity'])
  
  wavelength_df['wavelength'] += np.random.uniform(-1, 1, end-start) * 0.2
  ppm_df['ppm'] += np.random.uniform(-1, 1, end-start) * 2
  temp_df['temp'] += np.random.uniform(-1, 1, end-start) 
  temp_df['humidity'] += np.random.uniform(-1, 1, end-start)
  
  light_sensors = spark.createDataFrame(wavelength_df) \
    .withColumn("ts", col("ts").cast('timestamp')) \
    .select(col("room").alias("r"), col("ts").alias("light_ts"), col("wavelength"))
  temp_sensors = spark.createDataFrame(temp_df) \
    .withColumn("ts", col("ts").cast('timestamp')) \
    .select("room", "ts", "temp", "humidity")
  co2_sensors = spark.createDataFrame(ppm_df) \
    .withColumn("ts", col("ts").cast('timestamp')) \
    .select(col("room").alias("r"), col("ts").alias("co2_ts"), col("ppm"))
  ground_truth = spark.createDataFrame(df[['room', 'ts', 'person']]) \
    .withColumn("ts", col("ts").cast('timestamp'))  

  return temp_sensors, light_sensors, co2_sensors, ground_truth  

temp_sensors, light_sensors, co2_sensors, ground_truth = generate_dataset(1458031648, 1458089824)
fixed_temps = temp_sensors.select("room", "ts", "temp").sample(False, 0.01).withColumn("temp", temp_sensors.temp + 0.25)

# COMMAND ----------

# Review the generated DataFrames

# temperature/humidity sensor readings for each room
display(temp_sensors.limit(3))
# light sensor readings for each room
display(light_sensors.limit(3))
# CO2 sensor readings for each room
display(co2_sensors.limit(3))
# ground-truth data about when a person was in a room
display(ground_truth.limit(3))

# COMMAND ----------

# MAGIC %md ## Create the time series feature tables
# MAGIC
# MAGIC In this step you create the time series feature tables. Each table uses the room as the primary key.

# COMMAND ----------

from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store.entities.feature_lookup import FeatureLookup

fs = FeatureStoreClient()

# COMMAND ----------

# Creates a time-series feature table for the temperature sensor data using the room as a primary key and the time as the timestamp key.

# For Databricks Runtime 13.2 for Machine Learning or above:
fs.create_table(
    f"{database_name}.temp_sensors",
    primary_keys=["room", "ts"],
    timestamp_keys=["ts"],
    df=temp_sensors,
    description="Readings from temperature and humidity sensors",
)

# For Databricks Runtime 13.1 for Machine Learning or below:
# fs.create_table(
#     f"{database_name}.temp_sensors",
#     primary_keys=["room"],
#     timestamp_keys=["ts"],
#     df=temp_sensors,
#     description="Readings from temperature and humidity sensors",
# )

# Creates a time-series feature table for the light sensor data using the room as a primary key and the time as the timestamp key.

# For Databricks Runtime 13.2 for Machine Learning or above:
fs.create_table(
    f"{database_name}.light_sensors",
    primary_keys=["r", "light_ts"],
    timestamp_keys=["light_ts"],
    df=light_sensors,
    description="Readings from light sensors",
)

# For Databricks Runtime 13.1 for Machine Learning or below:
# fs.create_table(
#     f"{database_name}.light_sensors",
#     primary_keys=["r"],
#     timestamp_keys=["light_ts"],
#     df=light_sensors,
#     description="Readings from light sensors",
# )

# Creates a time-series feature table for the CO2 sensor data using the room as a primary key and the time as the timestamp key. 

# For Databricks Runtime 13.2 for Machine Learning or above:
fs.create_table(
    f"{database_name}.co2_sensors",
    primary_keys=["r", "co2_ts"],
    timestamp_keys=["co2_ts"],
    df=co2_sensors,
    description="Readings from CO2 sensors",
)

# For Databricks Runtime 13.1 for Machine Learning or below:
# fs.create_table(
#     f"{database_name}.co2_sensors",
#     primary_keys=["r"],
#     timestamp_keys=["co2_ts"],
#     df=co2_sensors,
#     description="Readings from CO2 sensors",
# )

# COMMAND ----------

# MAGIC %md The time series feature tables are now visible in the <a href="#feature-store" target="_blank">Feature Store UI</a>. The `Timestamp Keys` field is populated for these feature tables. 

# COMMAND ----------

# MAGIC %md ## Updating the time-series feature tables
# MAGIC
# MAGIC Suppose that after you create the feature table, you receive updated values. For example, maybe some temperature readings were incorrectly pre-processed and need to be updated in the temperature time series feature table. 

# COMMAND ----------

display(fixed_temps.limit(3))

# COMMAND ----------

# MAGIC %md When you write a DataFrame to a time series feature table, the DataFrame must specify all the features of the feature table. To update a single feature column in the time series feature table, you must first join the updated feature column with other features in the table, specifying both a primary key and a timestamp key. Then, you can update the feature table.

# COMMAND ----------

temp_ft = fs.read_table(f"{database_name}.temp_sensors").drop('temp')
temp_update_df = fixed_temps.join(temp_ft, ["room", "ts"])
fs.write_table(f"{database_name}.temp_sensors", temp_update_df, mode="merge")

# COMMAND ----------

# MAGIC %md ## Create a training set with point-in-time lookups on time series feature tables
# MAGIC
# MAGIC In this step, you create a training set using the ground truth data by performing point-in-time lookups for the sensor data in the time series feature tables. 
# MAGIC
# MAGIC The point-in-time lookup retrieves the latest sensor value as of the timestamp given by the ground truth data for the room given by the ground truth data. 

# COMMAND ----------

training_labels, test_labels = ground_truth.randomSplit([0.75, 0.25])

display(training_labels.limit(5))

# COMMAND ----------

# Create point-in-time feature lookups that define the features for the training set. Each point-in-time lookup must include a `lookup_key` and `timestamp_lookup_key`.
feature_lookups = [
    FeatureLookup(
        table_name=f"{database_name}.temp_sensors",
        feature_names=["temp", "humidity"],
        rename_outputs={
          "temp": "room_temperature",
          "humidity": "room_humidity"
        },
        lookup_key="room",
        timestamp_lookup_key="ts"
    ),
    FeatureLookup(
        table_name=f"{database_name}.light_sensors",
        feature_names=["wavelength"],
        rename_outputs={"wavelength": "room_light"},
        lookup_key="room",
        timestamp_lookup_key="ts",      
    ),
    FeatureLookup(
        table_name=f"{database_name}.co2_sensors",
        feature_names=["ppm"],
        rename_outputs={"ppm": "room_co2"},
        lookup_key="room",
        timestamp_lookup_key="ts",      
    ),  
]

training_set = fs.create_training_set(
    training_labels,
    feature_lookups=feature_lookups,
    exclude_columns=["room", "ts"],
    label="person",
)
training_df = training_set.load_df()

# COMMAND ----------

display(training_df.limit(5))

# COMMAND ----------

spark.sql(f"use schema {database_name}")

# COMMAND ----------

training_df.write.mode("overwrite").saveAsTable("trainingset")

# COMMAND ----------

# MAGIC %md ## Train the model

# COMMAND ----------

features_and_label = training_df.columns
training_data = training_df.toPandas()[features_and_label]

X_train = training_data.drop(["person"], axis=1)
y_train = training_data.person.astype(int)

import lightgbm as lgb
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

mlflow.lightgbm.autolog()

model = lgb.train(
  {"num_leaves": 32, "objective": "binary"}, 
  lgb.Dataset(X_train, label=y_train.values),
  5
)

# COMMAND ----------

# Register the model in Model Registry.
# When you use `log_model`, the model is packaged with feature metadata so it automatically looks up features from Feature Store at inference.
fs.log_model(
  model,
  artifact_path="model_packaged",
  flavor=mlflow.lightgbm,
  training_set=training_set,
  registered_model_name=model_name
)

# COMMAND ----------

# MAGIC %md ## Score data with point-in-time lookups on time series feature tables
# MAGIC
# MAGIC The point-in-time lookup metadata provided to create the training set is packaged with the model so that the same lookup can be performed during scoring.

# COMMAND ----------

from mlflow.tracking import MlflowClient
def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
      version_int = int(mv.version)
      if version_int > latest_version:
        latest_version = version_int
    return latest_version

# COMMAND ----------

scored = fs.score_batch(
  f"models:/{model_name}/{get_latest_model_version(model_name)}",
  test_labels,
  result_type="float",
)

# COMMAND ----------

from pyspark.sql.types import BooleanType

classify_udf = udf(lambda pred: pred > 0.5, BooleanType())
class_scored = scored.withColumn("person_prediction", classify_udf(scored.prediction))

display(class_scored.limit(5))

# COMMAND ----------

from pyspark.sql.functions import avg, round
display(class_scored.select(round(avg((class_scored.person_prediction == class_scored.person).cast("int")), 3).alias("accuracy")))
