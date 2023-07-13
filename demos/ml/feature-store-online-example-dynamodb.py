# Databricks notebook source
# MAGIC %md
# MAGIC # Online Feature Store example notebook
# MAGIC
# MAGIC This notebook illustrates the use of Databricks Feature Store to publish features to an online store for real-time 
# MAGIC serving and automated feature lookup. The problem is to predict the wine quality using a ML model
# MAGIC with a variety of static wine features and a realtime input. 
# MAGIC
# MAGIC ![wine picture](https://archive.ics.uci.edu/ml/assets/MLimages/Large186.jpg)
# MAGIC
# MAGIC This notebook creates an endpoint to predict the quality of a bottle of wine, given an ID and the realtime feature alcohol by volume (ABV).
# MAGIC
# MAGIC The notebook is structured as follows:
# MAGIC  
# MAGIC 1. Prepare the feature table
# MAGIC 2. Set up DynamoDB
# MAGIC     * This notebook uses DynamoDB. For a list of supported online stores, see the Databricks documentation ([AWS](https://docs.databricks.com/applications/machine-learning/feature-store/online-feature-stores.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/feature-store/online-feature-stores)).  
# MAGIC 3. Publish the features to online feature store
# MAGIC 4. Train and deploy the model
# MAGIC 5. Serve realtime queries with automatic feature lookup
# MAGIC 6. Clean up
# MAGIC
# MAGIC ### Data Set
# MAGIC
# MAGIC This example uses the [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality).
# MAGIC
# MAGIC ### Requirements
# MAGIC
# MAGIC * Databricks Runtime 10.4 LTS for Machine Learning or above
# MAGIC * Access to AWS DynamoDB
# MAGIC     - This notebook uses DynamoDB as the online store and guides you through how to generate secrets and register them with Databricks
# MAGIC     Secret Management.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/wine_quality_diagram.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Prepare the feature table
# MAGIC
# MAGIC Suppose you need to build an endpoint to predict wine quality with just the `wine_id`. There has to be a feature table saved in Feature Store where the endpoint can look up features of the wine by the `wine_id`. For the purpose of this demo, we need to prepare this feature table ourselves first. The steps are:
# MAGIC
# MAGIC 1. Load and clean the raw data.
# MAGIC 2. Separate features and labels.
# MAGIC 3. Save features into a feature table.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and clean the raw data 
# MAGIC
# MAGIC The raw data contains 12 columns including 11 features and the `quality` column. The `quality` column is an integer that ranges from 3 to 8. The goal is to build a model that predicts the `quality` value.

# COMMAND ----------

raw_data_frame = spark.read.load("/databricks-datasets/wine-quality/winequality-red.csv",format="csv",sep=";",inferSchema="true",header="true" )
display(raw_data_frame.limit(10))

# COMMAND ----------

# Have a look at the size of the raw data.
raw_data_frame.toPandas().shape

# COMMAND ----------

# MAGIC %md
# MAGIC There are some problems with the raw data:
# MAGIC 1. The column names contain space (' '), which is not compatible with Feature Store. 
# MAGIC 2. We need to add ID to the raw data so they can be looked up later by Feature Store.
# MAGIC
# MAGIC The following cell addresses these issues.

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
from pyspark.sql.functions import monotonically_increasing_id


def addIdColumn(dataframe, id_column_name):
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]


def renameColumns(df):
    renamed_df = df
    for column in df.columns:
        renamed_df = renamed_df.withColumnRenamed(column, column.replace(' ', '_'))
    return renamed_df


# Rename columns so that they are compatible with Feature Store
renamed_df = renameColumns(raw_data_frame)

# Add id column
id_and_data = addIdColumn(renamed_df, 'wine_id')

display(id_and_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's assume that the alcohol by volume (ABV) is a variable that changes over time after the wine is opened. The value will be provided as a realtime input in online inference. 
# MAGIC
# MAGIC Now, split the data into two parts and store only the part with static features to Feature Store. 

# COMMAND ----------

# wine_id and static features
id_static_features = id_and_data.drop('alcohol', 'quality')

# wine_id, realtime feature (alcohol), label (quality)
id_rt_feature_labels = id_and_data.select('wine_id', 'alcohol', 'quality')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a feature table
# MAGIC
# MAGIC Next, create a new hive database and save the feature data `id_static_features` into a feature table.

# COMMAND ----------

# %sql use catalog cjc

# COMMAND ----------

# MAGIC %sql
# MAGIC create database if not exists online_feature_store_example;

# COMMAND ----------

from databricks.feature_store.client import FeatureStoreClient

fs = FeatureStoreClient()
fs.create_table(
    name="online_feature_store_example.wine_static_features",
    primary_keys=["wine_id"],
    df=id_static_features,
    description="id and features of all wine",
)

# COMMAND ----------

# MAGIC %md
# MAGIC The feature data has been stored into the feature table. The next step is to set up access to AWS DynamoDB. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up DynamoDB Access Key
# MAGIC
# MAGIC In this section, you need to take some manual steps to make DynamoDB accessible to this notebook. Databricks creates and updates DynamoDB tables so that DynamoDB can work with Feature Store. The following steps create a new AWS IAM user with the required permissions. You can also choose to use your existing users or roles.
# MAGIC
# MAGIC ### Create an AWS IAM user and download secrets
# MAGIC 1. Go to AWS console http://console.aws.amazon.com, navigate to IAM, and click "Users".
# MAGIC 2. Click "Add users" and create a new user with "Access Key".
# MAGIC 3. Click Next and select policy `AmazonDynamoDBFullAccess`.
# MAGIC 4. Click Next until the user is created.
# MAGIC 5. Download the "Access key ID" and "Secret access key".
# MAGIC
# MAGIC ### Provide online store credentials using Databricks secrets
# MAGIC
# MAGIC **Note:** For simplicity, the commands below use predefined names for the scope and secrets. To choose your own scope and secret names, follow the process in the Databricks documentation ([AWS](https://docs.databricks.com/applications/machine-learning/feature-store/online-feature-stores.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/feature-store/online-feature-stores)).
# MAGIC
# MAGIC 1. Create two secret scopes in Databricks.
# MAGIC
# MAGIC     ```
# MAGIC     databricks secrets create-scope --scope feature-store-example-read
# MAGIC     databricks secrets create-scope --scope feature-store-example-write
# MAGIC     ```
# MAGIC
# MAGIC 2. Create secrets in the scopes.  
# MAGIC    **Note:** the keys should follow the format `<prefix>-access-key-id` and `<prefix>-secret-access-key` respectively. Again, for simplicity, these commands use predefined names here. When the commands run, you will be prompted to copy your secrets into an editor.
# MAGIC
# MAGIC     ```
# MAGIC     databricks secrets put --scope feature-store-example-read --key dynamo-access-key-id
# MAGIC     databricks secrets put --scope feature-store-example-read --key dynamo-secret-access-key
# MAGIC     
# MAGIC     databricks secrets put --scope feature-store-example-write --key dynamo-access-key-id
# MAGIC     databricks secrets put --scope feature-store-example-write --key dynamo-secret-access-key
# MAGIC     ```
# MAGIC     
# MAGIC Now the credentials are stored with Databricks Secrets. You will use them below to create the online store.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Publish the features to the online feature store
# MAGIC
# MAGIC This allows Feature Store to add a lineage information about the feature table and the online storage. So when the model serves real-time queries, it can lookup features from the online store for better performance.

# COMMAND ----------

from databricks.feature_store.online_store_spec import AmazonDynamoDBSpec

# Specify the online store.
# Note: these commands use the predefined secret prefix. If you used a different secret scope or prefix, edit these commands before running them.

online_store_spec = AmazonDynamoDBSpec(
  region="us-west-2",
  write_secret_prefix="feature-store-example-write/dynamo",
  read_secret_prefix="feature-store-example-read/dynamo",
  table_name = "feature_store_online_wine_features"
)

# Push the feature table to online store.
fs.publish_table("online_feature_store_example.wine_static_features", online_store_spec)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and deploy the model
# MAGIC
# MAGIC Now, you will train a classifier using features in the Feature Store. You only need to specify the primary key, and Feature Store will fetch the required features.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import logging
import mlflow.sklearn

from databricks.feature_store.entities.feature_lookup import FeatureLookup

# COMMAND ----------

# MAGIC %md
# MAGIC First, define a `TrainingSet`. The training set accepts a `feature_lookups` list, where each item represents some features from a feature table in the Feature Store. This example uses `wine_id` as the lookup key to fetch all the features from table `online_feature_store_example.wine_features`.

# COMMAND ----------

training_set = fs.create_training_set(
    id_rt_feature_labels,
    label='quality',
    feature_lookups=[
        FeatureLookup(
            table_name=f"online_feature_store_example.wine_static_features",
            lookup_key="wine_id"
        )
    ],
    exclude_columns=['wine_id'],
)

# Load the training data from Feature Store
training_df = training_set.load_df()

display(training_df)

# COMMAND ----------

# MAGIC %md
# MAGIC The next cell trains a RandomForestClassifier model.

# COMMAND ----------

X_train = training_df.drop('quality').toPandas()
y_train = training_df.select('quality').toPandas()

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train.values.ravel())

# COMMAND ----------

# MAGIC %md
# MAGIC Save the trained model using `log_model`. `log_model` also saves lineage information between the model and the features (through `training_set`). So, during serving, the model automatically knows where to fetch the features by just the lookup keys.

# COMMAND ----------

fs.log_model(
    model,
    artifact_path="model",
    flavor=mlflow.sklearn,
    training_set=training_set,
    registered_model_name="wine_quality_classifier"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve realtime queries with automatic feature lookup
# MAGIC
# MAGIC After calling `log_model`, a new version of the model is saved. To provision a serving endpoint, follow the steps below.
# MAGIC
# MAGIC 1. Click **Models** in the left sidebar. If you don't see it, switch to the Machine Learning Persona ([AWS](https://docs.databricks.com/workspace/index.html#use-the-sidebar)|[Azure](https://docs.microsoft.com/azure/databricks//workspace/index#use-the-sidebar)).
# MAGIC 2. Enable serving for the model named "wine_quality_classifier". See the Databricks documentation for details ([AWS](https://docs.databricks.com/applications/mlflow/model-serving.html#model-serving-from-model-registry)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/model-serving#model-serving-from-model-registry)).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send a query
# MAGIC
# MAGIC In the Serving page, there are three approaches for calling the model. You can try the "Browser" approach with a JSON format request, as shown below. But here we copy-pasted the Python approach to illustrate an programatic way.

# COMMAND ----------

# Fill in the Databricks access token value.
# Note: You can generate a new Databricks access token by going to left sidebar "Settings" > "User Settings" > "Access Tokens", or using databricks-cli.

DATABRICKS_TOKEN = "<DATABRICKS_TOKEN>"
assert DATABRICKS_TOKEN.strip() != "<DATABRICKS_TOKEN>"

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def predict(dataset):
    url = '<Replace with the URL shown in Serving page>'
    headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}'}
    data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    response = requests.request(method='POST', headers=headers, url=url, json=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC Now, suppose you opened a bottle of wine and you have a sensor to measure the current ABV from the bottle. Using the model and automated feature lookup with realtime serving, you can predict the quality of the wine using the measured ABV value as the realtime input "alcohol".

# COMMAND ----------

new_wine_ids = pd.DataFrame([(25, 7.9), (25, 11.0), (25, 27.9)], columns=['wine_id', "alcohol"])

print(predict(new_wine_ids))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notes on request format and API versions
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC The model serving endpoint accepts a JSON object as input.
# MAGIC ```
# MAGIC [ 
# MAGIC   {"wine_id": 25, "alcohol": 7.9}, 
# MAGIC   {"wine_id": 25, "alcohol": 11.0}
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC With Databricks Model, the endpoint takes a different body format:
# MAGIC ```
# MAGIC {
# MAGIC   "dataframe_records": [
# MAGIC     {"wine_id": 25, "alcohol": 7.9}, 
# MAGIC     {"wine_id": 25, "alcohol": 11.0}
# MAGIC   ]
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC Learn more about [Databricks Model Serving](https://docs.databricks.com/machine-learning/model-serving/migrate-model-serving.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up
# MAGIC
# MAGIC Follow this checklist to clean up the resources created by this notebook:
# MAGIC
# MAGIC 1. AWS DynamoDB Table
# MAGIC     * Go to AWS console and navigate to DynamoDB.
# MAGIC     * Delete the table `feature_store_online_wine_features`
# MAGIC 2. AWS user and access key
# MAGIC     * Go to AWS console and navigate to IAM.
# MAGIC     * Search and click on the newly created user.
# MAGIC     * Delete user or click "Make Inactive" on on the Access Key to disable the access.
# MAGIC 3. Secrets store on Databricks Secrets  
# MAGIC     `databricks secrets delete-scope --scope <scope-name>`
# MAGIC 4. Databricks access token
# MAGIC     * From the Databricks left sidebar, "Settings" > "User Settings" > "Access Tokens"
