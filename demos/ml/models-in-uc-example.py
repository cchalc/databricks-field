# Databricks notebook source
# MAGIC %md
# MAGIC # Models in Unity Catalog Example
# MAGIC This notebook illustrates how to use Models in Unity Catalog to build a machine learning application that forecasts the daily power output of a wind farm. The example shows how to:
# MAGIC
# MAGIC - Track and log models with MLflow.
# MAGIC - Register models to Unity Catalog.
# MAGIC - Describe models and deploy them for inference using aliases.
# MAGIC - Integrate registered models with production applications.
# MAGIC - Search and discover models in Unity Catalog.
# MAGIC - Archive and delete models.
# MAGIC
# MAGIC The article describes how to perform these steps using the MLflow Tracking and Models in Unity Catalog UIs and APIs.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC
# MAGIC 1. Unity Catalog must be enabled in your workspace.
# MAGIC
# MAGIC 2. Your workspace must be attached to a Unity Catalog metastore that supports privilege inheritance. This is true for all metastores created after August  25, 2022.
# MAGIC
# MAGIC 3. You must have access to run commands on a cluster with access to Unity Catalog.
# MAGIC
# MAGIC 4. This notebook creates models in the `main.default` schema by default. This requires `USE CATALOG` privilege on the `main` catalog, plus `CREATE MODEL` and `USE SCHEMA` privileges on the `main.default` schema. You can change the catalog and schema used in this notebook, as long as you have the same privileges on the catalog and schema of your choosing.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install MLflow Python client
# MAGIC Version 2.4.1 or above of the MLflow Python client is required to access Models in Unity Catalog. The following cell installs it, along with tensorflow which is used to train the ML model.

# COMMAND ----------

# MAGIC %md
# MAGIC     %pip install --upgrade "mlflow-skinny[databricks]>=2.4.1" tensorflow
# MAGIC     dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure MLflow client to access models in Unity Catalog
# MAGIC By default, the MLflow Python client creates models in the Databricks workspace model registry. To upgrade to models in Unity Catalog, configure the MLflow client as shown:

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load dataset
# MAGIC The following code loads a dataset containing weather data and power output information for a wind farm in the United States. The dataset contains `wind direction`, `wind speed`, and `air temperature` features sampled every six hours (once at `00:00`, once at `08:00`, and once at `16:00`), as well as daily aggregate power output (`power`), over several years.
# MAGIC

# COMMAND ----------

import pandas as pd
wind_farm_data = pd.read_csv("https://github.com/dbczumar/model-registry-demo-notebook/raw/master/dataset/windfarm_data.csv", index_col=0)

def get_training_data():
  training_data = pd.DataFrame(wind_farm_data["2014-01-01":"2018-01-01"])
  X = training_data.drop(columns="power")
  y = training_data["power"]
  return X, y

def get_validation_data():
  validation_data = pd.DataFrame(wind_farm_data["2018-01-01":"2019-01-01"])
  X = validation_data.drop(columns="power")
  y = validation_data["power"]
  return X, y

def get_weather_and_forecast():
  format_date = lambda pd_date : pd_date.date().strftime("%Y-%m-%d")
  today = pd.Timestamp('today').normalize()
  week_ago = today - pd.Timedelta(days=5)
  week_later = today + pd.Timedelta(days=5)

  past_power_output = pd.DataFrame(wind_farm_data)[format_date(week_ago):format_date(today)]
  weather_and_forecast = pd.DataFrame(wind_farm_data)[format_date(week_ago):format_date(week_later)]
  if len(weather_and_forecast) < 10:
    past_power_output = pd.DataFrame(wind_farm_data).iloc[-10:-5]
    weather_and_forecast = pd.DataFrame(wind_farm_data).iloc[-10:]

  return weather_and_forecast.drop(columns="power"), past_power_output["power"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train, register, and deploy model
# MAGIC The following code trains a neural network using TensorFlow Keras to predict power output based on the weather features in the dataset. MLflow APIs are used to register the fitted model to Unity Catalog.

# COMMAND ----------

# You can update the catalog and schema name containing the model in Unity Catalog if needed
CATALOG_NAME = "corey_abshire"
SCHEMA_NAME = "ml_team"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.wind_forecasting"

# COMMAND ----------

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from mlflow.models import infer_signature

def train_and_register_keras_model(X, y):
  with mlflow.start_run():
    model = Sequential()
    model.add(Dense(100, input_shape=(X.shape[-1],), activation="relu", name="hidden_layer"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")

    model.fit(X, y, epochs=100, batch_size=64, validation_split=.2)
    example_input = X[:10].to_numpy()
    example_output = model.predict(X)
    mlflow.tensorflow.log_model(
        model,
        artifact_path="model",
        signature=infer_signature(example_input, example_output),
        registered_model_name=MODEL_NAME
    )
  return model

X_train, y_train = get_training_data()
model = train_and_register_keras_model(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add model and model version descriptions using the API
# MAGIC You can use MLflow APIs to find the recently trained model version, then add descriptions to the model version and the registered model:

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
def get_latest_model_version(model_name):
  client = MlflowClient()
  model_version_infos = client.search_model_versions("name = '%s'" % model_name)
  return max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

latest_version = get_latest_model_version(model_name=MODEL_NAME)


# COMMAND ----------

client = MlflowClient()
client.update_registered_model(
  name=MODEL_NAME,
  description="This model forecasts the power output of a wind farm based on weather data. The weather data consists of three features: wind speed, wind direction, and air temperature."
)

client.update_model_version(
  name=MODEL_NAME,
  version=1,
  description="This model version was built using TensorFlow Keras. It is a feed-forward neural network with one hidden layer."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### View the model in the UI
# MAGIC You can view and manage registered models and model versions in Unity Catalog via the Data Explorer ([AWS](https://docs.databricks.com/data/index.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/data/)|[GCP](https://docs.gcp.databricks.com/data/index.html)).
# MAGIC Look for the model you just created under the `main` catalog and `default` schema.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy a model version for inference
# MAGIC Models in Unity Catalog support aliases ([AWS](https://docs.databricks.com/mlflow/model-registry.html#model-registry-concepts)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow/model-registry#model-registry-concepts)|[GCP](https://docs.gcp.databricks.com/mlflow/model-registry.html#model-registry-concepts)) for model deployment.
# MAGIC Aliases provide mutable, named references (e.g. "Champion", "Challenger") to a particular version of a registered model, that you can reference and target in downstream inference workflows. The following cell shows how to use MLflow APIs to assign the "Champion" alias to the newly-trained model version.

# COMMAND ----------

client = MlflowClient()
latest_version = get_latest_model_version(MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "Champion", latest_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model versions using the API
# MAGIC The MLflow Models component defines functions for loading models from several machine learning frameworks. For example, `mlflow.tensorflow.load_model()` is used to load TensorFlow models that were saved in MLflow format, and `mlflow.sklearn.load_model()` is used to load scikit-learn models that were saved in MLflow format.
# MAGIC
# MAGIC These functions can load models from Models in Unity Catalog by version number or alias:

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = "models:/{model_name}/1".format(model_name=MODEL_NAME)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

model_champion_uri = "models:/{model_name}@Champion".format(model_name=MODEL_NAME)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_champion_uri))
champion_model = mlflow.pyfunc.load_model(model_champion_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forecast power output with the champion model
# MAGIC
# MAGIC In this section, the champion model is used to evaluate weather forecast data for the wind farm. The `forecast_power()` application loads the latest version of the forecasting model from the specified stage and uses it to forecast power production over the next five days.
# MAGIC

# COMMAND ----------

from mlflow.tracking import MlflowClient

def plot(model_name, model_alias, model_version, power_predictions, past_power_output):
  import pandas as pd
  import matplotlib.dates as mdates
  from matplotlib import pyplot as plt
  index = power_predictions.index
  fig = plt.figure(figsize=(11, 7))
  ax = fig.add_subplot(111)
  ax.set_xlabel("Date", size=20, labelpad=20)
  ax.set_ylabel("Power\noutput\n(MW)", size=20, labelpad=60, rotation=0)
  ax.tick_params(axis='both', which='major', labelsize=17)
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
  ax.plot(index[:len(past_power_output)], past_power_output, label="True", color="red", alpha=0.5, linewidth=4)
  ax.plot(index, power_predictions.squeeze(), "--", label="Predicted by '%s'\nwith alias '%s' (Version %d)" % (model_name, model_alias, model_version), color="blue", linewidth=3)
  ax.set_ylim(ymin=0, ymax=max(3500, int(max(power_predictions.values) * 1.3)))
  ax.legend(fontsize=14)
  plt.title("Wind farm power output and projections", size=24, pad=20)
  plt.tight_layout()
  display(plt.show())

def forecast_power(model_name, model_alias):
  client = MlflowClient()
  model_version = client.get_model_version_by_alias(model_name, model_alias).version
  model_uri = "models:/{model_name}@{model_alias}".format(model_name=MODEL_NAME, model_alias=model_alias)
  model = mlflow.pyfunc.load_model(model_uri)
  weather_data, past_power_output = get_weather_and_forecast()
  power_predictions = pd.DataFrame(model.predict(weather_data))
  power_predictions.index = pd.to_datetime(weather_data.index)
  print(power_predictions)
  plot(model_name, model_alias, int(model_version), power_predictions, past_power_output)

forecast_power(MODEL_NAME, "Champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create and deploy a new model version
# MAGIC Classical machine learning techniques are also effective for power forecasting. The following code trains a random forest model using scikit-learn and registers it to Unity Catalog via the `mlflow.sklearn.log_model()` function.
# MAGIC

# COMMAND ----------

import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with mlflow.start_run():
  n_estimators = 300
  mlflow.log_param("n_estimators", n_estimators)

  rand_forest = RandomForestRegressor(n_estimators=n_estimators)
  rand_forest.fit(X_train, y_train)

  val_x, val_y = get_validation_data()
  mse = mean_squared_error(rand_forest.predict(val_x), val_y)
  print("Validation MSE: %d" % mse)
  mlflow.log_metric("mse", mse)

  # Specify the `registered_model_name` parameter of the `mlflow.sklearn.log_model()`
  # function to register the model to <UC>. This automatically
  # creates a new model version
  mlflow.sklearn.log_model(
    sk_model=rand_forest,
    artifact_path="sklearn-model",
    registered_model_name=MODEL_NAME,
    signature=infer_signature(val_x, val_y)
  )

# COMMAND ----------

# MAGIC %md ### Add description on new model version

# COMMAND ----------

new_model_version = get_latest_model_version(MODEL_NAME)

# COMMAND ----------

client.update_model_version(
  name=MODEL_NAME,
  version=new_model_version,
  description="This model version is a random forest containing 100 decision trees that was trained in scikit-learn."
)

# COMMAND ----------

# MAGIC %md ###  Mark new model version as Challenger and test the model
# MAGIC Before deploying a model to serve production traffic, it is often best practice to test it in on a sample
# MAGIC of production data or traffic. Previously, this notebook assigned the "Champion" alias to the model version serving
# MAGIC the majority of production workloads. The following code assigns the "Challenger" alias to the new
# MAGIC model version and evaluates its performance.

# COMMAND ----------

client.set_registered_model_alias(
  name=MODEL_NAME,
  alias="Challenger",
  version=new_model_version
)

forecast_power(MODEL_NAME, "Challenger")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy the new model version as the Champion model version
# MAGIC After verifying that the new model version performs well in tests, the following code assigns the
# MAGIC "Champion" alias to the new model version and uses the same `forecast_power` application code to produce
# MAGIC a power forecast

# COMMAND ----------

client.set_registered_model_alias(
  name=MODEL_NAME,
  alias="Champion",
  version=new_model_version
)

forecast_power(MODEL_NAME, "Champion")

# COMMAND ----------

# MAGIC %md 
# MAGIC There are now two model versions of the forecasting model: the model version trained in Keras model and the 
# MAGIC version trained in scikit-learn. Note that the "Challenger" alias remains assigned to the new, scikit-learn 
# MAGIC model version, so any downstream workloads that target the "Challenger" model version continue to run successfully:

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Archive and delete models
# MAGIC When a model version is no longer being used, you can delete it. You can also delete an entire registered 
# MAGIC model; this removes all of its associated model versions. Note that deleting a model version clears
# MAGIC any aliases assigned to the model version.
# MAGIC

# COMMAND ----------

client.delete_model_version(
   name=MODEL_NAME,
   version=1,
)

# COMMAND ----------

client = MlflowClient()
client.delete_registered_model(name=MODEL_NAME)
