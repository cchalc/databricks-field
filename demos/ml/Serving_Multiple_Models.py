# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC Demonstrates how to log many models into one MLflow 'multiplex' model that can serve from one or the other based on input. This can help scale serving of a large number of models in one VM rather than many small VMs.

# COMMAND ----------

# MAGIC %md
# MAGIC # Serving Multiple Models from a Registered Model
# MAGIC 
# MAGIC MLflow allows models to deploy as real-time REST APIs. At the moment, a single MLflow model serves from one instance (typically one VM) serving models. This can be problematic when many models need to be served. Imagine 1000 similar models need to be served, to make predictions for 1000 different types of input; running 1000 instances of the model service could waste resources.
# MAGIC 
# MAGIC Or simply, managing 1000 registered models could be unwieldy, if the models are almost always trained and released together.
# MAGIC 
# MAGIC One way around this is to package many models into a single custom model, which internally delegates to one of many models based on the input, and deploy that 'bundle' of models as a single model.
# MAGIC 
# MAGIC This notebook considers a toy example, predicting flight delays between airports. Imagine that it's desirable to create not one big model predicting delay between any two of the 255 airports in the data set, but 255 models, each of which predicts delays from one single airport to the others.
# MAGIC 
# MAGIC First, read the data and create and log 255 flight delay models:

# COMMAND ----------

from pyspark.sql.functions import col, dayofmonth, hour, minute, month, to_timestamp

delays_df = spark.read.option('header', True).option('inferSchema', True).csv("/databricks-datasets/flights/*.csv").\
  withColumn("theDate", to_timestamp(col("date").cast("string"), "MddHHmm")).\
  withColumn("month", month("theDate")).\
  withColumn("day", dayofmonth("theDate")).\
  withColumn("hour", hour("theDate")).\
  withColumn("minute", minute("theDate")).\
  drop("date", "theDate")

display(delays_df)

# COMMAND ----------

# MAGIC %md
# MAGIC The modeling here is very simplistic as it is not the purpose of the example. Spark groups data by origin airport, and then fits a model with scikit-learn's `RandomForestRegressor` to predict delays. Here, much is omitted for simplicity in this example: tuning, evaluation, logging metrics, autologging, etc.
# MAGIC 
# MAGIC The models are tagged with their origin airport.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
 
# Fits and logs one model for data from one origin airport
def fit_model_for_origin(data):
  with mlflow.start_run():
    one_hot = ColumnTransformer([("one_hot_dest", OneHotEncoder(), ["destination"])], remainder="passthrough")
    rfr = RandomForestRegressor(n_estimators=10)
    origin_df = data.drop("origin", axis=1)
    pipeline = Pipeline([("one_hot", one_hot), ("regressor", rfr)])
    pipeline.fit(origin_df.drop("delay", axis=1), origin_df["delay"])
    mlflow.sklearn.log_model(pipeline, "model")
    mlflow.set_tag("origin", data["origin"].head(1).item())
    # In a real use case, more evaluation and logging would go here!
  return data["origin"].to_frame() # dummy value
  
# Fit models on all origin airports in parallel and log to MLflow
delays_df.repartition(128).groupBy("origin").applyInPandas(fit_model_for_origin, schema="origin:string").count() # count() just to trigger

# COMMAND ----------

# MAGIC %md
# MAGIC The models are logged in MLflow as usual and can be seen in the notebook's experiment. They could be registered, tested, etc. independently as usual if desired.
# MAGIC 
# MAGIC To create the wrapper model, all of the 255 models per origin airport need to be loaded back, keyed by origin airport, using MLflow:

# COMMAND ----------

runs_df = spark.read.format("mlflow-experiment").load().filter("tags.origin IS NOT NULL")
# Maps origin airport code to model for that origin:
origin_to_model = dict([(r["origin"], mlflow.sklearn.load_model(f"runs:/{r['run_id']}/model")) for r in runs_df.select("tags.origin", "run_id").collect()])

# COMMAND ----------

# MAGIC %md
# MAGIC Define a custom `PythonModel` in MLflow that will simply contain all the models. Its `predict` method delegates to one or the other based on the value of "origin" in the input.

# COMMAND ----------

from mlflow.pyfunc import PythonModel
import pandas as pd

class OriginDelegatingModel(PythonModel):
  def __init__(self, origin_to_model_map):
    self.origin_to_model_map = origin_to_model_map
    
  def predict(self, context, model_input):
    def predict_for_origin(data):
      model = self.origin_to_model_map[data["origin"]]
      no_origin = data.drop("origin")
      no_origin_input = pd.DataFrame([no_origin.values], columns=no_origin.index.values)
      return model.predict(no_origin_input)
    return model_input.apply(predict_for_origin, result_type="expand", axis=1).iloc[:,0]

# COMMAND ----------

# MAGIC %md
# MAGIC This wrapper model is then logged as one new model in MLflow, and registered, so that it can serve requests. Caution! All 255 models in this case serialized in one big model. It could get large, to store and to load into memory.

# COMMAND ----------

from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

with mlflow.start_run():
  any_run_id = runs_df.head()["run_id"]
  env_yaml_path = MlflowClient().download_artifacts(any_run_id, "model/conda.yaml")
  input_example = delays_df.limit(1).toPandas()
  X = input_example.drop("delay", axis=1)
  y = input_example["delay"]
  mlflow.pyfunc.log_model("model",\
                          python_model=OriginDelegatingModel(origin_to_model),\
                          input_example=X, signature=infer_signature(X, y),\
                          conda_env=env_yaml_path,\
                          registered_model_name="origin_model_example")
