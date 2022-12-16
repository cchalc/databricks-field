# Databricks notebook source
# Pyspark and ML Imports
import os, json, requests
from pyspark.sql import functions as F
import pyspark.sql.types as t
#from pyspark.sql.functions import pandas_udf, PandasUDFType
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

import mlflow
import tensorflow as tf
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
tf.random.set_seed(42)

print(mlflow.__version__)
print(tf.__version__)

mlflow.tensorflow.autolog()
## mlflow.tensorflow.autolog(log_input_examples=True, log_model_signatures=True) # mlflow 1.25 release

# COMMAND ----------

# MAGIC %md
# MAGIC **Build Example data**

# COMMAND ----------

n_product = 20
## build example data
df= (spark.range(n_product*1000)
          .select(F.col("id").alias("record_id"), (F.col("id")%n_product).alias("device_id"))
          .withColumn("feature_1", F.rand() * 1)
          .withColumn("feature_2", F.rand() * 2)
          .withColumn("feature_3", F.rand() * 3)
      )
df_val = (spark.range(n_product*100)
          .select(F.col("id").alias("record_id"), (F.col("id")%n_product).alias("device_id"))
          .withColumn("feature_1", F.rand() * 1)
          .withColumn("feature_2", F.rand() * 2)
          .withColumn("feature_3", F.rand() * 3)
          .withColumn("abnormal", (F.col("feature_1")+F.col("feature_2")+F.col("feature_3")+F.rand())>3.)
      )
df.display()

# COMMAND ----------

df_train_and_eval = (df_val.withColumn("train_or_eval", F.lit("eval"))
                           .union(df.withColumn("abnormal", F.lit(None))
                                     .withColumn("train_or_eval", F.lit("train"))
                                  )
                    )
df_train_and_eval.cache().count()

# COMMAND ----------

# MAGIC %md **Distributed Model Training**
# MAGIC [Pandas UDFs](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html) allow us to vectorize Pandas code across multiple nodes in a cluster. Here we create a UDF to train an AE model against all the historic data for a particular device. We use a Grouped Map UDF as we perform this model training on the device group level.

# COMMAND ----------

def train_and_eval(features, df):        
    
    def build_auto_encoder_decoder(features, df_input):
        input_dim = len(features)
        inputs = Input(shape=input_dim)
        ## normalization layer
        scale_layer = Normalization()
        scale_layer.adapt(df_input)

        processed_input = scale_layer(inputs)
        ## AE model
        ae_model = Sequential([
        Dense(8, input_dim=input_dim, activation="relu"),
        Dense(2, activation="relu"), ## encoder
        Dense(2, activation="relu"), ## decoder
        Dense(8, activation="relu"),
        Dense(input_dim, activation="linear")
      ])
        ## final model
        outputs = ae_model(processed_input)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mse"])
        return model
      
    def select_threshold(error_h, error_f, beta=1):
        """
        select threshold based on fbeta
        """
        from sklearn.metrics import fbeta_score
        range_min, range_max = min(min(error_h), min(error_f)),max(max(error_h),max(error_f))
        sample_points = np.linspace(range_min, range_max, 100)
        metric_max = 0
        p_best = range_min
        for p in sample_points:
            precision = (error_h>p).sum()/((error_h>p).sum()+(error_f>p).sum())
            recall = (error_f>p).sum()/len(error_f)
            fbeta = (1. + beta**2)*precision*recall/(beta**2*precision+recall)
            if fbeta > metric_max:
                p_best = p
                metric_max = fbeta

        return p, metric_max    
    
    def train_and_eval_udf(df_pandas: pd.DataFrame) -> pd.DataFrame:
        '''
        Trains an sklearn model on grouped instances
        '''
        import mlflow
        mlflow.tensorflow.autolog()
        # Pull metadata
        device_id = df_pandas['device_id'].iloc[0]
        parent_run_id = df_pandas['parent_run_id'].iloc[0]
        # Train the model
        X = df_pandas[df_pandas.train_or_eval=="train"][features]
        model = build_auto_encoder_decoder(features, X)
        
        with mlflow.start_run(run_id=parent_run_id):
            with mlflow.start_run(nested=True) as child_run:
                mlflow.log_param("device_id", device_id)
                model.fit(X, X, validation_split=0.2, epochs=20,
                          callbacks=[EarlyStopping(patience=2,restore_best_weights=True)]
                         )
        ## get raw predictions of evaluaiton dataset
        X = df_pandas[df_pandas.train_or_eval=="eval"][features]
        #raw_predictions = pd.DataFrame(data=model.predict(X).values, columns=[f"{c}_pred" for c in features]) ## raw predictions from tf models
        raw_predictions = pd.DataFrame(data=model.predict(X), columns=[f"{c}_pred" for c in features]) ## raw predictions from tf models
        raw_predictions = pd.concat([df_pandas[df_pandas.train_or_eval=="eval"], raw_predictions], 
                                    axis=1)                               
        ## add additional error cols
        error_cols = []
        for c in features:
            raw_predictions[c+"_log_error"] = np.log((raw_predictions[f"{c}_pred"] - X[c]).abs())
            error_cols.append(c+"_log_error")
        ## reconstructed error
        raw_predictions["reconstructed_error"] = raw_predictions[error_cols].mean(axis=1)
        
        error_h = raw_predictions.loc[raw_predictions.abnormal==False, "reconstructed_error"].values
        error_f = raw_predictions.loc[raw_predictions.abnormal==True, "reconstructed_error"].values
        p, metric_max = select_threshold(error_h,error_f,beta=1)
        
        with mlflow.start_run(run_id=parent_run_id):
            with mlflow.start_run(run_id = child_run.info.run_id, nested=True):
                mlflow.log_param("threshold", p)
                mlflow.log_param("fbeta_eval", metric_max)
        
        returnDF = pd.DataFrame([[device_id, parent_run_id, child_run.info.run_id, p, metric_max]], 
                    columns=["device_id", "parent_run_id", "mlflow_run_id", "threshold", "fbeta_eval"])
        return returnDF
    
    ## udf return schema
    train_schema = t.StructType([t.StructField('device_id', t.IntegerType()), # unique pump ID
                           t.StructField('parent_run_id', t.StringType()),         # run id of mlflow run
                           t.StructField('mlflow_run_id', t.StringType()),         # run id of mlflow run
                           t.StructField('threshold', t.FloatType()),         # run id of mlflow run
                           t.StructField('fbeta_eval', t.FloatType()),         # run id of mlflow run
    ])
    ## train with mlflow logging
    with mlflow.start_run() as run:
        df_train_results = df.withColumn("parent_run_id", F.lit(run.info.run_id)).groupby("device_id").applyInPandas(train_and_eval_udf, train_schema)  
    
    return df_train_results, run

# COMMAND ----------

features = ["feature_1", "feature_2", "feature_3"]
df_train_results, run = train_and_eval(features, df_train_and_eval)

# COMMAND ----------

df_train_results.cache().count()
display(df_train_results)

# COMMAND ----------

# MAGIC %md
# MAGIC **Inference**

# COMMAND ----------

# MAGIC %md
# MAGIC Query individual model info based on parent mlflow run id

# COMMAND ----------

#parent_run_id = df_train_results.select("parent_run_id").first()[0]
parent_run_id = run.info.run_id
experiment_id = run.info.experiment_id
parent_run_id, experiment_id

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
query = f"tags.mlflow.parentRunId='{parent_run_id}' and params.device_id='0'"
runs = MlflowClient().search_runs(experiment_ids=[experiment_id], filter_string=query)
runs[0].to_dictionary()["info"]["run_id"], runs[0].to_dictionary()["data"]["params"]["threshold"]

# COMMAND ----------

# MAGIC %md
# MAGIC Get data prediction with Pandas UDFs

# COMMAND ----------

def predict(model_input, 
            features=features, 
            parent_run_id=parent_run_id):

    def predict_udf(df_pandas: pd.DataFrame) -> pd.DataFrame:
        '''
        load the trained model and get prediction
        '''
        import mlflow
        import numpy as np
        from mlflow.tracking.client import MlflowClient

        # Pull metadata
        device_id = df_pandas["device_id"].iloc[0]

        ## search mlflow_run_id and threshold
        query = f"tags.mlflow.parentRunId='{parent_run_id}' and params.device_id='{device_id}'"
        mlflow_run = MlflowClient().search_runs(experiment_ids=["3870835935320393"], filter_string=query)[0]
        
        threshold = float(mlflow_run.to_dictionary()["data"]["params"]["threshold"])
        mlflow_run_id = mlflow_run.to_dictionary()["info"]["run_id"]
        model_uri = f"runs:/{mlflow_run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)

        X = df_pandas[features]
        raw_predictions = pd.DataFrame(data=model.predict(X).values, columns=[f"{c}_pred" for c in features]) ## raw predictions from tf models
        raw_predictions = pd.concat([df_pandas, raw_predictions], axis=1)                               
        ## add additional error cols
        error_cols = []
        for c in features:
            raw_predictions[c+"_log_error"] = np.log((raw_predictions[f"{c}_pred"] - X[c]).abs())
            error_cols.append(c+"_log_error")
        ## reconstructed error > threshold
        raw_predictions["Prediction_is_anomaly"] = raw_predictions[error_cols].mean(axis=1)>threshold
        return raw_predictions.drop(error_cols+[f"{c}_pred" for c in features], axis=1)

    df_inf = model_input
             
    ## prediction output schema
    pred_schema = (df_inf
                   .select(*df_inf.columns, 
                           F.lit(True).cast(t.BooleanType()).alias("Prediction_is_anomaly")
                          )
                   .schema
                      )
    df_predicted = df_inf.groupby("device_id").applyInPandas(predict_udf, pred_schema)

    return df_predicted

# COMMAND ----------

n_product = 20
df_inference =(spark.range(n_product*100)
                    .select(F.col("id").alias("record_id"), (F.col("id")%n_product).alias("device_id"))
                    .withColumn("feature_1", F.rand() * 1)
                    .withColumn("feature_2", F.rand() * 2)
                    .withColumn("feature_3", F.rand() * 3)
                )

# COMMAND ----------

predict(df_inference).display()

# COMMAND ----------

# MAGIC %md
# MAGIC **Log model and artifacts**
# MAGIC 
# MAGIC 
# MAGIC **Reference**
# MAGIC - [custom pyfunc workflow](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-create-custom-workflows)
# MAGIC - [notebook example] - custom code model

# COMMAND ----------

class GroupByAEWrapperModel(mlflow.pyfunc.PythonModel):
  
    def __init__(self, features, parent_run_id):
        self.features = features
        self.parent_run_id = parent_run_id
    
    def featurize(self, df):
        return df
        
    def groupby_predict(self, context=None, model_input=None):
        
        def predict_udf(df_pandas: pd.DataFrame) -> pd.DataFrame:
            '''
            load the trained model and get prediction
            '''
            import mlflow
            import numpy as np
            from mlflow.tracking.client import MlflowClient

            # Pull metadata
            device_id = df_pandas["device_id"].iloc[0]
            
            ## search mlflow_run_id
            query = f"tags.mlflow.parentRunId='{self.parent_run_id}' and params.device_id='{device_id}'"

            mlflow_run = MlflowClient().search_runs(experiment_ids=["3870835935320393"], filter_string=query)[0]
            mlflow_run_id = mlflow_run.to_dictionary()["info"]["run_id"]
            model_uri = f"runs:/{mlflow_run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)

            threshold = float(mlflow_run.to_dictionary()["data"]["params"]["threshold"])
            
            X = df_pandas[self.features]
            raw_predictions = pd.DataFrame(data=model.predict(X).values, columns=[f"{c}_pred" for c in self.features]) ## raw predictions from tf models
            raw_predictions = pd.concat([df_pandas, raw_predictions], axis=1)                               
            ## add additional error cols
            error_cols = []
            for c in self.features:
                raw_predictions[c+"_log_error"] = np.log((raw_predictions[f"{c}_pred"] - X[c]).abs())
                error_cols.append(c+"_log_error")
            ## reconstructed error
            raw_predictions["Prediction_is_anomaly"] = raw_predictions[error_cols].mean(axis=1)>threshold
            return raw_predictions.drop(error_cols+[f"{c}_pred" for c in self.features], axis=1)

        df_inf = self.featurize(model_input)

        ## prediction output schema
        pred_schema = (df_inf
                       .select(*df_inf.columns, 
                               F.lit(True).cast(t.BooleanType()).alias("Prediction_is_anomaly")
                              )
                       .schema
                          )
        df_predicted = df_inf.groupby("device_id").applyInPandas(predict_udf, pred_schema)

        return df_predicted

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
      python_model=GroupByAEWrapperModel(features,parent_run_id), 
      artifact_path="groupby_models", 
    )

# COMMAND ----------

imported_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/groupby_models")
display(imported_model.groupby_predict(df_inference))

# COMMAND ----------


