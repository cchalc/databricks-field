# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Deploying AutoML model for realtime serving
# MAGIC
# MAGIC <img style="float: right;" src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/fsi/fraud-detection/fsi-fraud-real-time-serving.png" width="700px" />
# MAGIC
# MAGIC Let's deploy our model behind a scalable API to rate a Fraud likelihood in ms and reduce fraud in real-time.
# MAGIC
# MAGIC ## Databricks Model Serving
# MAGIC
# MAGIC Now that our model has been created with Databricks AutoML, we can easily flag it as Production Ready and turn on Databricks Model Serving.
# MAGIC
# MAGIC We'll be able to send HTTP Requests and get inference in real-time.
# MAGIC
# MAGIC Databricks Model Serving is a fully serverless:
# MAGIC
# MAGIC * One-click deployment. Databricks will handle scalability, providing blazing fast inferences and startup time.
# MAGIC * Scale down to zero as an option for best TCO (will shut down if the endpoint isn't used)
# MAGIC * Built-in support for multiple models & version deployed
# MAGIC * A/B Testing and easy upgrade, routing traffic between each versions while measuring impact
# MAGIC * Built-in metrics & monitoring
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=lakehouse&org_id=1444828305810485&notebook=%2F04-Data-Science-ML%2F04.3-Model-serving-realtime-inference-fraud&demo_name=lakehouse-fsi-fraud&event=VIEW&path=%2F_dbdemos%2Flakehouse%2Flakehouse-fsi-fraud%2F04-Data-Science-ML%2F04.3-Model-serving-realtime-inference-fraud&version=1">

# COMMAND ----------

# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-lakehouse-fsi-fraud-christopher_chalcraft` from the dropdown menu ([open cluster configuration](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0514-192212-a6e6n4pi/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('lakehouse-fsi-fraud')` or re-install the demo: `dbdemos.install('lakehouse-fsi-fraud')`*

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

# DBTITLE 1,Move the autoML model in production
client = mlflow.tracking.MlflowClient()
models = client.get_latest_versions("dbdemos_fsi_fraud")
models.sort(key=lambda m: m.version, reverse=True)
latest_model = models[0]

if latest_model.current_stage != 'Production':
  client.transition_model_version_stage("dbdemos_fsi_fraud", latest_model.version, stage = "Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Creating our Model Serving endpoint
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/fsi/fraud-detection/fsi-fraud-model-serving-ui.png" style="float: right; margin-left: 10px" width="400px"/>
# MAGIC
# MAGIC Let's start our Serverless Databricks Model Serving Endpoint. 
# MAGIC
# MAGIC To do it with the UI, make sure you are under the "Machine Learning" Persona (top left) and select Open your <a href="#mlflow/endpoints" target="_blank">"Model Serving"</a>. 
# MAGIC
# MAGIC Then select the model we created: `dbdemos_fsi_fraud` and the latest version available.
# MAGIC
# MAGIC In addition, we'll allow the endpoint to scale down to zero (the endpoint will shut down after a few minutes without requests). Because it's serverless, it'll be able to restart within a few seconds.
# MAGIC
# MAGIC *Note that the first startup time will take extra second as the model container image is being built. It's then saved to ensure faster startup time.*

# COMMAND ----------

# DBTITLE 1,Starting the model inference REST endpoint using Databricks API
#Start the enpoint using the REST API (you can do it using the UI directly)
serving_client.create_enpoint_if_not_exists("dbdemos_fsi_fraud", model_name="dbdemos_fsi_fraud", model_version = latest_model.version, workload_size="Small", scale_to_zero_enabled=True, wait_start = True)

# COMMAND ----------

# DBTITLE 1,Running HTTP REST inferences in realtime !
#Let's get our dataset example
p = ModelsArtifactRepository("models:/dbdemos_fsi_fraud/Production").download_artifacts("") 
dataset =  {"dataframe_split": Model.load(p).load_input_example(p).to_dict(orient='split')}

#Let's run 30 inferences. The first run will wakeup the endpoint if it scaled to zeros.
endpoint_url = f"{serving_client.base_url}/realtime-inference/dbdemos_fsi_fraud/invocations"
print(f"Sending requests to {endpoint_url}")
for i in range(3):
    starting_time = timeit.default_timer()
    inferences = requests.post(endpoint_url, json=dataset, headers=serving_client.headers).json()
    print(f"Inference time, end 2 end :{round((timeit.default_timer() - starting_time)*1000)}ms")
    print(inferences)

# COMMAND ----------

# MAGIC %md
# MAGIC ## We now have our first model version available in production for real time inference!
# MAGIC
# MAGIC Open your [model endpoint](#mlflow/endpoints/dbdemos_fsi_fraud).
# MAGIC
# MAGIC We can now start using the endpoint API to send queries and start detecting potential fraud in real-time.
# MAGIC
# MAGIC We would typically add an extra security challenge if the model thinks this is a potential fraud. 
# MAGIC
# MAGIC This would allow to make simple transaction flow, without customer friction when the model is confident it's not a fraud, and add more security layers when we flag it as a potential fraud.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: Release a new model version and deploy it without interrupting our service
# MAGIC
# MAGIC After a review of the notebook generated by Databricks AutoML, our Data Scientists realized that we could better detect fraud by retraining a model better at handling imbalanced dataset.
# MAGIC
# MAGIC Open the next notebook [04.4-Upgrade-to-imbalance-and-xgboost-model-fraud]($./04.4-Upgrade-to-imbalance-and-xgboost-model-fraud) to train a new model and add it to our registry.
