# Databricks notebook source
from databricks.feature_store import FeatureStoreClient
from mlflow.tracking import MlflowClient

# COMMAND ----------

fs = FeatureStoreClient()

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS feature_store_db.passenger_ticket_feautures;
# MAGIC DROP TABLE IF EXISTS feature_store_db.passenger_demographic_feautures;
# MAGIC DROP TABLE IF EXISTS feature_store_db.passenger_labels;

# COMMAND ----------

fs.drop_table(
  name='feature_store_db.ticket_features'
)

# COMMAND ----------

fs.drop_table(
  name='feature_store_db.passenger_demographic_feautures'
)

# COMMAND ----------

from mlflow.tracking import MlflowClient

# COMMAND ----------

client = MlflowClient()

# COMMAND ----------

client.transition_model_version_stage(
  name="feature_store_demo_model",
  version=1,
  stage="Archived"
)

# COMMAND ----------

client.delete_registered_model(name="feature_store_demo_model")

# COMMAND ----------


