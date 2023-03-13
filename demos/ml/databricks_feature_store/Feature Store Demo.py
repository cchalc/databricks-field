# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Feature Store example project
# MAGIC 
# MAGIC ## 1. Set up 
# MAGIC 
# MAGIC Run the **delta_table_setup** notebook to create the source tables used for feature generation.
# MAGIC 
# MAGIC - This notebook uses [arbitrary file support](https://docs.databricks.com/repos.html#work-with-non-notebook-files-in-a-databricks-repo) by referencing a function stored in a .py file. Also, note the use of [ipython autoloading](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html) for rapid development of functions and classes.  

# COMMAND ----------

# MAGIC %run ./delta_table_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Creating Feature Tables 
# MAGIC Run the **passenger_demographic_features** and **passenger_ticket_features** notebooks to create and populate the two feature store tables. 
# MAGIC     - Navitate to the Feature Store icon on the left pane of the Databricks UI. There will be two entries, one for each feature table.

# COMMAND ----------

# MAGIC %run ./passenger_demographic_features

# COMMAND ----------

# MAGIC %run ./passenger_ticket_features

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train a model
# MAGIC 
# MAGIC Run the **fit_model** notebook, which will perform the following tasks.
# MAGIC     -  Create an MLflow experiment
# MAGIC     - Create a training dataset by joining the two Feature Store tables
# MAGIC     - Fit a model to the training dataset
# MAGIC      - Log the model and the training dataset creation logic to the MLflow experiment
# MAGIC      - Create an entry for the model in the Model Registry
# MAGIC     - Promote the model to the 'Production' stage  

# COMMAND ----------

# MAGIC %run ./fit_model

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inference
# MAGIC 
# MAGIC Run the  **model_inference** notebook, which will perform the following tasks. 
# MAGIC     - Create a sample DataFrame of new record ids to score 
# MAGIC     - Create a helper function that given a model name and stage, will load the model's unique id
# MAGIC     - Apply the model to the record ids. MLflow joins the relevent features to the record ids before applying the model and generating a prediction.

# COMMAND ----------

# MAGIC %run ./model_inference
