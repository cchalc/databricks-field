# Databricks notebook source
# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-mlops-end2end-mark_oswald` from the dropdown menu ([open cluster configuration](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0321-181439-fp8wb6mu/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('mlops-end2end')` or re-install the demo: `dbdemos.install('mlops-end2end')`*

# COMMAND ----------

# MAGIC %md
# MAGIC # Use the best Auto-ML generated notebook to bootstrap our ML Project
# MAGIC 
# MAGIC We have selected the notebook from best run from the Auto ML experiment and reusing it to build our model.
# MAGIC 
# MAGIC All the code below has been automatically generated. As Data Scientist, I can tune it based on the business knowledge I have if needed.
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-2.png" width="1200">
# MAGIC 
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=1444828305810485&aip=1&t=event&ec=dbdemos&ea=VIEW&dp=%2F_dbdemos%2Fdata-science%2Fmlops-end2end%2F02_automl_baseline&uid=687989441193254">
# MAGIC <!-- [metadata={"description":"MLOps end2end workflow: Auto-ML notebook",
# MAGIC  "authors":["quentin.ambard@databricks.com"],
# MAGIC  "db_resources":{},
# MAGIC   "search_tags":{"vertical": "retail", "step": "Data Engineering", "components": ["auto-ml"]},
# MAGIC                  "canonicalUrl": {"AWS": "", "Azure": "", "GCP": ""}}] -->

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=false $catalog="hive_metastore"

# COMMAND ----------

# MAGIC %md
# MAGIC # XGBoost training
# MAGIC This is an auto-generated notebook. To reproduce these results, attach this notebook to the **SR_demo** cluster and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/3087977229142441/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/3087977229142439) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.
# MAGIC 
# MAGIC Runtime Version: _8.4.x-cpu-ml-scala2.12_

# COMMAND ----------

import mlflow
#Added for the demo purpose
run = get_automl_churn_run(f'{dbName}.dbdemos_mlops_churn_features', force_refresh = False)
# Use MLflow to track experiments
mlflow.set_experiment(run["experiment_path"])

target_col = "churn"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

import os
import uuid
import shutil
import pandas as pd

from mlflow.tracking import MlflowClient

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)

# Download the artifact and read it into a pandas DataFrame
input_client = MlflowClient()
input_data_path = input_client.download_artifacts(run["data_run_id"], "data", input_temp_dir)
df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))

# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `["customer_id"]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["online_security_no", "online_backup_yes", "phone_service_yes", "streaming_tv_yes", "dependents_no", "online_backup_no_internet_service", "tech_support_no", "tenure", "multiple_lines_yes", "payment_method_credit_card__automatic_", "online_backup_no", "phone_service_no", "payment_method_mailed_check", "partner_no", "streaming_tv_no_internet_service", "paperless_billing_yes", "streaming_movies_no_internet_service", "internet_service_dsl", "payment_method_electronic_check", "multiple_lines_no_phone_service", "contract_two_year", "tech_support_no_internet_service", "device_protection_no", "device_protection_no_internet_service", "senior_citizen", "payment_method_bank_transfer__automatic_", "dependents_yes", "device_protection_yes", "tech_support_yes", "streaming_movies_yes", "streaming_tv_no", "gender_female", "paperless_billing_no", "contract_month_to_month", "contract_one_year", "streaming_movies_no", "online_security_yes", "online_security_no_internet_service", "monthly_charges", "multiple_lines_no", "total_charges", "internet_service_fiber_optic", "gender_male", "internet_service_no", "partner_yes"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC 
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), ["contract_month_to_month", "contract_one_year", "contract_two_year", "dependents_no", "dependents_yes", "device_protection_no", "device_protection_no_internet_service", "device_protection_yes", "gender_female", "gender_male", "internet_service_dsl", "internet_service_fiber_optic", "internet_service_no", "monthly_charges", "multiple_lines_no", "multiple_lines_no_phone_service", "multiple_lines_yes", "online_backup_no", "online_backup_no_internet_service", "online_backup_yes", "online_security_no", "online_security_no_internet_service", "online_security_yes", "paperless_billing_no", "paperless_billing_yes", "partner_no", "partner_yes", "payment_method_bank_transfer__automatic_", "payment_method_credit_card__automatic_", "payment_method_electronic_check", "payment_method_mailed_check", "phone_service_no", "phone_service_yes", "senior_citizen", "streaming_movies_no", "streaming_movies_no_internet_service", "streaming_movies_yes", "streaming_tv_no", "streaming_tv_no_internet_service", "streaming_tv_yes", "tech_support_no", "tech_support_no_internet_service", "tech_support_yes", "tenure", "total_charges"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, ["online_security_no", "online_backup_yes", "phone_service_yes", "streaming_tv_yes", "dependents_no", "online_backup_no_internet_service", "tech_support_no", "tenure", "multiple_lines_yes", "payment_method_credit_card__automatic_", "online_backup_no", "phone_service_no", "payment_method_mailed_check", "partner_no", "streaming_tv_no_internet_service", "paperless_billing_yes", "streaming_movies_no_internet_service", "internet_service_dsl", "payment_method_electronic_check", "multiple_lines_no_phone_service", "contract_two_year", "device_protection_no", "senior_citizen", "device_protection_no_internet_service", "tech_support_no_internet_service", "payment_method_bank_transfer__automatic_", "dependents_yes", "device_protection_yes", "tech_support_yes", "streaming_movies_yes", "streaming_tv_no", "gender_female", "paperless_billing_no", "contract_month_to_month", "contract_one_year", "streaming_movies_no", "online_security_yes", "online_security_no_internet_service", "monthly_charges", "total_charges", "multiple_lines_no", "internet_service_fiber_optic", "gender_male", "internet_service_no", "partner_yes"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["contract_month_to_month", "contract_one_year", "contract_two_year", "dependents_no", "dependents_yes", "device_protection_no", "device_protection_no_internet_service", "device_protection_yes", "gender_female", "gender_male", "internet_service_dsl", "internet_service_fiber_optic", "internet_service_no", "multiple_lines_no", "multiple_lines_no_phone_service", "multiple_lines_yes", "online_backup_no", "online_backup_no_internet_service", "online_backup_yes", "online_security_no", "online_security_no_internet_service", "online_security_yes", "paperless_billing_no", "paperless_billing_yes", "partner_no", "partner_yes", "payment_method_bank_transfer__automatic_", "payment_method_credit_card__automatic_", "payment_method_electronic_check", "payment_method_mailed_check", "phone_service_no", "phone_service_yes", "senior_citizen", "streaming_movies_no", "streaming_movies_no_internet_service", "streaming_movies_yes", "streaming_tv_no", "streaming_tv_no_internet_service", "streaming_tv_yes", "tech_support_no", "tech_support_no_internet_service", "tech_support_yes"])]

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = numerical_transformers + categorical_one_hot_transformers

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC The input data is split by AutoML into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)
# MAGIC 
# MAGIC `_automl_split_col_8425` contains the information of which set a given row belongs to.
# MAGIC We use this column to split the dataset into the above 3 sets. 
# MAGIC The column should not be used for training so it is dropped after split is done.

# COMMAND ----------

from sklearn.model_selection import train_test_split

split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

X_train, X_val, y_train, y_val = train_test_split(split_X, split_y, random_state=520692802, stratify=split_y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/3087977229142441/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from xgboost import XGBClassifier

help(XGBClassifier)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(display='diagram')

xgbc_classifier = XGBClassifier(
  learning_rate=0.008098213761399603,
  max_depth=8,
  min_child_weight=3,
  subsample=0.12038118521316003,
  random_state=520692802,
)

model = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
    ("classifier", xgbc_classifier),
])

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(experiment_id=run['experiment_id'], run_name="random_forest") as mlflow_run:
    model.fit(X_train, y_train)
    
    # Log metrics for the training set
    skrf_training_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_train, y_train, prefix="training_", pos_label=1)

    # Log metrics for the validation set
    skrf_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_", pos_label=1)

    # Log metrics for the test set
    #skrf_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_", pos_label=1)

    # Display the logged metrics
    skrf_val_metrics = {k.replace("val_", ""): v for k, v in skrf_val_metrics.items()}
    #skrf_test_metrics = {k.replace("test_", ""): v for k, v in skrf_test_metrics.items()}
    display(pd.DataFrame([skrf_val_metrics], index=["validation"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC 
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC 
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = False

# COMMAND ----------

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, X_train.shape[0]), random_state=711162379)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(100, X_val.shape[0]), random_state=711162379)

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC 
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Let's register a first model version as example
model_name = "dbdemos_mlops_churn"
model_uri = f"runs:/{ run['best_trial_run_id'] }/model"
registered_model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix, ROC and Precision-Recall curves for validation data
# MAGIC 
# MAGIC We show the confusion matrix, ROC and Precision-Recall curves of the model on the validation data.
# MAGIC 
# MAGIC For the plots evaluated on the training and the test data, check the artifacts on the MLflow run page.

# COMMAND ----------

# Paste the entire output (%md ...) to an empty cell, and click the link to see the MLflow run page
print(f"%md [Link to model run page](#mlflow/experiments/{ run['experiment_id'] }/runs/{ run['best_trial_run_id'] }/artifactPath/model)")

# COMMAND ----------

# MAGIC %md [Link to model run page](#mlflow/experiments/4313537760615719/runs/d4b40d93f05d4ac192f166052a00df93/artifactPath/model)

# COMMAND ----------

import uuid
from IPython.display import Image

# Create temp directory to download MLflow model artifact
eval_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(eval_temp_dir, exist_ok=True)

# Download the artifact
eval_path = mlflow.artifacts.download_artifacts(run_id=run['best_trial_run_id'], dst_path=eval_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion matrix for validation dataset

# COMMAND ----------

eval_confusion_matrix_path = os.path.join(eval_path, "val_confusion_matrix.png")
display(Image(filename=eval_confusion_matrix_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROC curve for validation dataset

# COMMAND ----------

eval_roc_curve_path = os.path.join(eval_path, "val_roc_curve.png")
display(Image(filename=eval_roc_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision-Recall curve for validation dataset

# COMMAND ----------

eval_pr_curve_path = os.path.join(eval_path, "val_precision_recall_curve.png")
display(Image(filename=eval_pr_curve_path))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # Next step: setting up Webhook to automatically test Churn models
# MAGIC 
# MAGIC Now that we have a model ready, we need to deploy it to the registry and move it to STAGING. 
# MAGIC 
# MAGIC Before doing that, our ML Engineer team has to setup a workflow to programatically test the model and ensure it's quality. Let's see how this can be done with Databricks and MLFlow.
# MAGIC 
# MAGIC Next: [Setup model hook]($./03_webhooks_setup)
