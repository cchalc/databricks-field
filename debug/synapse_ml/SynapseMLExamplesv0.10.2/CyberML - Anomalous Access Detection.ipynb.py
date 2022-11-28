# Databricks notebook source
# MAGIC %md
# MAGIC # CyberML - Anomalous Access Detection
# MAGIC 
# MAGIC Here we demonstrate a novel CyberML model which can learn user access patterns and then automatically detect anomalous user access based on learned behavior.
# MAGIC The model internally uses Collaborative Filtering for Implicit Feedback as published here: http://yifanhu.net/PUB/cf.pdf
# MAGIC and is based on Apache Spark's implementation of this: https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html.
# MAGIC 
# MAGIC This notebook demonstrates a usage example of Anomalous Resource Access model.
# MAGIC All the model requires is a dataset in which there are 'users' which access 'resources'.
# MAGIC The model is based on Collaborative Filtering and it uses Machine Learning to learn access patterns of users and resources.
# MAGIC When a user accesses a resource which is outside of the user's learned profile then this access recieves a high anomaly score.
# MAGIC 
# MAGIC In this notebook we provide a usage example and a synthetic dataset in which there are 3 departments:
# MAGIC (1) Finance, (2) HR and (3) Engineering.
# MAGIC In the training data users access only a subset of resources from their own departments.
# MAGIC To evaluate the model we use two datasets.
# MAGIC The first contains access patterns unseen during training in which users access resources within their departments (again, resources they didn't access during training but within their department).
# MAGIC The latter contains users accessing resources from outside their department.
# MAGIC We then use the model to assign anomaly scores expecting that the first get low anomaly scores and the latter recieve high anomaly scores.
# MAGIC This is what this example demonstrates.
# MAGIC 
# MAGIC Note: the data does NOT contain information about departments, this information is implictly learned by the model by analyzing the access patterns.

# COMMAND ----------

# MAGIC %md
# MAGIC # Create an Azure Databricks cluster and install the following libs
# MAGIC 
# MAGIC 1. In Cluster Libraries install from library source Maven:
# MAGIC Coordinates: com.microsoft.azure:synapseml_2.12:0.10.2
# MAGIC Repository: https://mmlspark.azureedge.net/maven
# MAGIC 
# MAGIC 2. In Cluster Libraries install from PyPI the library called plotly

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup & Initialization

# COMMAND ----------

# this is used to produce the synthetic dataset for this test
from synapse.ml.cyber.dataset import DataFactory

# the access anomalies model generator
from synapse.ml.cyber.anomaly.collaborative_filtering import AccessAnomaly

from pyspark.sql import functions as f, types as t

# COMMAND ----------

from pyspark.sql import SparkSession

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC # Loadup datasets

# COMMAND ----------

from synapse.ml.core.platform import running_on_databricks, running_on_synapse

if running_on_databricks():
    spark.sparkContext.setCheckpointDir("dbfs:/checkpoint_path/")
else:
    spark.sparkContext.setCheckpointDir("./tmp/checkpoint_path/")

if running_on_synapse():
    from notebookutils.visualization import display

factory = DataFactory(
    num_hr_users=25,
    num_hr_resources=50,
    num_fin_users=35,
    num_fin_resources=75,
    num_eng_users=15,
    num_eng_resources=25,
    single_component=True,
)

training_pdf = factory.create_clustered_training_data(ratio=0.4)

# a tenant id is used when independant datasets originate from different tenants, in this example we set all tenants-ids to the same value
training_df = spark.createDataFrame(training_pdf).withColumn("tenant_id", f.lit(0))
ingroup_df = spark.createDataFrame(
    factory.create_clustered_intra_test_data(training_pdf)
).withColumn("tenant_id", f.lit(0))
outgroup_df = spark.createDataFrame(
    factory.create_clustered_inter_test_data()
).withColumn("tenant_id", f.lit(0))

# COMMAND ----------

training_df.show()

# COMMAND ----------

print(training_df.count())
print(ingroup_df.count())
print(outgroup_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC # Model setup & training

# COMMAND ----------

access_anomaly = AccessAnomaly(
    tenantCol="tenant_id",
    userCol="user",
    resCol="res",
    likelihoodCol="likelihood",
    maxIter=1000,
)

# COMMAND ----------

model = access_anomaly.fit(training_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Apply model & show result stats

# COMMAND ----------

ingroup_scored_df = model.transform(ingroup_df)

# COMMAND ----------

ingroup_scored_df.agg(
    f.min("anomaly_score").alias("min_anomaly_score"),
    f.max("anomaly_score").alias("max_anomaly_score"),
    f.mean("anomaly_score").alias("mean_anomaly_score"),
    f.stddev("anomaly_score").alias("stddev_anomaly_score"),
).show()

# COMMAND ----------

outgroup_scored_df = model.transform(outgroup_df)

# COMMAND ----------

outgroup_scored_df.agg(
    f.min("anomaly_score").alias("min_anomaly_score"),
    f.max("anomaly_score").alias("max_anomaly_score"),
    f.mean("anomaly_score").alias("mean_anomaly_score"),
    f.stddev("anomaly_score").alias("stddev_anomaly_score"),
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Examine results

# COMMAND ----------

#
# Select a subset of results to send to Log Analytics
#

full_res_df = outgroup_scored_df.orderBy(f.desc("anomaly_score")).cache()

from pyspark.sql.window import Window

w = Window.partitionBy("tenant_id", "user", "res").orderBy(f.desc("anomaly_score"))

# select values above threshold
results_above_threshold = full_res_df.filter(full_res_df.anomaly_score > 1.0)

# get distinct resource/user and corresponding timestamp and highest score
results_to_la = (
    results_above_threshold.withColumn("index", f.row_number().over(w))
    .orderBy(f.desc("anomaly_score"))
    .select("tenant_id", f.col("user"), f.col("res"), "anomaly_score")
    .where("index == 1")
    .limit(100)
    .cache()
)

# add a fake timestamp to the results
results_to_la = results_to_la.withColumn("timestamp", f.current_timestamp())

display(results_to_la)

# COMMAND ----------

# MAGIC %md
# MAGIC # Display all resource accesses by users with highest anomalous score

# COMMAND ----------

# !pip install plotly

# COMMAND ----------

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, offline

import numpy as np
import pandas as pd

print(__version__)  # requires version >= 1.9.0

# run plotly in offline mode
offline.init_notebook_mode()

# COMMAND ----------

# Find all server accesses of users with high predicted scores
# For display, limit to top 25 results
results_to_display = results_to_la.orderBy(f.desc("anomaly_score")).limit(25).cache()
interesting_records = full_res_df.join(results_to_display, ["user"], "left_semi")
non_anomalous_records = interesting_records.join(
    results_to_display, ["user", "res"], "left_anti"
)

top_non_anomalous_records = (
    non_anomalous_records.groupBy("tenant_id", "user", "res")
    .agg(
        f.count("*").alias("count"),
    )
    .select(f.col("tenant_id"), f.col("user"), f.col("res"), "count")
)

# pick only a subset of non-anomalous record for UI
w = Window.partitionBy(
    "tenant_id",
    "user",
).orderBy(f.desc("count"))

# pick top non-anomalous set
top_non_anomalous_accesses = (
    top_non_anomalous_records.withColumn("index", f.row_number().over(w))
    .orderBy(f.desc("count"))
    .select("tenant_id", f.col("user"), f.col("res"), f.col("count"))
    .where("index in (1,2,3,4,5)")
    .limit(25)
)

# add back anomalous record
fileShare_accesses = (
    top_non_anomalous_accesses.select("user", "res", "count")
    .union(results_to_display.select("user", "res", f.lit(1).alias("count")))
    .cache()
)

# COMMAND ----------

# get unique users and file shares
high_scores_df = fileShare_accesses.toPandas()
unique_arr = np.append(high_scores_df.user.unique(), high_scores_df.res.unique())

unique_df = pd.DataFrame(data=unique_arr, columns=["name"])
unique_df["index"] = range(0, len(unique_df.index))

# create index for source & target and color for the normal accesses
normal_line_color = "rgba(211, 211, 211, 0.8)"
anomolous_color = "red"
x = (
    pd.merge(high_scores_df, unique_df, how="left", left_on="user", right_on="name")
    .drop(["name"], axis=1)
    .rename(columns={"index": "userIndex"})
)
all_access_index_df = (
    pd.merge(x, unique_df, how="left", left_on="res", right_on="name")
    .drop(["name"], axis=1)
    .rename(columns={"index": "resIndex"})
)
all_access_index_df["color"] = normal_line_color

# results_to_display index, color and
y = results_to_display.toPandas().drop(
    ["tenant_id", "timestamp", "anomaly_score"], axis=1
)
y = (
    pd.merge(y, unique_df, how="left", left_on="user", right_on="name")
    .drop(["name"], axis=1)
    .rename(columns={"index": "userIndex"})
)
high_scores_index_df = (
    pd.merge(y, unique_df, how="left", left_on="res", right_on="name")
    .drop(["name"], axis=1)
    .rename(columns={"index": "resIndex"})
)
high_scores_index_df["count"] = 1
high_scores_index_df["color"] = anomolous_color

# substract 1 for the red entries in all_access df
hsi_df = high_scores_index_df[["user", "res", "count"]].rename(
    columns={"count": "hsiCount"}
)
all_access_updated_count_df = pd.merge(
    all_access_index_df,
    hsi_df,
    how="left",
    left_on=["user", "res"],
    right_on=["user", "res"],
)
all_access_updated_count_df["count"] = np.where(
    all_access_updated_count_df["hsiCount"] == 1,
    all_access_updated_count_df["count"] - 1,
    all_access_updated_count_df["count"],
)
all_access_updated_count_df = all_access_updated_count_df.loc[
    all_access_updated_count_df["count"] > 0
]
all_access_updated_count_df = all_access_updated_count_df[
    ["user", "res", "count", "userIndex", "resIndex", "color"]
]

# combine the two tables
frames = [all_access_updated_count_df, high_scores_index_df]
display_df = pd.concat(frames, sort=True)
# display_df.head()

# COMMAND ----------

data_trace = dict(
    type="sankey",
    domain=dict(x=[0, 1], y=[0, 1]),
    orientation="h",
    valueformat=".0f",
    node=dict(
        pad=10,
        thickness=30,
        line=dict(color="black", width=0),
        label=unique_df["name"].dropna(axis=0, how="any"),
    ),
    link=dict(
        source=display_df["userIndex"].dropna(axis=0, how="any"),
        target=display_df["resIndex"].dropna(axis=0, how="any"),
        value=display_df["count"].dropna(axis=0, how="any"),
        color=display_df["color"].dropna(axis=0, how="any"),
    ),
)

layout = dict(
    title="All resources accessed by users with highest anomalous scores",
    height=772,
    font=dict(size=10),
)

fig = dict(data=[data_trace], layout=layout)

p = plot(fig, output_type="div")

if running_on_databricks():
    displayHTML(p)
else:
    import IPython

    IPython.display.HTML(p)

# COMMAND ----------


