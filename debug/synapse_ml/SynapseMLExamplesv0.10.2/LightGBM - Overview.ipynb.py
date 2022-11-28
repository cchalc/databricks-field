# Databricks notebook source
# MAGIC %md
# MAGIC # LightGBM

# COMMAND ----------

# MAGIC %md
# MAGIC [LightGBM](https://github.com/Microsoft/LightGBM) is an open-source,
# MAGIC distributed, high-performance gradient boosting (GBDT, GBRT, GBM, or
# MAGIC MART) framework. This framework specializes in creating high-quality and
# MAGIC GPU enabled decision tree algorithms for ranking, classification, and
# MAGIC many other machine learning tasks. LightGBM is part of Microsoft's
# MAGIC [DMTK](http://github.com/microsoft/dmtk) project.
# MAGIC 
# MAGIC ### Advantages of LightGBM
# MAGIC 
# MAGIC -   **Composability**: LightGBM models can be incorporated into existing
# MAGIC     SparkML Pipelines, and used for batch, streaming, and serving
# MAGIC     workloads.
# MAGIC -   **Performance**: LightGBM on Spark is 10-30% faster than SparkML on
# MAGIC     the Higgs dataset, and achieves a 15% increase in AUC.  [Parallel
# MAGIC     experiments](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#parallel-experiment)
# MAGIC     have verified that LightGBM can achieve a linear speed-up by using
# MAGIC     multiple machines for training in specific settings.
# MAGIC -   **Functionality**: LightGBM offers a wide array of [tunable
# MAGIC     parameters](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst),
# MAGIC     that one can use to customize their decision tree system. LightGBM on
# MAGIC     Spark also supports new types of problems such as quantile regression.
# MAGIC -   **Cross platform** LightGBM on Spark is available on Spark, PySpark, and SparklyR
# MAGIC 
# MAGIC ### LightGBM Usage:
# MAGIC 
# MAGIC - LightGBMClassifier: used for building classification models. For example, to predict whether a company will bankrupt or not, we could build a binary classification model with LightGBMClassifier.
# MAGIC - LightGBMRegressor: used for building regression models. For example, to predict the house price, we could build a regression model with LightGBMRegressor.
# MAGIC - LightGBMRanker: used for building ranking models. For example, to predict website searching result relevance, we could build a ranking model with LightGBMRanker.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bankruptcy Prediction with LightGBM Classifier
# MAGIC 
# MAGIC <img src="https://mmlspark.blob.core.windows.net/graphics/Documentation/bankruptcy image.png" width="800" style="float: center;"/>
# MAGIC 
# MAGIC In this example, we use LightGBM to build a classification model in order to predict bankruptcy.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read dataset

# COMMAND ----------

from pyspark.sql import SparkSession

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()

from synapse.ml.core.platform import *

from synapse.ml.core.platform import materializing_display as display

# COMMAND ----------

df = (
    spark.read.format("csv")
    .option("header", True)
    .option("inferSchema", True)
    .load(
        "wasbs://publicwasb@mmlspark.blob.core.windows.net/company_bankruptcy_prediction_data.csv"
    )
)
# print dataset size
print("records read: " + str(df.count()))
print("Schema: ")
df.printSchema()

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split the dataset into train and test

# COMMAND ----------

train, test = df.randomSplit([0.85, 0.15], seed=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Add featurizer to convert features to vector

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

feature_cols = df.columns[1:]
featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_data = featurizer.transform(train)["Bankrupt?", "features"]
test_data = featurizer.transform(test)["Bankrupt?", "features"]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check if the data is unbalanced

# COMMAND ----------

display(train_data.groupBy("Bankrupt?").count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Training

# COMMAND ----------

from synapse.ml.lightgbm import LightGBMClassifier

model = LightGBMClassifier(
    objective="binary", featuresCol="features", labelCol="Bankrupt?", isUnbalance=True
)

# COMMAND ----------

model = model.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC By calling "saveNativeModel", it allows you to extract the underlying lightGBM model for fast deployment after you train on Spark.

# COMMAND ----------

from synapse.ml.lightgbm import LightGBMClassificationModel

if running_on_synapse():
    model.saveNativeModel("/models/lgbmclassifier.model")
    model = LightGBMClassificationModel.loadNativeModelFromFile(
        "/models/lgbmclassifier.model"
    )
if running_on_synapse_internal():
    model.saveNativeModel("Files/models/lgbmclassifier.model")
    model = LightGBMClassificationModel.loadNativeModelFromFile(
        "Files/models/lgbmclassifier.model"
    )
else:
    model.saveNativeModel("/tmp/lgbmclassifier.model")
    model = LightGBMClassificationModel.loadNativeModelFromFile(
        "/tmp/lgbmclassifier.model"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Importances Visualization

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

feature_importances = model.getFeatureImportances()
fi = pd.Series(feature_importances, index=feature_cols)
fi = fi.sort_values(ascending=True)
f_index = fi.index
f_values = fi.values

# print feature importances
print("f_index:", f_index)
print("f_values:", f_values)

# plot
x_index = list(range(len(fi)))
x_index = [x / len(fi) for x in x_index]
plt.rcParams["figure.figsize"] = (20, 20)
plt.barh(
    x_index, f_values, height=0.028, align="center", color="tan", tick_label=f_index
)
plt.xlabel("importances")
plt.ylabel("features")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Prediction

# COMMAND ----------

predictions = model.transform(test_data)
predictions.limit(10).toPandas()

# COMMAND ----------

from synapse.ml.train import ComputeModelStatistics

metrics = ComputeModelStatistics(
    evaluationMetric="classification",
    labelCol="Bankrupt?",
    scoredLabelsCol="prediction",
).transform(predictions)
display(metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quantile Regression for Drug Discovery with LightGBMRegressor
# MAGIC 
# MAGIC <img src="https://mmlspark.blob.core.windows.net/graphics/Documentation/drug.png" width="800" style="float: center;"/>
# MAGIC 
# MAGIC In this example, we show how to use LightGBM to build a simple regression model.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read dataset

# COMMAND ----------

triazines = spark.read.format("libsvm").load(
    "wasbs://publicwasb@mmlspark.blob.core.windows.net/triazines.scale.svmlight"
)

# COMMAND ----------

# print some basic info
print("records read: " + str(triazines.count()))
print("Schema: ")
triazines.printSchema()
display(triazines.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split dataset into train and test

# COMMAND ----------

train, test = triazines.randomSplit([0.85, 0.15], seed=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Training

# COMMAND ----------

from synapse.ml.lightgbm import LightGBMRegressor

model = LightGBMRegressor(
    objective="quantile", alpha=0.2, learningRate=0.3, numLeaves=31
).fit(train)

# COMMAND ----------

print(model.getFeatureImportances())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Prediction

# COMMAND ----------

scoredData = model.transform(test)
display(scoredData)

# COMMAND ----------

from synapse.ml.train import ComputeModelStatistics

metrics = ComputeModelStatistics(
    evaluationMetric="regression", labelCol="label", scoresCol="prediction"
).transform(scoredData)
display(metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LightGBM Ranker

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read dataset

# COMMAND ----------

df = spark.read.format("parquet").load(
    "wasbs://publicwasb@mmlspark.blob.core.windows.net/lightGBMRanker_train.parquet"
)
# print some basic info
print("records read: " + str(df.count()))
print("Schema: ")
df.printSchema()
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Training

# COMMAND ----------

from synapse.ml.lightgbm import LightGBMRanker

features_col = "features"
query_col = "query"
label_col = "labels"
lgbm_ranker = LightGBMRanker(
    labelCol=label_col,
    featuresCol=features_col,
    groupCol=query_col,
    predictionCol="preds",
    leafPredictionCol="leafPreds",
    featuresShapCol="importances",
    repartitionByGroupingColumn=True,
    numLeaves=32,
    numIterations=200,
    evalAt=[1, 3, 5],
    metric="ndcg",
)

# COMMAND ----------

lgbm_ranker_model = lgbm_ranker.fit(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Prediction

# COMMAND ----------

dt = spark.read.format("parquet").load(
    "wasbs://publicwasb@mmlspark.blob.core.windows.net/lightGBMRanker_test.parquet"
)
predictions = lgbm_ranker_model.transform(dt)
predictions.limit(10).toPandas()
