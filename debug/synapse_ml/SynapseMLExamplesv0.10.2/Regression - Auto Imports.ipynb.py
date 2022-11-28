# Databricks notebook source
# MAGIC %md
# MAGIC ## Regression - Auto Imports
# MAGIC 
# MAGIC This sample notebook is based on the Gallery [Sample 6: Train, Test, Evaluate
# MAGIC for Regression: Auto Imports
# MAGIC Dataset](https://gallery.cortanaintelligence.com/Experiment/670fbfc40c4f44438bfe72e47432ae7a)
# MAGIC for AzureML Studio.  This experiment demonstrates how to build a regression
# MAGIC model to predict the automobile's price.  The process includes training, testing,
# MAGIC and evaluating the model on the Automobile Imports data set.
# MAGIC 
# MAGIC This sample demonstrates the use of several members of the synapseml library:
# MAGIC - [`TrainRegressor`
# MAGIC   ](https://mmlspark.blob.core.windows.net/docs/0.10.2/pyspark/synapse.ml.train.html?#module-synapse.ml.train.TrainRegressor)
# MAGIC - [`SummarizeData`
# MAGIC   ](https://mmlspark.blob.core.windows.net/docs/0.10.2/pyspark/synapse.ml.stages.html?#module-synapse.ml.stages.SummarizeData)
# MAGIC - [`CleanMissingData`
# MAGIC   ](https://mmlspark.blob.core.windows.net/docs/0.10.2/pyspark/synapse.ml.featurize.html?#module-synapse.ml.featurize.CleanMissingData)
# MAGIC - [`ComputeModelStatistics`
# MAGIC   ](https://mmlspark.blob.core.windows.net/docs/0.10.2/pyspark/synapse.ml.train.html?#module-synapse.ml.train.ComputeModelStatistics)
# MAGIC - [`FindBestModel`
# MAGIC   ](https://mmlspark.blob.core.windows.net/docs/0.10.2/pyspark/synapse.ml.automl.html?#module-synapse.ml.automl.FindBestModel)
# MAGIC 
# MAGIC First, import the pandas package so that we can read and parse the datafile
# MAGIC using `pandas.read_csv()`

# COMMAND ----------

from pyspark.sql import SparkSession

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

data = spark.read.parquet(
    "wasbs://publicwasb@mmlspark.blob.core.windows.net/AutomobilePriceRaw.parquet"
)

# COMMAND ----------

# MAGIC %md
# MAGIC To learn more about the data that was just read into the DataFrame,
# MAGIC summarize the data using `SummarizeData` and print the summary.  For each
# MAGIC column of the DataFrame, `SummarizeData` will report the summary statistics
# MAGIC in the following subcategories for each column:
# MAGIC * Feature name
# MAGIC * Counts
# MAGIC   - Count
# MAGIC   - Unique Value Count
# MAGIC   - Missing Value Count
# MAGIC * Quantiles
# MAGIC   - Min
# MAGIC   - 1st Quartile
# MAGIC   - Median
# MAGIC   - 3rd Quartile
# MAGIC   - Max
# MAGIC * Sample Statistics
# MAGIC   - Sample Variance
# MAGIC   - Sample Standard Deviation
# MAGIC   - Sample Skewness
# MAGIC   - Sample Kurtosis
# MAGIC * Percentiles
# MAGIC   - P0.5
# MAGIC   - P1
# MAGIC   - P5
# MAGIC   - P95
# MAGIC   - P99
# MAGIC   - P99.5
# MAGIC 
# MAGIC Note that several columns have missing values (`normalized-losses`, `bore`,
# MAGIC `stroke`, `horsepower`, `peak-rpm`, `price`).  This summary can be very
# MAGIC useful during the initial phases of data discovery and characterization.

# COMMAND ----------

from synapse.ml.stages import SummarizeData

summary = SummarizeData().transform(data)
summary.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Split the dataset into train and test datasets.

# COMMAND ----------

# split the data into training and testing datasets
train, test = data.randomSplit([0.6, 0.4], seed=123)
train.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Now use the `CleanMissingData` API to replace the missing values in the
# MAGIC dataset with something more useful or meaningful.  Specify a list of columns
# MAGIC to be cleaned, and specify the corresponding output column names, which are
# MAGIC not required to be the same as the input column names. `CleanMissiongData`
# MAGIC offers the options of "Mean", "Median", or "Custom" for the replacement
# MAGIC value.  In the case of "Custom" value, the user also specifies the value to
# MAGIC use via the "customValue" parameter.  In this example, we will replace
# MAGIC missing values in numeric columns with the median value for the column.  We
# MAGIC will define the model here, then use it as a Pipeline stage when we train our
# MAGIC regression models and make our predictions in the following steps.

# COMMAND ----------

from synapse.ml.featurize import CleanMissingData

cols = ["normalized-losses", "stroke", "bore", "horsepower", "peak-rpm", "price"]
cleanModel = (
    CleanMissingData().setCleaningMode("Median").setInputCols(cols).setOutputCols(cols)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will create two Regressor models for comparison: Poisson Regression
# MAGIC and Random Forest.  PySpark has several regressors implemented:
# MAGIC * `LinearRegression`
# MAGIC * `IsotonicRegression`
# MAGIC * `DecisionTreeRegressor`
# MAGIC * `RandomForestRegressor`
# MAGIC * `GBTRegressor` (Gradient-Boosted Trees)
# MAGIC * `AFTSurvivalRegression` (Accelerated Failure Time Model Survival)
# MAGIC * `GeneralizedLinearRegression` -- fit a generalized model by giving symbolic
# MAGIC   description of the linear preditor (link function) and a description of the
# MAGIC   error distribution (family).  The following families are supported:
# MAGIC   - `Gaussian`
# MAGIC   - `Binomial`
# MAGIC   - `Poisson`
# MAGIC   - `Gamma`
# MAGIC   - `Tweedie` -- power link function specified through `linkPower`
# MAGIC Refer to the
# MAGIC [Pyspark API Documentation](http://spark.apache.org/docs/latest/api/python/)
# MAGIC for more details.
# MAGIC 
# MAGIC `TrainRegressor` creates a model based on the regressor and other parameters
# MAGIC that are supplied to it, then trains data on the model.
# MAGIC 
# MAGIC In this next step, Create a Poisson Regression model using the
# MAGIC `GeneralizedLinearRegressor` API from Spark and create a Pipeline using the
# MAGIC `CleanMissingData` and `TrainRegressor` as pipeline stages to create and
# MAGIC train the model.  Note that because `TrainRegressor` expects a `labelCol` to
# MAGIC be set, there is no need to set `linkPredictionCol` when setting up the
# MAGIC `GeneralizedLinearRegressor`.  Fitting the pipe on the training dataset will
# MAGIC train the model.  Applying the `transform()` of the pipe to the test dataset
# MAGIC creates the predictions.

# COMMAND ----------

# train Poisson Regression Model
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml import Pipeline
from synapse.ml.train import TrainRegressor

glr = GeneralizedLinearRegression(family="poisson", link="log")
poissonModel = TrainRegressor().setModel(glr).setLabelCol("price").setNumFeatures(256)
poissonPipe = Pipeline(stages=[cleanModel, poissonModel]).fit(train)
poissonPrediction = poissonPipe.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, repeat these steps to create a Random Forest Regression model using the
# MAGIC `RandomRorestRegressor` API from Spark.

# COMMAND ----------

# train Random Forest regression on the same training data:
from pyspark.ml.regression import RandomForestRegressor

rfr = RandomForestRegressor(maxDepth=30, maxBins=128, numTrees=8, minInstancesPerNode=1)
randomForestModel = TrainRegressor(model=rfr, labelCol="price", numFeatures=256).fit(
    train
)
randomForestPipe = Pipeline(stages=[cleanModel, randomForestModel]).fit(train)
randomForestPrediction = randomForestPipe.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC After the models have been trained and scored, compute some basic statistics
# MAGIC to evaluate the predictions.  The following statistics are calculated for
# MAGIC regression models to evaluate:
# MAGIC * Mean squared error
# MAGIC * Root mean squared error
# MAGIC * R^2
# MAGIC * Mean absolute error
# MAGIC 
# MAGIC Use the `ComputeModelStatistics` API to compute basic statistics for
# MAGIC the Poisson and the Random Forest models.

# COMMAND ----------

from synapse.ml.train import ComputeModelStatistics

poissonMetrics = ComputeModelStatistics().transform(poissonPrediction)
print("Poisson Metrics")
poissonMetrics.toPandas()

# COMMAND ----------

randomForestMetrics = ComputeModelStatistics().transform(randomForestPrediction)
print("Random Forest Metrics")
randomForestMetrics.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC We can also compute per instance statistics for `poissonPrediction`:

# COMMAND ----------

from synapse.ml.train import ComputePerInstanceStatistics


def demonstrateEvalPerInstance(pred):
    return (
        ComputePerInstanceStatistics()
        .transform(pred)
        .select("price", "prediction", "L1_loss", "L2_loss")
        .limit(10)
        .toPandas()
    )


demonstrateEvalPerInstance(poissonPrediction)

# COMMAND ----------

# MAGIC %md
# MAGIC and with `randomForestPrediction`:

# COMMAND ----------

demonstrateEvalPerInstance(randomForestPrediction)
