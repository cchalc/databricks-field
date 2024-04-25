# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *

# import dlt


lspq_path = "/databricks-datasets/samples/lending_club/parquet/"

# COMMAND ----------

df = spark.read.parquet(lspq_path)
display(df)

# COMMAND ----------

spark.sql("use catalog cjc")
spark.sql("use schema loan_risk")

# COMMAND ----------

# Get list of tables in the loan_risk schema
display(spark.sql("SHOW TABLES IN cjc.loan_risk"))

# COMMAND ----------

# df_clean = spark.read.table("cjc.loan_risk.lendingclub_clean")
# display(df_clean)

# COMMAND ----------

# MAGIC %sql use schema scratch
# MAGIC -- Query to create scratch data here: https://e2-demo-field-eng.cloud.databricks.com/sql/editor/32fb948f-cc0e-44d1-b4d7-e8001b77bc76?o=1444828305810485

# COMMAND ----------

df_train = spark.read.table("cjc.scratch.loan_risk_train")
display(df_train)

# COMMAND ----------

# Setting variables to predict bad loans
myY = "bad_loan"
categoricals = ["term", "home_ownership", "purpose", "addr_state","verification_status","application_type"]
numerics = ["loan_amnt", "emp_length", "annual_inc", "dti", "delinq_2yrs", "revol_util", "total_acc", "credit_length_in_years"]
myX = categoricals + numerics

# COMMAND ----------

df_train.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's add the MoE/IV calculations to the training dataset
# MAGIC References:
# MAGIC - [weight of evidence and information value summary](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
# MAGIC - [Woe_IV_EDA_V2.1](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#notebook/880375545542772/command/880375545542784) <-- client notebook
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# Function to identify categorical and numerical variables
def identify_variable_types(df, exclude_cols):
    categorical_vars = [field.name for field in df.schema.fields if field.dataType == StringType() and field.name not in exclude_cols]
    numerical_vars = [field.name for field in df.schema.fields if (field.dataType == IntegerType() or field.dataType == DoubleType()) and field.name not in exclude_cols]
    return categorical_vars, numerical_vars

# WoE and IV Calculation for a given feature
def calculate_woe_iv(df, feature, variable, target, var_type):
    total_events = df.filter(col(target) == 1).count()
    total_non_events = df.filter(col(target) == 0).count()
        
    # Calculate counts and events per category/bucket
    agg_df = df.groupBy(feature).agg(F.count("*").alias("total"), F.sum(target).alias("events"))
    agg_df = agg_df.withColumn("non_events", col("total") - col("events"))

    # Calculate WoE and IV
    agg_df = agg_df.withColumn("dist_event", (col("events") / (total_events+total_non_events))) # total_events)) # col("total")))
    agg_df = agg_df.withColumn("dist_non_event", (col("non_events") / (total_events+total_non_events))) # col("total")))
    agg_df = agg_df.withColumn("woe", log(col("dist_event") / col("dist_non_event")))
    agg_df = agg_df.withColumn("iv", (col("dist_event") - col("dist_non_event")) * col("woe"))
    agg_df = agg_df.withColumn("variable", F.lit(variable))
    agg_df = agg_df.withColumn("variable_type", F.lit(var_type))
    agg_df = agg_df.withColumn("target", F.lit(target))
    
    # Handle any division by zero or log of zero issues by replacing with zeros
    agg_df = agg_df.na.fill({'woe': 0, 'iv': 0})

    agg_df = agg_df.withColumnRenamed(feature,"bin")

    
    return agg_df.agg(F.sum("iv").alias("IV")), agg_df, total_events, total_non_events
    #return total_df, agg_df, total_events, total_non_events

# Calculate the aggegations of the WoE dataset
def aggegate_woe_iv(df):
    agg_df = df.groupBy("target", "variable")\
        .agg(F.sum("total").alias("total"),\
             F.avg("dist_event").alias("events"),
             F.avg("dist_non_event").alias("non_events"),
             F.avg("woe").alias("WOE"),
             F.sum("iv").alias("IV")
                 )
    return agg_df

def elapsed_time(start_time, finish_time) -> str:
    elapsed = finish_time - start_time
    return time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:15], time.gmtime(elapsed))
    
# print("Elapsed time: " + time.strftime("%H:%M:%S.{}".format(str(elapsed_time % 1)[2:])[:15], time.gmtime(elapsed_time)))

# COMMAND ----------

target_variable_names = ["bad_loan"]
num_of_bins = 10
exclude_cols = ["bad_loan"]
categorical_vars, numerical_vars = identify_variable_types(df_train, exclude_cols)

# COMMAND ----------

print(f"numerical vars: {numerical_vars}, categorical vars: {categorical_vars}")

# COMMAND ----------

## See: https://github.com/marshackVB/parallel_models_blog/blob/master/parallel_models_udf.py#L195 
# from typing import List
# from pyspark.ml.feature import ColumnTransformer, QuantileDiscretizer, SimpleImputer
# from pyspark.ml import Pipeline
# from pyspark.ml.pipeline import PipelineModel

# def create_preprocessing_transform(categorical_features: List[str], numerical_features: List[str], num_of_bins: int, var: str) -> ColumnTransformer:
#   """
#   Create a preprocessing pipeline for a given variable
#   """
  
#   categorical_pipe = Pipeline(
#         [
#             ("discretizer", QuantileDiscretizer(numBuckets=num_of_bins, inputCol=var, outputCol=var+"_bin"))
#         ]
#     )

#   numerical_pipe_quantile = Pipeline(
#         [
#             ("imputer", SimpleImputer(strategy="median"))
#         ]
#     )

#   preprocessing = ColumnTransformer(
#         [
#             ("categorical", categorical_pipe,        categorical_features),
#             ("numeric",     numerical_pipe_quantile, numerical_features)
#         ],
#         remainder='drop'
#     )

#   return preprocessing

# COMMAND ----------

schema = StructType([StructField('bin', DoubleType(), True),
            StructField('total', LongType(), False),
            StructField('events', DoubleType(), True),
            StructField('non_events', DoubleType(), True),
            StructField('dist_event', DoubleType(), True),
            StructField('dist_non_event', DoubleType(), True),
            #StructField('perc_non_event', DoubleType(), True),
            StructField('woe', DoubleType(), False),
            StructField('iv', DoubleType(), False),
            StructField('variable', StringType(), False),
            StructField('variable_type', StringType(), False),
            StructField('target', StringType(), False)])

schema_Total = StructType([StructField('variable', StringType(), False),
                         StructField('total_events', DoubleType(), True),
                         StructField('total_non_events', LongType(), True),
                         StructField('IV', DoubleType(), False),
                         StructField('WOE', DoubleType(), True)])

# COMMAND ----------

result_df = spark.createDataFrame([], schema)


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, log
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml import Pipeline

import time
start_time = time.time()


# COMMAND ----------

# Result Dataframe
# Looping the target variables
for target_variable in target_variable_names:
    # Discretize numerical variables and calculate WoE/IV
    for var in numerical_vars:
        discretizer = QuantileDiscretizer(
            numBuckets=num_of_bins, inputCol=var, outputCol=var + "_bin"
        )
        df = discretizer.fit(df_train).transform(df_train)
        print(
            "Calculating WoE & IV - Target: {} - Feature: {}".format(
                target_variable, var
            )
        )
        iv_df, woe_df, total_events, total_non_events = calculate_woe_iv(
            df, var + "_bin", var, target_variable, "numerical"
        )
        result_df = result_df.union(woe_df)

    # For categorical variables, directly calculate WoE/IV
    for var in categorical_vars:
        print(
            "Calculating WoE & IV - Target: {} - Feature: {}".format(
                target_variable, var
            )
        )
        iv_df, woe_df, total_events, total_non_events = calculate_woe_iv(
            df_train, var, var, target_variable, "categorical"
        )
        result_df = result_df.union(woe_df)

# COMMAND ----------

display(result_df)

# COMMAND ----------

# MAGIC %md ### Try with config DF

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, DoubleType, LongType, StringType

schema = StructType([
    StructField('bin', DoubleType(), True),
    StructField('total', LongType(), False),
    StructField('events', DoubleType(), True),
    StructField('non_events', DoubleType(), True),
    StructField('dist_event', DoubleType(), True),
    StructField('dist_non_event', DoubleType(), True),
    StructField('woe', DoubleType(), False),
    StructField('iv', DoubleType(), False),
    StructField('variable', StringType(), False),
    StructField('variable_type', StringType(), False),
    StructField('target', StringType(), False)
])

result_df = spark.createDataFrame([], schema)

result_df.write.format("delta").mode("overwrite").saveAsTable("loan_risk_results")

# COMMAND ----------

from pyspark.sql.functions import explode, array, lit

numerical_df = spark.createDataFrame([(tv, nv, "numerical") for tv in target_variable_names for nv in numerical_vars], ["target_variable", "variable", "type"])
categorical_df = spark.createDataFrame([(tv, cv, "categorical") for tv in target_variable_names for cv in categorical_vars], ["target_variable", "variable", "type"])

config_df = numerical_df.union(categorical_df)


# COMMAND ----------

display(config_df)

# COMMAND ----------

config_pdf = config_df.toPandas()
config_pdf.head()

# COMMAND ----------

# So all the workers have access
df_train.cache()

# COMMAND ----------

broadcast_function = spark.sparkContext.broadcast(calculate_woe_iv)

# COMMAND ----------

pdf_train = df_train.toPandas()

# COMMAND ----------

pdf_train.head()

# COMMAND ----------

import pandas as pd
import numpy as np

def calculate_woe_iv_pandas(df, feature, variable, target, var_type):
    total_events = df[df[target] == 1].shape[0]
    total_non_events = df[df[target] == 0].shape[0]

    # Calculate counts and events per category/bucket
    agg_df = df.groupby(feature).agg(total=('target', 'size'),
                                     events=(target, 'sum')).reset_index()
    agg_df['non_events'] = agg_df['total'] - agg_df['events']

    # Calculate WoE and IV
    agg_df['dist_event'] = agg_df['events'] / (total_events + total_non_events)
    agg_df['dist_non_event'] = agg_df['non_events'] / (total_events + total_non_events)
    agg_df['woe'] = np.log(agg_df['dist_event'] / agg_df['dist_non_event'].replace(0, np.nan))
    agg_df['iv'] = (agg_df['dist_event'] - agg_df['dist_non_event']) * agg_df['woe']
    agg_df['variable'] = variable
    agg_df['variable_type'] = var_type
    agg_df['target'] = target

    # Handle any division by zero or log of zero issues by replacing with zeros
    agg_df['woe'] = agg_df['woe'].replace([np.inf, -np.inf, np.nan], 0)
    agg_df['iv'] = agg_df['iv'].replace([np.inf, -np.inf, np.nan], 0)

    agg_df = agg_df.rename(columns={feature: "bin"})

    # Calculate total IV for the variable
    total_iv = agg_df['iv'].sum()

    return total_iv, agg_df, total_events, total_non_events


# COMMAND ----------

import pandas as pd

def apply_woe(config_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function applies the WOE transformation to the given config_df.
    :param config_df: A pandas DataFrame containing the configuration parameters.
    :return: A pandas DataFrame containing the WOE transformation parameters.
    """
    # calculate_woe_iv = broadcast_function.value
    # from pyspark.ml.feature import QuantileDiscretizer
    
    target_variable = config_df['target_variable'].iloc[0]
    var = config_df['variable'].iloc[0]
    var_type = config_df['type'].iloc[0]
    num_of_bins = 10
    
    if var_type == 'categorical':
      iv_df, woe_df, total_events, total_non_events = calculate_woe_iv_pandas(
        pdf_train, var, var, target_variable, "categorical"
      )
      # woe_pdf = woe_df.toPandas()
      return woe_df
    
    # elif var_type == 'numerical':
    #   discretizer = QuantileDiscretizer(
    #     numBuckets=num_of_bins, inputCol=var, outputCol=var + "_bin"
    #     )
    #   df = discretizer.fit(df_train).transform(df_train)
    #   iv_df, woe_df, total_events, total_non_events = calculate_woe_iv(
    #     df, var + "_bin", var, target_variable, "numerical"
    #     )
    #   woe_pdf = woe_df.toPandas()
    #   return woe_pdf
    
    else:
      print("Invalid variable type")
      return None



# COMMAND ----------

res_df = (config_df
          .groupby("target_variable")
          .applyInPandas(apply_woe, schema=schema)
)

# COMMAND ----------

res_df.show()

# COMMAND ----------


tmp_df = calculate_woe_iv_pandas(pdf_train, "home_ownership", "home_ownership", "bad_loan", "categorical")

# COMMAND ----------


