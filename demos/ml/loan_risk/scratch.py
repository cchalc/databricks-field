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

df_clean = spark.read.table("cjc.loan_risk.lendingclub_clean")
display(df_clean)

# COMMAND ----------

# Setting variables to predict bad loans
myY = "bad_loan"
categoricals = ["term", "home_ownership", "purpose", "addr_state","verification_status","application_type"]
numerics = ["loan_amnt", "emp_length", "annual_inc", "dti", "delinq_2yrs", "revol_util", "total_acc", "credit_length_in_years"]
myX = categoricals + numerics

# COMMAND ----------

df_train = spark.read.table("cjc.loan_risk.train_data")
display(df_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's add the MoE/IV calculations to the training dataset
# MAGIC References:
# MAGIC - [weight of evidence and information value summary](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F
from pyspark.ml.feature import QuantileDiscretizer

def iv_woe_spark(data, target, bins=10, show_woe=False):
    # Empty DataFrames
    newDF = pd.DataFrame(columns=["Variable", "IV"])
    woeDF = pd.DataFrame(columns=["Variable", "Cutoff", "N", "Events", "% of Events", "Non-Events", "% of Non-Events", "WoE", "IV"])
    
    # Extract Column Names
    cols = [col for col in data.columns if col != target]
    
    for ivars in cols:
        if data.schema[ivars].dataType in ['integer', 'double'] and data.select(ivars).distinct().count() > 10:
            # Bin the data for continuous variables using QuantileDiscretizer
            discretizer = QuantileDiscretizer(numBuckets=bins, inputCol=ivars, outputCol=ivars+"_bin")
            datatemp = data.select(ivars, target)
            datatemp2 = discretizer.fit(datatemp).transform(datatemp)
            d0 = datatemp2.select(ivars+"_bin", target)
        else:
            d0 = data.select(ivars, target)
            d0 = d0.withColumnRenamed(ivars, ivars+"_bin")

        # Calculate the number of events in each group (bin)
        d = d0.groupBy(ivars+"_bin").agg(
            F.sum(target).alias('Events'),
            F.count(target).alias('N')
        )
        
        # Events
        total_events = d.select(F.sum('Events')).collect()[0][0]
        d = d.withColumn('% of Events', F.greatest(F.col('Events'), F.lit(0.5)) / total_events)

        # Non-Events
        d = d.withColumn('Non-Events', F.col('N') - F.col('Events'))
        total_non_events = d.select(F.sum('Non-Events')).collect()[0][0]
        d = d.withColumn('% of Non-Events', F.greatest(F.col('Non-Events'), F.lit(0.5)) / total_non_events)

        # WOE / IV
        d = d.withColumn('WoE', F.log(F.col('% of Events') / F.col('% of Non-Events')))
        d = d.withColumn('IV', (F.col('% of Events') - F.col('% of Non-Events')) * F.col('WoE'))
        d = d.withColumn('Variable', F.lit(ivars))
        
        # Compute IV for this variable and print it
        iv_value = d.agg(F.sum('IV')).collect()[0][0]
        # print("Information value of " + ivars + " is " + str(round(iv_value, 6)))
        
        # Append IV and WoE data to woeDF
        temp_woe = d.toPandas()
        woeDF = pd.concat([woeDF, temp_woe], axis=0, ignore_index=True)
        
        # Append IV to newDF
        temp_iv = pd.DataFrame([(ivars, iv_value)], columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp_iv], axis=0, ignore_index=True)
        
        # Show WoE Table
        if show_woe:
            print(temp_woe)

    return newDF, woeDF

# newDF, woeDF = iv_woe_spark(data, target, bins=10, show_woe=True)

# COMMAND ----------

df_new, df_woe = iv_woe_spark(df_train, "bad_loan", bins=10, show_woe=True)

# COMMAND ----------

display(df_woe)
