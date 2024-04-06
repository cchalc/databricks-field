# Databricks notebook source
import pandas as pd
#import pyspark.pandas as pd

# import numpy as np
# import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.types import StructType, StructField
import json
import random
from typing import Iterator
import time
from datetime import datetime

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")

# If we want to control the batch size we can set the configuration parameter spark.sql.execution.arrow.maxRecordsPerBatch to the desired value when the spark session is created. This only affects the iterator like pandas UDFs and will apply even if we use one partition.
# spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 100000)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Simple Pandas Dataframe

# COMMAND ----------

sim_tpl_range_list = []
for i in range(0, 499999+1):
    sim_tpl_range_list.append(i)

sim_fo_range_list = []
for i in range(2, 3+1):
    sim_fo_range_list.append(i)

sim_tpl_range = {'sim_tpl': sim_tpl_range_list} 
sim_fo_range = {'sim_fo': sim_fo_range_list} 
df_sim_tpl = pd.DataFrame(sim_tpl_range)
df_sim_fo = pd.DataFrame(sim_fo_range) 
df_sim_tpl['key'] = 1
df_sim_fo['key'] = 1
df_sims = pd.merge(df_sim_tpl, df_sim_fo, on ='key').drop(columns='key')

# COMMAND ----------

#df_sims.memory_usage(deep=True).sum()/1000000

# COMMAND ----------

sdf_sims = spark.createDataFrame(df_sims)
display(sdf_sims)

# COMMAND ----------

sdf_sims.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the function to map (apply to each row)

# COMMAND ----------

def row_func(row):
    sum_var = row['sim_tpl'] + row['sim_fo']
    product_var = row['sim_tpl'] * row['sim_fo']
    return pd.Series([sum_var, product_var])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # mapInPandas usage
# MAGIC
# MAGIC https://docs.databricks.com/pandas/pandas-function-apis.html
# MAGIC

# COMMAND ----------

df = spark.createDataFrame([(1, 21), (2, 30)], ("id", "age"))

def filter_func(iterator):
    for pdf in iterator:
        yield pdf[pdf.id == 1]

df.mapInPandas(filter_func, schema=df.schema).show()
# +---+---+
# | id|age|
# +---+---+
# |  1| 21|
# +---+---+

# COMMAND ----------

# MAGIC %md
# MAGIC ### UDF 

# COMMAND ----------

def dispatch(frames):
    for frame in frames:
        frame['batch_num_rand'] = random.randint(1, 1000000)
        frame['batch_start_time'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        frame[['added_result','multiplied_result']] = frame.apply(lambda row: row_func(row), axis=1)
        frame['batch_end_time'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        yield frame
    ## ** BUG ** this yield the last frame in iterator so that's why your count doesn't match in the end.
    #yield frame

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Map - parallelized

# COMMAND ----------

return_sdf_schema = StructType.fromJson(json.loads(sdf_sims.schema.json()))
return_sdf_schema.add(StructField("added_result", T.DoubleType()))
return_sdf_schema.add(StructField("multiplied_result", T.DoubleType()))
return_sdf_schema.add(StructField("batch_num_rand", T.IntegerType()))
return_sdf_schema.add(StructField("batch_start_time", T.StringType()))
return_sdf_schema.add(StructField("batch_end_time", T.StringType()))

res_sdf = sdf_sims.mapInPandas(dispatch, schema=return_sdf_schema)

res_sdf.cache()
print(res_sdf.count())
print(len(res_sdf.columns))

assert res_sdf.count() == sdf_sims.count(), "Count don't match"

# COMMAND ----------


