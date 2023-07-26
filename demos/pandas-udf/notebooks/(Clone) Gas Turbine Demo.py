# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Demo

# COMMAND ----------

# MAGIC %pip install dbldatagen
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data simulator
# MAGIC
# MAGIC 1 millions records

# COMMAND ----------

from datetime import timedelta, datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import count, when, isnan, col, lit, countDistinct
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType, LongType

import dbldatagen as dg

interval = timedelta(days=1, hours=1)
start = datetime(2017, 10, 1, 0, 0, 0)
end = datetime(2018, 10, 1, 6, 0, 0)

schema = StructType([
    StructField("site_id", IntegerType(), True),
    StructField("site_cd", StringType(), True),
    StructField("sector_status_desc", StringType(), True),
    StructField("c", BooleanType(), True),
    StructField("s1", LongType(), True),
    StructField("s2", LongType(), True),
    StructField("s3", LongType(), True),
    StructField("s4", LongType(), True),

])

# will have implied column `id` for ordinal of row
x3 = (dg.DataGenerator(sparkSession=spark, name="gaz_turbine_bronze", rows=1000000, partitions=20)
      .withSchema(schema)
      # withColumnSpec adds specification for existing column
      .withColumnSpec("site_id", minValue=1, maxValue=20, step=1)
      # base column specifies dependent column
      .withIdOutput()
      .withColumnSpec("site_cd", prefix='site', baseColumn='site_id')
      .withColumnSpec("s1", minValue=1, maxValue=200, random=True)
      .withColumnSpec("s2", minValue=1, maxValue=200, random=True)
      .withColumnSpec("s3", minValue=1, maxValue=200, random=True)
      .withColumnSpec("s4", minValue=1, maxValue=200, random=True)
      .withColumn("sector_status_desc", "string", minValue=1, maxValue=200, step=1, prefix='status', random=True)
      # withColumn adds specification for new column
      .withColumn("rand", "float", expr="floor(rand() * 350) * (86400 + 3600)")
      .withColumn("last_sync_dt", "timestamp", begin=start, end=end, interval=interval, random=True)
      )

x3_output = x3.build(withTempView=True)

print(x3_output.count())
x3_output.printSchema()
display(x3_output)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Generate value on all records using Spark withColumn
# MAGIC
# MAGIC Spark is able to do arithmetic operations on columns using withColumn.

# COMMAND ----------

calcDF = (x3_output
  .withColumn("sum_s1_s4", col("s1") + col("s4"))
  .withColumn("mul_s1_s4", col("s1") * col("s4"))
)

display(calcDF)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Call user defined function
# MAGIC
# MAGIC If you have more complex logic to apply to column(s), you can use a udf instead.
# MAGIC
# MAGIC You could have multiple UDF that focus on a specific metrics. In other words, you don't need a big UDF that tries to do everything at once. divide-and-conquer

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf

# Imports
import pyspark.pandas as ps
import numpy as np


# COMMAND ----------

# create pandas_udf
@pandas_udf(LongType())
def dispatch(s1: pd.Series, s2: pd.Series, s3: pd.Series) -> pd.Series:
    return s1 + s2 * s3

# COMMAND ----------

display(x3_output
  .withColumn("complex_logic", dispatch("s1", "s2", "s3"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Use apply function from Pyspark Pandas
# MAGIC
# MAGIC If you still need to work on all columns at a time, you can leverage Pandas on PySpark and use the apply function.

# COMMAND ----------

from pyspark.sql.types import FloatType

psdf = x3_output.select("s1", "s3").pandas_api()

def division(data):
  return data[0] / data[1]
   
divideDF = psdf.apply(division, axis=1)

df = ps.merge(psdf, divideDF, left_index=True, right_index=True, how='left')

display(df)

# COMMAND ----------


