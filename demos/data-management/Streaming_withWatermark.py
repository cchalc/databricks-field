# Databricks notebook source
# DBTITLE 1,Getting the data, emptying directories, casting a timestamp.
#imports
import pandas as pd
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from random import random

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
dbutils.fs.rm("dbfs:/home/{user}/demo/source-data".format(user=user), True)
dbutils.fs.rm("dbfs:/home/{user}/demo/checkpoint/".format(user=user), True)
dbutils.fs.rm("dbfs:/home/{user}/demo/auto-loader".format(user=user), True)
dbutils.fs.rm("dbfs:/home/{user}/demo/copy-into".format(user=user), True)

#set up copy into path and remove folder
copyInto = "/ml/covid_us_states_delta/copyinto"
dbutils.fs.rm(copyInto, recurse=True)

#Read in data
url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
dfpd = pd.read_csv(url)

#Convert Pandas dataframe to Spark dataframe
df = spark.createDataFrame(dfpd)
df = (df
      .withColumn("date1",df['date']
      .cast(TimestampType())) # Watermark needs a timestamp column
      .drop("date")
      .withColumnRenamed("date1","date")
      .select("date", "state", "fips", "cases", "deaths")
     )

df1, df2, df3, df4 = df.randomSplit([0.25, 0.25, 0.25, 0.25])

#add label column to each dataframe
df1 = df1.withColumn("label", lit('df1'))
df2 = df2.withColumn("label", lit('df2'))
df3 = df3.withColumn("label", lit('df3'))
df4 = df4.withColumn("label", lit('df4'))

print("userid:", user)
print("df1 rows:", df1.count())
print("df2 rows:", df2.count())
print("df3 rows:", df3.count())
print("total rows:", df1.count() + df2.count() + df3.count())

# COMMAND ----------

# DBTITLE 1,Write data to GCS / S3 directory
df1.write.parquet('dbfs:/home/{user}/demo/source-data/df1'.format(user=user,random=random()))
display(dbutils.fs.ls("dbfs:/home/{user}/demo/source-data/".format(user=user)))

# COMMAND ----------

# DBTITLE 1,Autoloader Stream, apply watermark to drop duplicates
df_write = (spark.readStream.format("cloudFiles") \
            .option("cloudFiles.validateOptions", "false") \
            .option("cloudFiles.format", "parquet") \
            .option("cloudFiles.region", "us-west-2") \
            .option("cloudFiles.includeExistingFiles", "true") \
            .schema(df1.schema) \
            .load("dbfs:/home/{user}/demo/source-data".format(user=user))
         )

df_write \
  .withWatermark("date", "10 Seconds") \
  .dropDuplicates(["date", "state"])\
  .writeStream \
  .format("delta") \
  .option("checkpointLocation", "dbfs:/home/{user}/demo/checkpoint/".format(user=user)) \
  .start("dbfs:/home/{user}/demo/auto-loader".format(user=user))

# COMMAND ----------

# DBTITLE 1,Write more data to GCS / S3 directory
df2.write.parquet('dbfs:/home/{user}/demo/source-data/df2'.format(user=user,random=random()))
df3.write.parquet('dbfs:/home/{user}/demo/source-data/df3'.format(user=user,random=random()))
display(dbutils.fs.ls("dbfs:/home/{user}/demo/source-data/".format(user=user)))
