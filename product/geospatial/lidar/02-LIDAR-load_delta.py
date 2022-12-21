# Databricks notebook source
import json

import pandas as pd

from pdal import Pipeline
from glob import glob

from pyspark import Row
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extracting point cloud data from 'laz' format files

# COMMAND ----------

# DBTITLE 1,Create my 'target' dataframe, with paths to files to original LAZ files
lidar_inputs = glob("/dbfs/home/stuart@databricks.com/datasets/lidar/*.laz")

lidar_inputs_sdf = (
  spark.createDataFrame(
    [Row(pth) for pth in lidar_inputs], 
    schema=StructType([StructField("path", StringType())])
  )
)

lidar_inputs_sdf.display()

# COMMAND ----------

# DBTITLE 1,Function to extract raw data as a numpy array and return in a Pandas DF
def read_laz(pdf: pd.DataFrame) -> pd.DataFrame:
  
  # extract filename for PDAL
  fl = pdf.loc[0, "path"]
  
  # PDAL pipeline, simplest case
  params = {"pipeline": [{"type": "readers.las", "filename": fl}]}
  
  pipeline = Pipeline(json.dumps(params))
  
  # get points + sensor parameters from file and generate Pandas dataframe
  arr_size = pipeline.execute()
  arr = pipeline.arrays[0]
  output_pdf = pd.DataFrame(arr)
  
  # grab the file metadata
  metadata = json.loads(pipeline.metadata)["metadata"]["readers.las"]
  
  # parse out useful metadata and store inline with points
  output_pdf["_minXYZ"] = json.dumps({"minX": metadata["minx"], "minY": metadata["miny"], "minZ":  metadata["minx"]})
  output_pdf["_maxXYZ"] = json.dumps({"maxX": metadata["maxx"], "maxY": metadata["maxy"], "maxZ":  metadata["maxx"]})
  output_pdf["_horizontalCRS"] = json.dumps(metadata["srs"]["horizontal"])
  output_pdf["_verticalCRS"] = json.dumps(metadata["srs"]["vertical"])
  output_pdf["_file_path"] = fl
  
  # return dataframe
  return output_pdf

# COMMAND ----------

# DBTITLE 1,GroupBy + Apply execution of the function, in parallel across the cluster
lidar_schema = StructType([
  StructField("X", FloatType()),
  StructField("Y", FloatType()),
  StructField("Z", FloatType()),
  StructField("Intensity", IntegerType()),
  StructField("ReturnNumber", IntegerType()),
  StructField("NumberOfReturns", IntegerType()),
  StructField("ScanDirectionFlag", IntegerType()),
  StructField("EdgeOfFlightLine", IntegerType()),
  StructField("Classification", IntegerType()),
  StructField("ScanAngleRank", FloatType()),
  StructField("UserData", IntegerType()),
  StructField("PointSourceId", IntegerType()),
  StructField("GpsTime", DoubleType()),
  StructField("Red", IntegerType()),
  StructField("Green", IntegerType()),
  StructField("Blue", IntegerType()),
  StructField("_minXYZ", StringType()),
  StructField("_maxXYZ", StringType()),
  StructField("_horizontalCRS", StringType()),
  StructField("_verticalCRS", StringType()),
  StructField("_file_path", StringType())
])

# Apply my python code to a Spark dataframe
lidar_data_sdf = (
  lidar_inputs_sdf
  .groupBy("path")
  .applyInPandas(read_laz, schema=lidar_schema)
)

# COMMAND ----------

# MAGIC %md
# MAGIC What's going on here?
# MAGIC - Task and data are serialized and sent to worker nodes
# MAGIC - Workers allocate the task to a free executor
# MAGIC - Executor runs commands in Python interpreter, with data sent / returned between JVM and Python using the Apache Arrow standard.
# MAGIC - Spark's execution's model is lazy: nothing happens unless we call an action like 'write' on the DataFrame.

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists stuart.lidar_raw

# COMMAND ----------

spark.conf.set("spark.sql.adaptive.enabled", "false")

# COMMAND ----------

lidar_data_sdf.write.format("delta").mode("overwrite").saveAsTable("stuart.lidar_raw")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) as points from stuart.lidar_raw

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from stuart.lidar_raw

# COMMAND ----------

# MAGIC %sql
# MAGIC optimize stuart.lidar_raw zorder by (X, Y, Z)
