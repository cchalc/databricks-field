# Databricks notebook source
import json
import os
import shutil

import pandas as pd

from subprocess import run, PIPE
from pdal import Pipeline
from glob import glob

from pyspark import Row
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data sourcing and pre-processing

# COMMAND ----------

# MAGIC %md
# MAGIC Example of downloading a single OS grid ref. chip of LIDAR Point Cloud data.

# COMMAND ----------

session = "ade4ccfb1358480cbec0a6cbe40ce9ce78628"
grid_refs = ["TQ36ne", "TQ36se", "TQ46nw", "TQ46sw"]

urls = [f"https://environment.data.gov.uk/UserDownloads/interactive/{session}/NLP/National-LIDAR-Programme-Point-Cloud-2018-{grid_ref}.zip" for grid_ref in grid_refs]
local_uris = [f"/tmp/National-LIDAR-Programme-Point-Cloud-2018-{grid_ref}.zip" for grid_ref in grid_refs]

for url, local_uri in zip(urls, local_uris):
  output_wget = run(['wget', '-O', local_uri, url], capture_output=False)
  print(output_wget)
  output_unzip = run(['unzip', '-d', '/tmp', local_uri])
  print(output_unzip)

# COMMAND ----------

# MAGIC %fs ls file:/tmp/P_10767

# COMMAND ----------

# DBTITLE 1,Move these downloaded LAZ files to DBFS
dbutils.fs.cp("file:/tmp/P_10767/", "/home/stuart@databricks.com/datasets/lidar/raw/", True)

# COMMAND ----------

# MAGIC %fs ls /home/stuart@databricks.com/datasets/lidar/raw/

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-processing
# MAGIC For demo purposes, we can divide these files into many smaller parts to facilitate parallel processing.

# COMMAND ----------

def dbfs_to_local(path: str) -> str:
  return f"dbfs{path}" if path[0] == "/" else f"dbfs/{path}"

def local_to_dbfs(path: str) -> str:
  return path.split("/dbfs")[-1]

# COMMAND ----------

# DBTITLE 1,A function that will split an input LAZ file into multiple smaller files (based on a path supplied in a Pandas DataFrame)
def split_laz(pdf: pd.DataFrame) -> pd.DataFrame:
  in_path = pdf.loc[0, "input_uri"]
  output_path_local = pdf.loc[0, "output_path"]
  output_filename_stem = os.path.splitext(os.path.basename(in_path))[0]
  
  os.makedirs(output_path_local, exist_ok=True)
  os.makedirs(f"/tmp/{output_filename_stem}", exist_ok=True)
  
  params = {
    "pipeline": [
      {"type": "readers.las", "filename": in_path},
      {"type":"filters.divider", "count": 50},
      {
        "type":"writers.las", 
        "compression": "lazperf",
        "filename": f"/tmp/{output_filename_stem}/{output_filename_stem}_#.laz"
      }
    ]}
  pipeline = Pipeline(json.dumps(params))
  pipeline.execute()
  for filename in glob(os.path.join(f"/tmp/{output_filename_stem}", '*.laz')):
    shutil.copy(filename, output_path_local)
  return pd.DataFrame([pdf["input_uri"], pd.Series(["OK"])])

# COMMAND ----------

# DBTITLE 1,Create my 'target' dataframe, with paths to files to original LAZ files
lidar_inputs = glob("/dbfs/home/stuart@databricks.com/datasets/lidar/raw/*.laz")
output_dir = "/dbfs/home/stuart@databricks.com/datasets/lidar/"

lidar_inputs_sdf = (
  spark.createDataFrame(
    [Row(pth, output_dir) for pth in lidar_inputs], 
    schema=StructType([
      StructField("input_uri", StringType()),
      StructField("output_path", StringType())
    ])
  )
)

lidar_inputs_sdf.display()

# COMMAND ----------

# DBTITLE 1,GroupBy + Apply execution of the function, in parallel across the cluster
split_results = lidar_inputs_sdf.groupBy("input_uri", "output_path").applyInPandas(split_laz, schema="result string")
split_results.write.csv("/home/stuart@databricks.com/datasets/lidar/split_log")

# COMMAND ----------

# MAGIC %fs ls /home/stuart@databricks.com/datasets/lidar
