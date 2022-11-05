# Databricks notebook source
# MAGIC %md
# MAGIC # Workspace Analysis
# MAGIC 
# MAGIC The goal of this notebook is to gather workspace info and provide insights.
# MAGIC 
# MAGIC Notes
# MAGIC - please run on DBR 8+
# MAGIC - user running notebook should be admin for best results
# MAGIC - may want to switch View -> Results Only
# MAGIC - open outline view to help navigate (the arrow on the left of the notebook)

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

# setup ----------
import requests
import json
import mlflow
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql.types import DataType, StructType, ArrayType
from copy import copy
# We take a dataframe and return a new one with required changes

def cleanDataFrame(df):
    # Returns a new sanitized field name (this function can be anything really)
    def sanitizeFieldName(s):
        return s.replace(":", "_").replace(" ", "_").replace("-", "_").replace("&", "_").replace("\"", "_")\
            .replace("[", "_").replace("]", "_").replace(".", "_")
    
    # We call this on all fields to create a copy and to perform any 
    # changes we might want to do to the field.
    def sanitizeField(field):
        field = copy(field)
        field.name = sanitizeFieldName(field.name)
        # We recursively call cleanSchema on all types
        field.dataType = cleanSchema(field.dataType)
        return field
    
    def cleanSchema(dataType):
        dataType = copy(dataType)
        # If the type is a StructType we need to recurse otherwise 
        # we can return since we've reached the leaf node
        if isinstance(dataType, StructType):
            # We call our sanitizer for all top level fields
            dataType.fields = [sanitizeField(f) for f in dataType.fields]
        elif isinstance(dataType, ArrayType):
            dataType.elementType = cleanSchema(dataType.elementType)
        return dataType

    # Now since we have the new schema we can create a new DataFrame 
    # by using the old Frame's RDD as data and the new schema as the 
    # schema for the data
    return spark.createDataFrame(df.rdd, cleanSchema(df.schema))

token = dbutils.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.entry_point.getDbutils().notebook().getContext().apiUrl().get()
auth = {'Authorization': 'Bearer {}'.format(token), 'Content-Type': 'text/json'}
cloud = "AZURE" if "azuredatabricks" in host else "AWS"
spark.sql("CREATE DATABASE IF NOT EXISTS workspace_analysis")
spark.sql("USE workspace_analysis")

def fetch_and_save_from_API(api_name, json_element_name, table_name, simpleList=False):
  try:
    api_response = requests.get('{}/api/2.0/{}'.format(host, api_name), headers=auth)
    mydf = ''
    # simpleList is for SCIM Groups
    if (simpleList): 
      z = list(map(lambda x: Row(name=x), api_response.json()['group_names']))
      rdd = sc.parallelize(z)
      mydf = cleanDataFrame(spark.createDataFrame(rdd))
    # all other APIs
    else:
      rdd = sc.parallelize(api_response.json()[json_element_name]).map(lambda x: json.dumps(x)) 
      mydf = cleanDataFrame(spark.read.json(rdd))

    (mydf
      .write
      .mode("overwrite")
      .option("mergeSchema", "true")
      .saveAsTable("workspace_analysis.{}".format(table_name))
    )
  except:
    print("Was not able to run {}".format(api_name))
    pass

def sql(sqltext):
  try:
    return display(spark.sql(sqltext))
  except:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC # Clusters

# COMMAND ----------

fetch_and_save_from_API('clusters/list', 'clusters', 'clusters')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto-termination
# MAGIC Clusters without auto-termination, or longer than 2 hours. 0 means no autotermination
# MAGIC 
# MAGIC ### Concept
# MAGIC If a cluster is set with auto termination of X minutes. And said cluster does not see any cluster activity (Spark jobs, Structured Streaming, JDBC calls) for a period equal to or greater than X minutes between the current time and the last cluster activity, said cluster will be shutdown and deprovisioned. While deprovisioned, the cluster won't be able to run notebooks or jobs - but its configuration will be stored so that an exact copy of it can be reused the next time the cluster starts.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Except in those cases where a cluster NEEDS to be on 24/7, it is a good practice to configure the cluster with auto-termination.
# MAGIC 
# MAGIC Non terminated clusters will continue to accumulate DBU and cloud instance charges, incurring in avoidable costs.
# MAGIC 
# MAGIC Auto termination is triggered by a period of absence of cluster commands but NOT by commands run by SSH or bash.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Management Guide ([AWS](https://docs.databricks.com/clusters/clusters-manage.html#terminate-a-cluster)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/clusters-manage#--terminate-a-cluster))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The following analysis shows:
# MAGIC - Clusters with no auto termination - autotermination_minutes = 0.
# MAGIC - Clusters with auto termination set to longer than 2 hours - autotermination_minutes > 2.

# COMMAND ----------

sql("""
  SELECT cluster_id, cluster_name, autotermination_minutes, state, cluster_cores, from_unixtime(start_time / 1000,'yyyy-MM-dd HH:mm:ss') as start_date
  FROM clusters
  WHERE (autotermination_minutes = 0 OR autotermination_minutes > 120) AND cluster_name NOT LIKE "job-%-run-%"
  ORDER BY state ASC, start_date DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## DBR Version
# MAGIC 
# MAGIC ### Concept
# MAGIC Databricks Runtime is the set of software artifacts that run on the clusters of machines managed by Databricks. It includes Spark but also adds a number of components and updates that substantially improve the usability, performance, and security of big data analytics. The primary differentiations are:
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Databricks recommends to use Long Term Support (LTS) versions of runtimes for production jobs.
# MAGIC 
# MAGIC Databricks releases runtimes as Beta and GA versions. Databricks supports GA versions for six months, unless the runtime version is:
# MAGIC - A Long Term Support (LTS) version. LTS versions are released every six months and supported for two full years.
# MAGIC - A Databricks Light runtime. Light versions are supported until either 12 months after release or two months after the next Databricks Light release, whichever comes first.
# MAGIC Workloads on unsupported runtime versions may continue to run, but they receive no Databricks support or fixes.
# MAGIC 
# MAGIC ### Documentation
# MAGIC User Guide ([AWS](https://docs.databricks.com/runtime/index.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/runtime/))  
# MAGIC Runtime Releases ([AWS](https://docs.databricks.com/release-notes/runtime/releases.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/release-notes/runtime/releases))  
# MAGIC Runtime Support Lifecycle ([AWS](https://docs.databricks.com/release-notes/runtime/databricks-runtime-ver.html#runtime-support)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/release-notes/runtime/databricks-runtime-ver#runtime-support))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The following analysis shows the list of unsuccessfully terminated cluster in the last 150 days.

# COMMAND ----------

sql("""
  SELECT spark_version, count(*)
  FROM clusters
  GROUP BY 1
  ORDER BY spark_version DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Non-SUCCESS Terminations
# MAGIC 
# MAGIC ### Concept
# MAGIC 
# MAGIC ### Best Practice
# MAGIC 
# MAGIC ### Documentation
# MAGIC (AWS) (Azure)
# MAGIC 
# MAGIC ### Analysis
# MAGIC The following analysis shows the list of unsuccessfully terminated cluster in the last 150 days.

# COMMAND ----------

sql("""
  SELECT cluster_id, cluster_name, from_unixtime(terminated_time / 1000,'yyyy-MM-dd HH:mm:ss') as terminated_time, state_message, termination_reason
  FROM clusters 
  WHERE termination_reason.type <> "SUCCESS"
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spot analysis
# MAGIC 
# MAGIC ### Concept
# MAGIC A spot instance is at its simplest, an unused asset of the cloud provider. Cloud providers offer deep discounts when customers use these spot instances based on the region, VM type, etc.
# MAGIC 
# MAGIC At Databricks spot instances are used in the form of executor nodes belonging to a Databricks cluster. A Databricks cluster can leverage any combination of on-demand and spot instances. 
# MAGIC 
# MAGIC A node will be assigned to a spot instance as long as the customer’s provided spot bid amount is greater or equal to the Cloud provider hourly-evaluated spot market price. If at any point while the cluster is running the spot bid amount falls below the spot market price, the node assigned to spot instances  will be automatically terminated.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Spot instances can potentially reduce the compute cost of your workloads up to 90%*. One drawback being that instances are reclaimed if the spot bid price falls below the spot market price.
# MAGIC 
# MAGIC Good candidates for spot instances are those with loose SLA’s as spot instances may be reclaimed by the cloud provider at any moment, potentially incurring in failed tasks.
# MAGIC Notes
# MAGIC 
# MAGIC ### Documentation
# MAGIC Company Blog ([AWS](https://databricks.com/blog/2016/10/25/running-apache-spark-clusters-with-spot-instances-in-databricks.html)) ([Azure](https://databricks.com/blog/2021/05/25/leverage-unused-compute-capacity-for-data-ai-with-azure-spot-instances-and-azure-databricks.html))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The following analysis shows a list of clusters and their availability (on demand or spot).

# COMMAND ----------

if cloud == "AZURE":
  sql(""""
    SELECT cluster_id, cluster_name, azure_attributes.availability
    FROM clusters
    ORDER BY availability
  """)
elif cloud == "AWS":
  sql("""
    SELECT cluster_id, cluster_name, aws_attributes.availability
    FROM clusters 
    ORDER BY availability
  """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Autoscaling Jobs
# MAGIC You may want to turn off autoscaling for production jobs
# MAGIC 
# MAGIC ### Concept
# MAGIC Scaling a cluster is the processing of adding/removing executor nodes in a cluster based on the workload, typically to optimize for the utilization of resources such as processing power or memory.
# MAGIC 
# MAGIC In Databricks, clusters can be configured to scale automatically based on a periodical report of detailed statistics given by each of its executors. 
# MAGIC - Downscaling: If the statistics show that executor nodes have been idle, a cluster will release and deprovision those nodes marked as idle - this will occur after making sure that their removal won’t affect current or downstream tasks, as idle clusters may contain partitions necessary for other tasks to complete.
# MAGIC - Upscaling: If the statistics show that executors are approaching a preset limit, a cluster will demand more executors which will be provisioned without interrupting the current job.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Autoscaling tends to work best for high-concurrency interactive clusters where the demand of resources fluctuates and is not easily predictable. Since Databricks can precisely target workers for scale-down under low utilization, clusters can be resized much more aggressively in response to load. This keeps wasted compute resources to a minimum while also maintaining the responsiveness of the cluster. Similarly, if a spike in demand occurs, a Databricks cluster can react and increase its number of executors to keep responsiveness high without sacrificing efficiency.
# MAGIC 
# MAGIC For production jobs, autoscaling is usually a poor choice since this loads are predictable and can be better managed with a specifically tailored cluster configuration. Additionally, adding clusters can take time which can impact SLAs.
# MAGIC 
# MAGIC Autoscaling in Databricks is different from traditional autoscaling for other services as the scaling can occur in the middle of a job rather than at the boundaries between jobs.
# MAGIC 
# MAGIC ### Documentation
# MAGIC User Guide ([AWS](https://docs.databricks.com/clusters/configure.html#autoscaling)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/configure#autoscaling))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The following analysis contains a list of jobs where autoscale has been configured.

# COMMAND ----------

sql("""
  SELECT * FROM clusters
  WHERE autoscale IS NOT NULL
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Photon
# MAGIC 
# MAGIC ### Concept
# MAGIC Photon is a new execution engine (entirely written in C++) on the Databricks Lakehouse platform that provides extremely fast query performance at low cost for SQL workloads, directly on your data lake. With Photon, most analytics workloads can meet or exceed data warehouse performance without actually moving any data into a data warehouse.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC While the new engine is designed to accelerate all workloads, during preview, Photon is focused on running SQL workloads faster, while reducing your total cost per workload. Ultimately, Photon will support all data and machine learning use cases as well.
# MAGIC 
# MAGIC Photon activation depends on whether you are using Databricks clusters or Databricks SQL endpoints.
# MAGIC - Databricks clusters: Photon can be activated by selecting a runtime containing Photon.
# MAGIC - Databricks SQL endpoints: Photon is enabled by default.
# MAGIC 
# MAGIC ### Documentation
# MAGIC User Guide ([AWS](https://docs.databricks.com/runtime/photon.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/runtime/photon))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The following two analyses show the existence of photon clusters in your workspace.
# MAGIC 1. The first one shows the amount of clusters or jobs that either used (is_photon) or didn’t use (not_photon) Photon in the last 150 days.
# MAGIC 2. The second one shows the name of the clusters or jobs that either used (is_photon) or didn’t use (not_photon) Photon in the last 150 days.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Are you using photon?

# COMMAND ----------

sql("""
  SELECT photon_status, count(*) as count
  FROM (
    SELECT
    CASE WHEN spark_version LIKE "%photon%" THEN "is_photon" ELSE "not_photon" END as photon_status
    FROM clusters
  )
  GROUP BY photon_status
""")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Photon Clusters or Jobs

# COMMAND ----------

sql("""
  SELECT
    CASE WHEN default_tags.RunName IS NULL THEN cluster_name ELSE default_tags.RunName END as cluster_or_job_name,
    CASE WHEN default_tags.RunName IS NULL THEN False ELSE True END as is_job
  FROM clusters
  WHERE spark_version LIKE "%photon%"
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instance Types
# MAGIC 
# MAGIC ### Concept
# MAGIC Databricks clusters consist of one driver node and zero to many executor nodes. These nodes are behind the scenes Virtual Machines (VMs) that the cloud provider provisions on demand. Each of these VMs incur different costs per hour of usage according to their resources.
# MAGIC 
# MAGIC There are many types of Virtual Machines but they are often split into families according to their characteristics:
# MAGIC - General purpose: An overall balanced VM.
# MAGIC - Memory optimized: Good for tasks that require to load big files/partitions in a node.
# MAGIC - Compute optimized: Good for those tasks that can be distributed across multiple CPUs.
# MAGIC - GPU: Good for applications that require intensive and vectorized machine learning and data processing.
# MAGIC - Photon: Special instances that are enabled to run with Databricks’ Photon engine.
# MAGIC Within these types, one can find multiple choices that coincide with specific combinations of CPUs and memory (some even have SSD).
# MAGIC 
# MAGIC ### Best Practice
# MAGIC The best practice is to understand the workloads that the cluster will have to process as well as what their SLAs are and what the budget constraint is. An ETL job with Spark will have different requirements than the training of a Neural Network or a single-node interactive exploratory data analysis.
# MAGIC 
# MAGIC In general, few powerful VMs are better than several smaller VMs, the cost of the most expensive instances  tends to even out as the execution times are lowered.
# MAGIC 
# MAGIC It is really important to distinguish Virtual Machine costs versus Databrick Unit costs.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Best Practices ([AWS](https://docs.databricks.com/clusters/configure.html#cluster-node-type)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/configure#--cluster-node-type))  
# MAGIC VM Costs ([AWS](https://databricks.com/product/aws-pricing/instance-types)) ([Azure](https://azure.microsoft.com/en-us/pricing/details/databricks/#instance-type-support))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The following analysis shows the distribution of instance types in the workspace. To understand the costs.

# COMMAND ----------

sql("""
  SELECT instance_source.node_type_id, count(*) as count
  FROM clusters 
  GROUP BY 1
  ORDER BY count DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Init Scripts
# MAGIC 
# MAGIC ### Concept
# MAGIC An init script is a shell script that runs during startup of each cluster node before the Apache Spark driver or worker JVM starts.
# MAGIC 
# MAGIC Some examples of tasks performed by init scripts include:
# MAGIC - Install packages and libraries not included in Databricks Runtime. 
# MAGIC - Modify the JVM system classpath in special cases.
# MAGIC - Set system properties and environment variables used by the JVM.
# MAGIC - Modify Spark configuration parameters.
# MAGIC 
# MAGIC There are two kind of init scripts:
# MAGIC - Cluster-scoped: Which run on clusters that are explicitly configured to do so.
# MAGIC - Global: Which run on every cluster in the workspace.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Init scripts are a great way to standardize your clusters with company/department-wide libraries, packages and variables.
# MAGIC 
# MAGIC ### Documentation
# MAGIC User Guide ([AWS](https://docs.databricks.com/clusters/init-scripts.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/init-scripts))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The following analysis shows a list of the init scripts in the workspace as well as which clusters use them (if any).

# COMMAND ----------

sql("""
  SELECT init_scripts, collect_set(cluster_id) as cluster_ids
  FROM clusters
  GROUP BY 1
  ORDER BY init_scripts DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom Tags
# MAGIC 
# MAGIC ### Concept
# MAGIC To monitor cost and accurately attribute Databricks usage to your organization’s business units and teams (for chargebacks, for example), you can tag clusters and pools. These tags propagate both to DBU usage reports and compute cost analysis reports provided by your cloud service.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC It is a good practice to add tags regarding the Cost Center utilizing the cluster as well as the department, initiative or project behind it. That way reports can be fine-grained.
# MAGIC 
# MAGIC Tag keys and values can contain only ISO 8859-1 (latin1) characters, any non-recognized character will be ignored.
# MAGIC 
# MAGIC Tag propagation can take some time and will only be reflected after cluster restart.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Admin Guide ([AWS](https://docs.databricks.com/administration-guide/account-settings/usage-detail-tags-aws.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/administration-guide/account-settings/usage-detail-tags-azure))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The following analysis shows the custom tags that have been added to your workspace and the values that these have.

# COMMAND ----------

import re

try:
  str = spark.table("clusters").select("custom_tags").dtypes[0][1]
  str_2 = re.search("<(.*)>", str)[1]
  tags = list(filter(None, re.split(":string,?", str_2)))
  unpivotExpr = "stack({}, {}) as (value, tag)".format(len(tags), ", ".join(["custom_tags.`{}`, '{}'".format(x, x) for x in tags ]))
  tags_df = spark.table("clusters").select(F.expr(unpivotExpr))
except:
  pass

# COMMAND ----------

# MAGIC %md
# MAGIC #### Number of Clusters with Tag

# COMMAND ----------

try:
  (tags_df
   .filter(col("tag") != "ResourceClass")
   .filter(col("value").isNotNull())
   .groupBy("tag")
   .agg(F.count("tag").alias("count"))
   .orderBy(col("count").desc())
   .display()
  )
except:
  pass

# COMMAND ----------

# MAGIC %md
# MAGIC #### Values Per Tag

# COMMAND ----------

try:
  (tags_df
  .filter(col("tag") != "ResourceClass")
  .groupBy("tag")
  .agg(F.collect_set("value").alias("values"))
  .display())
except:
  pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Configuration
# MAGIC 
# MAGIC ### Concept
# MAGIC 
# MAGIC ### Best Practice
# MAGIC 
# MAGIC ### Documentation
# MAGIC (AWS) (Azure)
# MAGIC 
# MAGIC ### Analysis

# COMMAND ----------

sql("""
  SELECT cluster_log_conf, collect_set(cluster_id) as cluster_ids
  FROM clusters
  GROUP BY 1
  ORDER BY 1 DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single-Node and High-Concurrency
# MAGIC - **Single-Node**: Useful when there isn't a lot of data involved, or you're training with single-node ML algorithms
# MAGIC - **High-Concurrency** (displays as *Serverless*):  Good for shared clusters or used for BI. Not recommended for ML or Jobs
# MAGIC 
# MAGIC ### Concept
# MAGIC There are three cluster modes provided by Databricks - Single-Node/Standard/High Concurrency. The choice of which cluster mode to choose highly depends on the type of workloads to be run by it:
# MAGIC - **Standard clusters** are ideal for processing large amounts of data with Apache Spark.
# MAGIC - **Single Node clusters** are intended for jobs that use small amounts of data or non-distributed workloads such as single-node machine learning libraries.
# MAGIC - **High Concurrency clusters** are ideal for groups of users who need to share resources or run ad-hoc jobs. Administrators usually create High Concurrency clusters. Databricks recommends enabling autoscaling for High Concurrency clusters.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC To reduce cost, it is a best practice to use the right cluster mode for the right workload, this will ensure an optimal use of resources.
# MAGIC Standard clusters tend to be the go-to for scheduled massively parallel jobs as their resources are not shared with any other workloads.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Single-Node Clusters ([AWS](https://docs.databricks.com/clusters/single-node.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/single-node))  
# MAGIC Best Practices ([AWS](https://docs.databricks.com/clusters/cluster-config-best-practices.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/cluster-config-best-practices))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The analysis shows the cluster types and the cluster id’s belonging to each one.

# COMMAND ----------

sql("""
  SELECT custom_tags.ResourceClass, collect_set(cluster_id) as cluster_ids
  FROM clusters
  WHERE custom_tags.ResourceClass IS NOT NULL
  GROUP BY custom_tags.ResourceClass
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Most workers
# MAGIC 
# MAGIC ### Concept
# MAGIC A cluster consists of one driver node and zero or more worker nodes.
# MAGIC 
# MAGIC You can pick separate cloud provider instance types for the driver and worker nodes, although by default the driver node uses the same instance type as the worker node. Different families of instance types fit different use cases, such as memory-intensive or compute-intensive workloads.
# MAGIC - Driver Node: The driver node maintains state information of all notebooks attached to the cluster. The driver node also maintains the SparkContext and interprets all the commands you run from a notebook or a library on the cluster, and runs the Apache Spark master that coordinates with the Spark executors.
# MAGIC - Worker Node: Databricks worker nodes run the Spark executors and other services required for the proper functioning of the clusters. When you distribute your workload with Spark, all of the distributed processing happens on worker nodes. Databricks runs one executor per worker node; therefore the terms executor and worker are used interchangeably in the context of the Databricks architecture.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Choose the number and instance type of the worker nodes based on the type of workload that the cluster will run. While having lots of worker nodes and a beefier instance type will result in an inefficient DBU consumption, having few nodes and an instance type that lacks resources will translate into longer processing times at best or runtime errors as a worst case scenario.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Worker Nodes ([AWS](https://docs.databricks.com/clusters/configure.html#cluster-node-type)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/configure#--cluster-node-type))  
# MAGIC Cluster Sizing ([AWS](https://docs.databricks.com/clusters/configure.html#cluster-size-and-autoscaling)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/configure#cluster-size-and-autoscaling))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The analysis outputs a list of clusters ordered by the number of worker nodes.

# COMMAND ----------

sql("""
  SELECT cluster_id, cluster_name, default_tags.RunName as job_name, num_workers
  FROM clusters 
  ORDER BY num_workers DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML Runtime
# MAGIC 
# MAGIC ### Concept
# MAGIC Databricks Runtime for Machine Learning (Databricks Runtime ML) automates the creation of a cluster optimized for machine learning. Databricks Runtime ML clusters include the most popular machine learning libraries, such as TensorFlow, PyTorch, Keras, and XGBoost, and also include libraries required for distributed training such as Horovod. Using Databricks Runtime ML speeds up cluster creation and ensures that the installed library versions are compatible.
# MAGIC 
# MAGIC Databricks Runtime ML is built on Databricks Runtime. For example, Databricks Runtime 7.3 LTS for Machine Learning is built on Databricks Runtime 7.3 LTS. The libraries included in the base Databricks Runtime are listed in the Databricks Runtime release notes.
# MAGIC 
# MAGIC For applications that require intensive and vectorized machine learning and data processing, it is possible to use a Databricks ML Runtime for GPU.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC It is far more practical to use a DBR ML than downloading and installing each of the libraries needed for a Data Science/Machine Learning workload. By using the ML runtimes, you skip the time needed to download and install these libraries as well as issues with compatibility.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Databricks Runtime release notes ([AWS](https://docs.databricks.com/release-notes/runtime/releases.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/release-notes/runtime/releases))  
# MAGIC Databricks Runtime ML ([AWS](https://docs.databricks.com/runtime/mlruntime.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/runtime/mlruntime))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The analysis shows two lists:
# MAGIC - one with all clusters using the CPU DBR ML
# MAGIC - another shows all of the clusters using GPU DBR ML.

# COMMAND ----------

# MAGIC %md
# MAGIC #### CPU MLR

# COMMAND ----------

sql("""
  SELECT cluster_id, cluster_name
  FROM clusters
  WHERE spark_version LIKE "%cpu%"
""")

# COMMAND ----------

# MAGIC %md
# MAGIC #### GPU MLR

# COMMAND ----------

sql("""
  SELECT cluster_id, cluster_name
  FROM clusters
  WHERE spark_version LIKE "%gpu%"
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Jobs

# COMMAND ----------

my_json = """[{"job_id": 1, "settings": {"name": "aviation_edge_job", "new_cluster": {"spark_version": "7.0.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/alex.barreto@databricks.com/Customers/GS/Aviation Edge/aviation_edge_bronze"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1597247620883, "creator_user_name": "alex.barreto@databricks.com"}, {"job_id": 2, "settings": {"name": "Example Job", "new_cluster": {"spark_version": "7.0.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/andrew.hombach@databricks.com/Quickstart Notebook"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1598274667886, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 5, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1604950647334, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 6, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1604954587397, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 7, "settings": {"name": "Example Job", "new_cluster": {"spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1604954589880, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 8, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1605205086567, "creator_user_name": "spencer.mcghin@databricks.com"}, {"job_id": 9, "settings": {"name": "New Job", "new_cluster": {"spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1605220056054, "creator_user_name": "spencer.mcghin@databricks.com"}, {"job_id": 10, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "0 0 0 * * ?", "timezone_id": "US/Pacific", "pause_status": "UNPAUSED"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1605220109946, "creator_user_name": "spencer.mcghin@databricks.com"}, {"job_id": 13, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1605662442263, "creator_user_name": "chris.grant@databricks.com"}, {"job_id": 17, "settings": {"name": "Test job", "new_cluster": {"spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1606928075067, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 18, "settings": {"name": "test job", "new_cluster": {"spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1606928133450, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 19, "settings": {"name": "Test", "new_cluster": {"spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1606928219851, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 27, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1607033893324, "creator_user_name": "tj.cycyota@databricks.com"}, {"job_id": 28, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1607034106605, "creator_user_name": "tj.cycyota@databricks.com"}, {"job_id": 32, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1607089949203, "creator_user_name": "julian.pereda@databricks.com"}, {"job_id": 37, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK", "spot_bid_price_percent": 100, "ebs_volume_type": "GENERAL_PURPOSE_SSD", "ebs_volume_count": 3, "ebs_volume_size": 100}, "node_type_id": "m4.10xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "autoscale": {"min_workers": 2, "max_workers": 5}}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/julian.pereda@databricks.com/2020-10-23 EDL PIMS TWAVG ( tmp_delta)"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1607162053850, "creator_user_name": "julian.pereda@databricks.com"}, {"job_id": 45, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1607532957893, "creator_user_name": "julian.pereda@databricks.com"}, {"job_id": 46, "settings": {"name": "Rapid Start Job", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1607544998861, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 47, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1607545113785, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 55, "settings": {"name": "aviation_edge_job", "new_cluster": {"spark_version": "7.0.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/alex.barreto@databricks.com/Customers/GS/Aviation Edge/aviation_edge_bronze"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1608578970514, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 56, "settings": {"name": "Example Job", "new_cluster": {"spark_version": "7.0.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/andrew.hombach@databricks.com/Quickstart Notebook"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1608579336752, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 57, "settings": {"name": "aviation_edge_job", "new_cluster": {"spark_version": "7.0.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/alex.barreto@databricks.com/Customers/GS/Aviation Edge/aviation_edge_bronze"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1608579336887, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 58, "settings": {"name": "Example Job", "new_cluster": {"spark_version": "7.0.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/andrew.hombach@databricks.com/Quickstart Notebook"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1608579457313, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 59, "settings": {"name": "aviation_edge_job", "new_cluster": {"spark_version": "7.0.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/alex.barreto@databricks.com/Customers/GS/Aviation Edge/aviation_edge_bronze"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1608579457463, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 61, "settings": {"name": "MJM", "new_cluster": {"cluster_name": "", "spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 0, "availability": "SPOT", "instance_profile_arn": "arn:aws:iam::826763667205:instance-profile/cse2_s3Access", "spot_bid_price_percent": 100, "ebs_volume_type": "GENERAL_PURPOSE_SSD", "ebs_volume_count": 3, "ebs_volume_size": 100}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 1}, "email_notifications": {"on_start": ["mohan.mathews@databricks.com"], "on_success": ["mohan.mathews@databricks.com"], "on_failure": ["mohan.mathews@databricks.com"], "no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "0 0 8 * * ?", "timezone_id": "US/Eastern", "pause_status": "UNPAUSED"}, "notebook_task": {"notebook_path": "/Users/mohan.mathews@databricks.com/TestNB"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1609874491012, "creator_user_name": "mohan.mathews@databricks.com"}, {"job_id": 62, "settings": {"name": "example_job", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1609940922729, "creator_user_name": "luke.watkins@databricks.com"}, {"job_id": 63, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1609953647911, "creator_user_name": "luke.watkins@databricks.com"}, {"job_id": 64, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/andrew.hombach@databricks.com/Quickstart Notebook"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1610470160983, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 65, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1610621186784, "creator_user_name": "luke.watkins@databricks.com"}, {"job_id": 66, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1610690989567, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 67, "settings": {"name": "DevSales Run", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1611092466915, "creator_user_name": "spencer.mcghin@databricks.com"}, {"job_id": 68, "settings": {"name": "new job", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1611222949918, "creator_user_name": "luke.watkins@databricks.com"}, {"job_id": 69, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1611225246653, "creator_user_name": "luke.watkins@databricks.com"}, {"job_id": 70, "settings": {"name": "Test", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/andrew.hombach@databricks.com/2020-11-09 - Amazon Kinesis Example"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1611246722391, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 71, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1611247580331, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 72, "settings": {"name": "Test", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/andrew.hombach@databricks.com/Rapid Starts/General Rapid Start/Notebook functionality"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1611248354499, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 73, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.4.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/andrew.hombach@databricks.com/Quickstart Notebook"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1611248455568, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 74, "settings": {"name": "testjoblei", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/lei.pan@databricks.com/testjob"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1611771702685, "creator_user_name": "lei.pan@databricks.com"}, {"job_id": 75, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1611786843605, "creator_user_name": "tj.cycyota@databricks.com"}, {"job_id": 76, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1611830992275, "creator_user_name": "luke.watkins@databricks.com"}, {"job_id": 81, "settings": {"name": "test_jar_Adriel", "new_cluster": {"spark_version": "6.4.x-scala2.11", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK", "spot_bid_price_percent": 100, "ebs_volume_count": 0}, "node_type_id": "i3.xlarge", "enable_elastic_disk": false, "num_workers": 2}, "email_notifications": {}, "timeout_seconds": 0, "spark_jar_task": {"jar_uri": "b9deeaff_440b_46c1_90e7_122d73e069e8-logging_app_for_databricks_assembly_0_1.jar", "main_class_name": "Main", "run_as_repl": true}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1612464880526, "creator_user_name": "tj.cycyota@databricks.com"}, {"job_id": 82, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1612487427771, "creator_user_name": "dylan.gessner@databricks.com"}, {"job_id": 83, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1612544107122, "creator_user_name": "dylan.gessner@databricks.com"}, {"job_id": 84, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1612544110140, "creator_user_name": "dylan.gessner@databricks.com"}, {"job_id": 87, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/julian.pereda@databricks.com/test"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1612952034151, "creator_user_name": "julian.pereda@databricks.com"}, {"job_id": 88, "settings": {"name": "Tester_bwz_jam", "existing_cluster_id": "0129-172120-copse746", "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/brandon.williams@databricks.com/bw_tester"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1613070317369, "creator_user_name": "brandon.williams@databricks.com"}, {"job_id": 89, "settings": {"name": "ETL test", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1613081218261, "creator_user_name": "jeff.welch@databricks.com"}, {"job_id": 90, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK", "spot_bid_price_percent": 100, "ebs_volume_count": 0}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1613387743600, "creator_user_name": "kunal.gaurav@databricks.com"}, {"job_id": 91, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1613402007167, "creator_user_name": "anestis.pontikakis@databricks.com"}, {"job_id": 93, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1613512856814, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 95, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1613663222122, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 98, "settings": {"name": "Demo Job", "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "MULTI_TASK"}, "created_time": 1614588032798, "creator_user_name": "antony.chia@databricks.com"}, {"job_id": 107, "settings": {"name": "kunaldb", "new_cluster": {"cluster_name": "", "spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK", "spot_bid_price_percent": 100, "ebs_volume_count": 0}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/kunal.gaurav@databricks.com/Rapid Start Template"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1614931941399, "creator_user_name": "kunal.gaurav@databricks.com"}, {"job_id": 108, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1615199826437, "creator_user_name": "luke.watkins@databricks.com"}, {"job_id": 109, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1615340453579, "creator_user_name": "debora.reis@databricks.com"}, {"job_id": 110, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1615340455019, "creator_user_name": "debora.reis@databricks.com"}, {"job_id": 117, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK", "spot_bid_price_percent": 100, "ebs_volume_count": 0}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1615501271819, "creator_user_name": "sharon@databricks.com"}, {"job_id": 118, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1615818081097, "creator_user_name": "andrew.hombach@databricks.com"}, {"job_id": 121, "settings": {"name": "Untitled", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1616090361922, "creator_user_name": "luke.watkins@databricks.com"}, {"job_id": 125, "settings": {"name": "my_job", "new_cluster": {"spark_version": "7.5.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK", "spot_bid_price_percent": 100, "ebs_volume_count": 0}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Repos/Production/NotebookGitTest/Integration_Test"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1616704382543, "creator_user_name": "tj.cycyota@databricks.com"}, {"job_id": 135, "settings": {"name": "aarp-alter-table-scratch", "new_cluster": {"spark_version": "8.0.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "0 0 0 1 * ?", "timezone_id": "US/Pacific", "pause_status": "UNPAUSED"}, "notebook_task": {"notebook_path": "/Users/justin.boyd@databricks.com/aarp-alter-table-scratch"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1617721189484, "creator_user_name": "justin.boyd@databricks.com"}, {"job_id": 140, "settings": {"name": "Quickstart Notebook", "new_cluster": {"spark_version": "8.1.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "0 0 0 1 * ?", "timezone_id": "US/Central", "pause_status": "UNPAUSED"}, "notebook_task": {"notebook_path": "/Users/rich.hauser@databricks.com/Quickstart Notebook"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1618947854376, "creator_user_name": "rich.hauser@databricks.com"}, {"job_id": 142, "settings": {"name": "Rakeshtest", "new_cluster": {"cluster_name": "", "spark_version": "7.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK", "spot_bid_price_percent": 100, "ebs_volume_count": 0}, "node_type_id": "i3.xlarge", "enable_elastic_disk": false, "num_workers": 1}, "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/prakash.jha@databricks.com/notebookurl"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1619185634870, "creator_user_name": "rakesh.parija@databricks.com"}, {"job_id": 144, "settings": {"name": "dbricks_etl_loadagetable", "new_cluster": {"spark_version": "8.1.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK", "spot_bid_price_percent": 100, "ebs_volume_type": "GENERAL_PURPOSE_SSD", "ebs_volume_count": 3, "ebs_volume_size": 100}, "node_type_id": "m4.large", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 1}, "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "24 19 9 * * ?", "timezone_id": "UTC", "pause_status": "PAUSED"}, "notebook_task": {"notebook_path": "/Users/shashank.kava@databricks.com/demo-etl-notebook", "base_parameters": {"colName": "age"}}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1620048226321, "creator_user_name": "shashank.kava@databricks.com"}, {"job_id": 148, "settings": {"name": "Matt Test Job", "existing_cluster_id": "0505-133941-lures497", "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/matthew.walding@databricks.com/Quickstart Notebook"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1620222013675, "creator_user_name": "matthew.walding@databricks.com"}, {"job_id": 197, "settings": {"name": "test-job", "existing_cluster_id": "0524-233136-knee223", "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "37 32 16 * * ?", "timezone_id": "UTC", "pause_status": "PAUSED"}, "notebook_task": {"notebook_path": "/Users/dylan.gessner@databricks.com/upload data"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1621899188209, "creator_user_name": "dylan.gessner@databricks.com"}, {"job_id": 207, "settings": {"name": "Random", "new_cluster": {"spark_version": "8.2.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "0 0 6 1 * ?", "timezone_id": "US/Pacific", "pause_status": "UNPAUSED"}, "notebook_task": {"notebook_path": "/Users/mohan.mathews@databricks.com/Random"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1621952569069, "creator_user_name": "mohan.mathews@databricks.com"}, {"job_id": 255, "settings": {"name": "dlt", "new_cluster": {"spark_version": "8.2.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "0 0 * * * ?", "timezone_id": "US/Pacific", "pause_status": "UNPAUSED"}, "notebook_task": {"notebook_path": "/Users/alexander.desroches@databricks.com/dlt"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1622047817370, "creator_user_name": "alexander.desroches@databricks.com"}, {"job_id": 355, "settings": {"name": "Ceneviva Test", "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "max_concurrent_runs": 1, "format": "MULTI_TASK"}, "created_time": 1622736886099, "creator_user_name": "nick.ceneviva@databricks.com"}, {"job_id": 558, "settings": {"name": "Workflow", "existing_cluster_id": "0602-213030-finds122", "email_notifications": {"on_failure": ["mohan.mathews@databricks.com"], "no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "0 0 9 * * ?", "timezone_id": "US/Eastern", "pause_status": "UNPAUSED"}, "notebook_task": {"notebook_path": "/Users/mohan.mathews@databricks.com/Travelers/Data Engineering/Pipeline/solutions/Workflow"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1623289008813, "creator_user_name": "mohan.mathews@databricks.com"}, {"job_id": 597, "settings": {"name": "Rapid Start", "existing_cluster_id": "0602-213030-finds122", "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "0 0 10 * * ?", "timezone_id": "US/Pacific", "pause_status": "UNPAUSED"}, "notebook_task": {"notebook_path": "/Users/mohan.mathews@databricks.com/Travelers/General Rapid Start/Rapid Start"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1623289609820, "creator_user_name": "mohan.mathews@databricks.com"}, {"job_id": 5009, "settings": {"name": "Tester", "new_cluster": {"cluster_name": "", "spark_version": "8.2.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK", "spot_bid_price_percent": 100, "ebs_volume_count": 0}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "34 9 12 * * ?", "timezone_id": "UTC", "pause_status": "PAUSED"}, "notebook_task": {"notebook_path": "/Repos/brandon.williams@databricks.com/cse_bw/End_to_End_Distributed_ML_MLOps_with_MLflow"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1623957112527, "creator_user_name": "brandon.williams@databricks.com"}, {"job_id": 9489, "settings": {"name": "BW_2", "new_cluster": {"spark_version": "8.2.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK", "spot_bid_price_percent": 100, "ebs_volume_count": 0}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/brandon.williams@databricks.com/bw_tester"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1624781429262, "creator_user_name": "brandon.williams@databricks.com"}, {"job_id": 9564, "settings": {"name": "BW_3", "new_cluster": {"spark_version": "8.2.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK", "spot_bid_price_percent": 100, "ebs_volume_count": 0}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/brandon.williams@databricks.com/bw_tester"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1624781449596, "creator_user_name": "brandon.williams@databricks.com"}, {"job_id": 12583, "settings": {"name": "nl_dlt_demo", "email_notifications": {"on_failure": ["nicole@databricks.com"], "no_alert_for_skipped_runs": true}, "timeout_seconds": 0, "pipeline_task": {"pipeline_id": "38a672ad-9875-4bf2-ba63-b3ff25e62ca3"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1625244426924, "creator_user_name": "jingting.lu@databricks.com"}, {"job_id": 15443, "settings": {"name": "Quickstart Notebook", "new_cluster": {"spark_version": "8.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "0 0 */2 * * ?", "timezone_id": "US/Pacific", "pause_status": "UNPAUSED"}, "notebook_task": {"notebook_path": "/Users/tejas.nagdulikar@databricks.com/Quickstart Notebook"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1625759666051, "creator_user_name": "tejas.nagdulikar@databricks.com"}, {"job_id": 18831, "settings": {"name": "ryan_job_demo", "new_cluster": {"spark_version": "8.2.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 6}, "email_notifications": {}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/ryan.nagy@databricks.com/Demo/git_demo"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1626370937535}, {"job_id": 21835, "settings": {"name": "DeltaLiveTables", "new_cluster": {"spark_version": "8.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "0 0 */2 * * ?", "timezone_id": "US/Pacific", "pause_status": "UNPAUSED"}, "notebook_task": {"notebook_path": "/Users/alexander.desroches@databricks.com/Demos/Delta/DeltaLiveTables"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1626720135261, "creator_user_name": "alexander.desroches@databricks.com"}, {"job_id": 22590, "settings": {"name": "Task1", "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "6 15 4 * * ?", "timezone_id": "UTC", "pause_status": "PAUSED"}, "pipeline_task": {"pipeline_id": "efe3656f-1197-410f-96f4-cb5d6e510612"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1626768923239, "creator_user_name": "gautam.shah@databricks.com"}, {"job_id": 23188, "settings": {"name": "idbml-airbnb-price-jh-job", "existing_cluster_id": "0720-135905-gazes1", "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "notebook_task": {"notebook_path": "/Users/josephine.ho@databricks.com/IDBML-1.0.0/IDBML 06 - Scheduling a Machine Learning Workflow"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1626792509868, "creator_user_name": "josephine.ho@databricks.com"}, {"job_id": 23279, "settings": {"name": "idbml-airbnb-dg-price-retrain", "new_cluster": {"spark_version": "8.3.x-cpu-ml-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK", "spot_bid_price_percent": 100, "ebs_volume_count": 0}, "node_type_id": "i3.xlarge", "enable_elastic_disk": false, "num_workers": 4}, "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "2 48 10 20 * ?", "timezone_id": "UTC", "pause_status": "UNPAUSED"}, "notebook_task": {"notebook_path": "/Users/dylan.gessner@databricks.com/DMED/IDBML-1.0.0/IDBML 06 - Scheduling a Machine Learning Workflow"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1626792730502, "creator_user_name": "dylan.gessner@databricks.com"}, {"job_id": 24197, "settings": {"name": "luke_multistep", "email_notifications": {"no_alert_for_skipped_runs": false}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "30 34 10 * * ?", "timezone_id": "UTC", "pause_status": "PAUSED"}, "pipeline_task": {"pipeline_id": "d34fd74e-704b-4df0-8f81-9b4695852af2"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1626860101398, "creator_user_name": "luke.watkins@databricks.com"}, {"job_id": 24702, "settings": {"name": "2021-07-07 - S3 Example", "new_cluster": {"spark_version": "8.3.x-scala2.12", "aws_attributes": {"zone_id": "us-west-2c", "first_on_demand": 1, "availability": "SPOT_WITH_FALLBACK"}, "node_type_id": "i3.xlarge", "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"}, "enable_elastic_disk": false, "num_workers": 8}, "email_notifications": {}, "timeout_seconds": 0, "schedule": {"quartz_cron_expression": "0 4 * * * ?", "timezone_id": "US/Pacific", "pause_status": "UNPAUSED"}, "notebook_task": {"notebook_path": "/Users/anastasia.chueva@databricks.com/2021-07-07 - S3 Example"}, "max_concurrent_runs": 1, "format": "SINGLE_TASK"}, "created_time": 1626887858324, "creator_user_name": "anastasia.chueva@databricks.com"}]"""

# COMMAND ----------

fetch_and_save_from_API('jobs/list', 'jobs', 'jobs')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Clusters
# MAGIC 
# MAGIC ### Concept
# MAGIC Clusters can be divided into two types: all-purpose (or interactive) and job (or automated).
# MAGIC - **All-purpose clusters**: Also called interactive clusters. These are clusters that do not terminate immediately after the execution of all of their workloads is over. These clusters will only shutdown if prompted to do so manually or by registering a period of inactivity equal to a set parameter. Additionally, these clusters can be interacted with (hence the name) via the notebook interface in the UI - or tools like Databricks Connect (AWS) (Azure) - meaning that you can exchange inputs and outputs to and from the cluster in a non-linear fashion. Many users can interact with these clusters at the same time - even other Databricks Jobs can be run on this type of cluster. These clusters can be restarted.
# MAGIC - **Job clusters**: Also called automated clusters. These are clusters that only exist during the execution of a Notebook or Script (called Databricks Jobs) and will only terminate after the execution of the workload is over (or an error has been encountered). These clusters only run the workloads that they are commanded with while they were created and can’t accept other interactions meanwhile. These clusters can’t be restarted.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Interactive clusters tend to be used more for the development phase of a project, running quick scripts or ad hoc analysis and for running small jobs where spinning a full job cluster would take more time than running the actual script.
# MAGIC 
# MAGIC Job clusters shine when running production jobs or automated tests, among the advantages are that these clusters won’t compete for resources with other jobs as well as being more economical in the price per DBU.
# MAGIC 
# MAGIC ### Documentation
# MAGIC User Guide ([AWS](https://docs.databricks.com/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/jobs))  
# MAGIC Jobs 2.0 API ([AWS](https://docs.databricks.com/dev-tools/api/2.0/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/2.0/jobs))  
# MAGIC Jobs 2.1 API ([AWS](https://docs.databricks.com/dev-tools/api/latest/jobs.html)) ([Azure](https://redocly.github.io/redoc/?url=https://docs.microsoft.com/azure/databricks/_static/api-refs/jobs-2.1-azure.yaml
# MAGIC ))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The analysis shows all of the Interactive Clusters existing in a workspace.

# COMMAND ----------

sql("""
  SELECT *
  FROM jobs
  WHERE settings.existing_cluster_id IS NOT NULL
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scheduled
# MAGIC 
# MAGIC ### Concept
# MAGIC A job is a way to run non-interactive code in a Databricks cluster. A scheduled job is a way to run a job at a determined time. A scheduled job can run in both interactive and automated clusters.
# MAGIC 
# MAGIC An instance of a job is called a run.
# MAGIC 
# MAGIC A scheduled job is a template with a few main components:
# MAGIC - Logic: This is the code to be run, which can be a notebook, jar, python file etc with the commands to be run.
# MAGIC - Cluster: The machine(s) where the logic will run (it can be both interactive and automated clusters).
# MAGIC - Parameters (optional): Arguments to pass to the job dynamically.
# MAGIC - Schedule (optional): A period schedule for the job.
# MAGIC - Email Notifications (optional): A list of emails to be notified when certain events are triggered.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Scheduled jobs tend to be developed on interactive clusters and moved to job clusters as they mature.
# MAGIC 
# MAGIC It is important to point out that an automated cluster needs to have a similar configuration to the interactive cluster used for development (access to metastore, drivers, libraries, etc).
# MAGIC 
# MAGIC It is a good practice to have a notification whenever any production job fails to alert teams downstream and commence remedial actions.
# MAGIC Notes
# MAGIC 
# MAGIC ### Documentation
# MAGIC User Guide ([AWS](https://docs.databricks.com/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/jobs))  
# MAGIC Jobs 2.0 API ([AWS](https://docs.databricks.com/dev-tools/api/2.0/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/2.0/jobs))  
# MAGIC Jobs 2.1 API ([AWS](https://docs.databricks.com/dev-tools/api/latest/jobs.html)) ([Azure](https://redocly.github.io/redoc/?url=https://docs.microsoft.com/azure/databricks/_static/api-refs/jobs-2.1-azure.yaml
# MAGIC ))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The analysis shows all jobs with an active schedule existing in a workspace.

# COMMAND ----------

sql("""
  SELECT job_id, settings.name
  FROM jobs
  WHERE settings.schedule.pause_status = "UNPAUSED"
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Autoscaling
# MAGIC 
# MAGIC ### Concept
# MAGIC Databricks can choose the appropriate number of workers required to run a job. This is referred to as autoscaling. The alternative to this is to set a fixed number of workers which could be idle (and thus inefficient) if the workload is not demanding enough resources.
# MAGIC 
# MAGIC With autoscaling, Databricks dynamically reallocates workers to account for the characteristics of your job. Certain parts of a pipeline may be more computationally demanding than others, and Databricks automatically adds additional workers during these phases of a job (and removes them when they’re no longer needed).
# MAGIC 
# MAGIC Depending on the constant size of the cluster and the workload, autoscaling gives you one or both of these benefits at the same time. The cluster size can go below the minimum number of workers selected when the cloud provider terminates instances. In this case, Databricks continuously retries to re-provision instances in order to maintain the minimum number of workers.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Autoscaling thus offers two advantages:
# MAGIC - Workloads can run faster compared to a constant-sized under-provisioned cluster.
# MAGIC - Autoscaling clusters can reduce overall costs compared to a statically-sized cluster.
# MAGIC 
# MAGIC Streaming jobs should not be used with autoscaling outside of the Delta Live Tables environment.
# MAGIC High Concurrency clusters that are shared among different users are good candidates for autoscaling.
# MAGIC 
# MAGIC Databricks offers two types of cluster node autoscaling: standard and optimized. For a discussion of the benefits of optimized autoscaling, see the blog post on Optimized Autoscaling.
# MAGIC - Automated (job) clusters always use optimized autoscaling. 
# MAGIC - All-purpose (interactive) clusters perform autoscaling based on the cluster and workspace configuration:
# MAGIC  - For clusters running Databricks Runtime 6.4 and above, optimized autoscaling is used by all-purpose clusters in the Premium plan (or, for customers who subscribed to Databricks before March 3, 2020, the Operational Security package).
# MAGIC  - Standard autoscaling is used by all-purpose clusters running Databricks Runtime 6.3 and below, as well as all all-purpose clusters on the Standard plan.
# MAGIC 
# MAGIC ### Documentation
# MAGIC User Guide ([AWS](https://docs.databricks.com/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/jobs))  
# MAGIC Jobs 2.0 API ([AWS](https://docs.databricks.com/dev-tools/api/2.0/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/2.0/jobs))  
# MAGIC Jobs 2.1 API ([AWS](https://docs.databricks.com/dev-tools/api/latest/jobs.html)) ([Azure](https://redocly.github.io/redoc/?url=https://docs.microsoft.com/azure/databricks/_static/api-refs/jobs-2.1-azure.yaml
# MAGIC ))
# MAGIC 
# MAGIC ### Analysis
# MAGIC This analysis shows a list of the clusters that have been set up to autoscale.

# COMMAND ----------

sql("""
  SELECT *
  FROM jobs
  WHERE settings.new_cluster.autoscaling
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retries
# MAGIC - you want max_retries to be infinite on streaming jobs (-1)
# MAGIC 
# MAGIC ### Concept
# MAGIC Whenever a Databricks Job fails, a job can be retried based on the retry policy. A retrying job will restart the cluster and attempt to run the same task again until successful or the number of times set on the retry policy has been reached.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Streaming jobs should have the max_retries set to -1.
# MAGIC 
# MAGIC ### Documentation
# MAGIC User Guide ([AWS](https://docs.databricks.com/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/jobs))  
# MAGIC Jobs 2.0 API ([AWS](https://docs.databricks.com/dev-tools/api/2.0/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/2.0/jobs))  
# MAGIC Jobs 2.1 API ([AWS](https://docs.databricks.com/dev-tools/api/latest/jobs.html)) ([Azure](https://redocly.github.io/redoc/?url=https://docs.microsoft.com/azure/databricks/_static/api-refs/jobs-2.1-azure.yaml
# MAGIC ))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The analysis shows the jobs (by job id) grouped by the max retries parameters.

# COMMAND ----------

sql("""
  SELECT max_retries, collect_set(job_id) as job_ids
  FROM (SELECT *,settings.max_retries as max_retries
  FROM jobs)
  GROUP BY 1
  ORDER BY 1 DESC
""")

# COMMAND ----------

# DBTITLE 1,Retry on Timeout
sql("""
  SELECT retry_on_timeout, collect_set(job_id) as job_ids
  FROM (SELECT *, CASE WHEN settings.retry_on_timeout = 'true' THEN 'true' ELSE 'false' END as retry_on_timeout
  FROM jobs)
  GROUP BY 1
  ORDER BY retry_on_timeout DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Email notifications
# MAGIC 
# MAGIC ### Concept
# MAGIC A Databricks Job can be configured to trigger the delivery of e-mails based on specific milestones reached during the execution of it to alert different people:
# MAGIC - Start: A notification will be sent when the job begins.
# MAGIC - Success: A notification will be sent when the job successfully completes. A job run is considered to have completed successfully if it ends with a TERMINATED life_cycle_state and a SUCCESSFUL result_state.
# MAGIC - Failure: A notification will be sent when the job is unsuccessful. A job run is considered to have completed unsuccessfully if it ends with an INTERNAL_ERROR life_cycle_state or a SKIPPED, FAILED, or TIMED_OUT result_state.
# MAGIC 
# MAGIC These e-mails can be sent to a single or multiple addresses whenever a job starts, succeeds or fails.
# MAGIC 
# MAGIC This functionality can be activated while creating or modifying a job via the GUI or REST API.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Different types of mails can be received by different recipients, i.e. Failure alerts can be routed to a support team while Start alerts can be sent to a data engineer.
# MAGIC 
# MAGIC ### Documentation
# MAGIC User Guide ([AWS](https://docs.databricks.com/jobs.html#alerts)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/jobs#--alerts))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table shows the ID and Name of the jobs that do not have a notification on failure.

# COMMAND ----------

sql("""
  SELECT job_id,
  CASE WHEN settings.name = 'Untitled' THEN NULL ELSE settings.name END as job_name
  FROM (SELECT *, settings.email_notifications.on_failure as on_failure FROM jobs)
  WHERE on_failure IS NULL
  ORDER BY job_name DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-Task
# MAGIC 
# MAGIC ### Concept
# MAGIC Multi-Task jobs are Databricks Jobs that can be configured to have more than one Notebook/Script/JAR - or task. With Multi Task Jobs you can set dependencies between tasks via a Direct Acyclic Graph which works as a flexible orchestrator for Databricks resources.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Different tasks within a Multi Task Jobs can use different clusters, therefore jobs can be efficient by using the right resources.
# MAGIC 
# MAGIC ### Documentation
# MAGIC User Guide ([AWS](https://docs.databricks.com/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/jobs))  
# MAGIC Data Engineering Guide ([AWS](https://docs.databricks.com/data-engineering/jobs/index.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/data-engineering/jobs/))  
# MAGIC Jobs 2.1 API ([AWS](https://docs.databricks.com/dev-tools/api/latest/jobs.html)) ([Azure](https://redocly.github.io/redoc/?url=https://docs.microsoft.com/azure/databricks/_static/api-refs/jobs-2.1-azure.yaml
# MAGIC ))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The analysis shows the count of jobs as SINGLE_TASK or MULTI_TASK in the workspace.

# COMMAND ----------

sql("""
  SELECT settings.format, count(*) as count
  FROM jobs
  GROUP BY settings.format
  ORDER BY count DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jobs Without a Name
# MAGIC 
# MAGIC ### Concept
# MAGIC A job can be identified by both an id (a number guaranteed to be unique for a given workspace) as well as a name (a human readable name). While an id is always given to a job, a name is not required and multiple jobs can have the same name.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC It is best practice to always have descriptive and unique names for a job.
# MAGIC 
# MAGIC ### Documentation
# MAGIC User Guide ([AWS](https://docs.databricks.com/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/jobs))  
# MAGIC Jobs 2.0 API ([AWS](https://docs.databricks.com/dev-tools/api/2.0/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/2.0/jobs))  
# MAGIC Jobs 2.1 API ([AWS](https://docs.databricks.com/dev-tools/api/latest/jobs.html)) ([Azure](https://redocly.github.io/redoc/?url=https://docs.microsoft.com/azure/databricks/_static/api-refs/jobs-2.1-azure.yaml
# MAGIC ))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table shows the jobs that do not have a name.

# COMMAND ----------

sql("""
  SELECT job_id, settings.schedule
  FROM jobs
  WHERE settings.name = "Untitled"
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Libraries
# MAGIC 
# MAGIC ### Concept
# MAGIC To make third-party or custom code available to notebooks and jobs running on your clusters, you can install a library. Libraries can be written in Python, Java, Scala, and R. You can upload Java, Scala, and Python libraries and point to external packages in PyPI, Maven, and CRAN repositories.
# MAGIC 
# MAGIC Code that runs as a Job, same as code that runs interactively may depend on libraries.
# MAGIC 
# MAGIC Libraries within jobs depend on the cluster that the job is running, but can be installed in any of the ways made available by Databricks.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC It is always a best practice to fix and specify a library as much as possible by stating the version when installing it.
# MAGIC After the initial development of code and before promoting it to a job running on an automated cluster, it is a good practice to identify the minimum requirements of dependencies and copying it as the configuration of the automated cluster.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Libraries ([AWS](https://docs.databricks.com/libraries/index.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/libraries/))  
# MAGIC Cluster Libraries ([AWS](https://docs.databricks.com/libraries/cluster-libraries.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/libraries/cluster-libraries))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The analysis identifies the jobs that have cluster libraries installed. It identifies the Job by id and name and lists the libraries used.

# COMMAND ----------

sql("""
  SELECT job_id, settings.name, settings.libraries
  FROM jobs
  WHERE settings.libraries IS NOT NULL
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Job Runs
# MAGIC 
# MAGIC ### Concept
# MAGIC A Job run is an instance of a Job. While the Job is the template containing the logic, cluster, libraries and schedule. A Job run is created whenever the Job template is materialized, the code containing the logic is loaded onto the cluster, optional parameters are used and the commands are interpreted.
# MAGIC 
# MAGIC ### Documentation
# MAGIC User Guide ([AWS](https://docs.databricks.com/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/jobs))  
# MAGIC Jobs 2.0 API ([AWS](https://docs.databricks.com/dev-tools/api/2.0/jobs.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/2.0/jobs))  
# MAGIC Jobs 2.1 API ([AWS](https://docs.databricks.com/dev-tools/api/latest/jobs.html)) ([Azure](https://redocly.github.io/redoc/?url=https://docs.microsoft.com/azure/databricks/_static/api-refs/jobs-2.1-azure.yaml
# MAGIC ))

# COMMAND ----------

fetch_and_save_from_API('jobs/runs/list', 'runs', 'job_runs')

# COMMAND ----------

sql("""
  SELECT *
  FROM job_runs
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jobs Summary
# MAGIC 
# MAGIC ### Concept
# MAGIC 
# MAGIC ### Best Practice
# MAGIC 
# MAGIC ### Documentation
# MAGIC (AWS) (Azure)
# MAGIC 
# MAGIC ### Analysis
# MAGIC The analysis shows a pie chart that represents the percentage of the total runs attributed to a specific job by job name.
# MAGIC This can help you understand which jobs are run most often.

# COMMAND ----------

sql("""
  SELECT job_id, run_name, count(*) as count
  FROM job_runs
  GROUP BY 1, 2
  ORDER BY count DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Longest Runs
# MAGIC 
# MAGIC ### Concept
# MAGIC 
# MAGIC ### Best Practice
# MAGIC 
# MAGIC ### Documentation
# MAGIC (AWS) (Azure)
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table below shows the jobs in order from longest to shortest running time.
# MAGIC This can help you identify bottlenecks and optimization opportunities.

# COMMAND ----------

sql("""
  SELECT double(execution_duration / 1000) AS execution_duration_seconds,
    job_id, run_name, run_id
  FROM job_runs
  ORDER BY 1 DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jobs with runs with biggest Max-min time
# MAGIC 
# MAGIC ### Concept
# MAGIC 
# MAGIC ### Best Practice
# MAGIC 
# MAGIC ### Documentation
# MAGIC (AWS) (Azure)
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table below shows the jobs ordered by how much difference there is between the shortest and longest run.
# MAGIC This can help you identify bottlenecks and optimization opportunities.

# COMMAND ----------

sql("""
  SELECT *, max_execution_duration_seconds - min_execution_duration_seconds as diff_seconds
  FROM (
    SELECT job_id, run_name,
      min(execution_duration) / 1000 as min_execution_duration_seconds,
      max(execution_duration) / 1000 as max_execution_duration_seconds
    FROM job_runs
    GROUP BY 1, 2
  )
  ORDER BY diff_seconds DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Failed Runs
# MAGIC 
# MAGIC ### Concept
# MAGIC 
# MAGIC ### Best Practice
# MAGIC 
# MAGIC ### Documentation
# MAGIC (AWS) (Azure)
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table below shows the runs that did not complete successfully along with the parent job info.

# COMMAND ----------

sql("""
  SELECT job_id, run_name, from_unixtime(start_time / 1000,'yyyy-MM-dd HH:mm:ss') as start_time, state
  FROM job_runs
  WHERE state.result_state <> "SUCCESS"
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Pools
# MAGIC 
# MAGIC ### Concept
# MAGIC Databricks pools reduce cluster start and auto-scaling times by maintaining a set of idle, ready-to-use instances. When a cluster is attached to a pool, cluster nodes are created using the pool’s idle instances. If the pool has no idle instances, the pool expands by allocating a new instance from the instance provider in order to accommodate the cluster’s request. When a cluster releases an instance, it returns to the pool and is free for another cluster to use. Only clusters attached to a pool can use that pool’s idle instances.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC You can specify a different pool for the driver node and worker nodes, or use the same pool for both.
# MAGIC 
# MAGIC There are a few recommended ways to manage this cost:
# MAGIC - Manually edit the size of your pool to meet your needs.
# MAGIC - If you’re only running interactive workloads during business hours, make sure the pool’s “Min Idle” instance count is set to zero after hours.
# MAGIC - If your automated data pipeline runs for a few hours at night, set the “Min Idle” count a few minutes before the pipeline starts and then revert it to zero afterwards.
# MAGIC - Alternatively, always keep a “Min Idle” of zero, but set the “Idle Instance Auto Termination” timeout to meet your needs.
# MAGIC  - The first job run on the pool will start slowly, but subsequent jobs run within the timeout period will start quickly.
# MAGIC  - When the jobs are done, all instance in the pool will terminate after the idle timeout period, avoiding cloud provider costs.
# MAGIC 
# MAGIC Databricks does not charge DBUs for idle instances not in use by a Databricks cluster, but cloud provider infrastructure costs do apply.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Company Blog ([Databricks](https://databricks.com/blog/2019/11/11/databricks-pools-speed-up-data-pipelines.html))  
# MAGIC Pool Configuration ([AWS](https://docs.databricks.com/clusters/instance-pools/index.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/instance-pools/))  
# MAGIC Best Practices ([AWS](https://docs.databricks.com/clusters/instance-pools/pool-best-practices.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/instance-pools/pool-best-practices))  
# MAGIC API ([AWS](https://docs.databricks.com/dev-tools/api/latest/instance-pools.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/instance-pools))  

# COMMAND ----------

fetch_and_save_from_API('instance-pools/list', 'instance_pools', 'pools')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preloaded Spark Versions
# MAGIC 
# MAGIC ### Concept
# MAGIC You can speed up cluster launches by selecting a Databricks Runtime version to be loaded on idle instances in the pool. If a user selects that runtime when they create a cluster backed by the pool, that cluster will launch even more quickly than a pool-backed cluster that doesn’t use a preloaded Databricks Runtime version.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC You can specify a different pool for the driver node and worker nodes, or use the same pool for both.
# MAGIC 
# MAGIC A pool without a preloaded Spark version slows down cluster launches, as it causes the Databricks Runtime version to download on demand to idle instances in the pool. When the cluster releases the instances in the pool, the Databricks Runtime version remains cached on those instances. The next cluster creation operation that uses the same Databricks Runtime version might benefit from this caching behavior, but it is not guaranteed.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Preloaded Spark Versions ([AWS](https://docs.databricks.com/clusters/instance-pools/configure.html#preload-databricks-runtime-version)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/instance-pools/configure#--preload-databricks-runtime-version))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table below shows the pools that have a preloaded Spark version.

# COMMAND ----------

sql("""
  SELECT preloaded_spark_versions, collect_set(instance_pool_name) as names
  FROM pools
  GROUP BY 1
  ORDER BY preloaded_spark_versions DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pools with Min Idle
# MAGIC 
# MAGIC ### Concept
# MAGIC The minimum number of instances the pool keeps idle. These instances do not terminate, regardless of the setting specified in Idle Instance Auto Termination. If a cluster consumes idle instances from the pool, Azure Databricks provisions additional instances to maintain the minimum.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC To benefit fully from pools, you can pre-populate newly created pools. Set the Min Idle instances greater than zero in the pool configuration.
# MAGIC - If you’re only running interactive workloads during business hours, make sure the pool’s “Min Idle” instance count is set to zero after hours.
# MAGIC - If your automated data pipeline runs for a few hours at night, set the “Min Idle” count a few minutes before the pipeline starts and then revert it to zero afterwards.
# MAGIC 
# MAGIC Alternatively, if you’re following the recommendation to set this value to zero, use a starter job to ensure that newly created pools have available instances for clusters to access.
# MAGIC 
# MAGIC With the starter job approach, schedule a job with flexible execution time requirements to run before jobs with more strict performance requirements or before users start using interactive clusters. After the job finishes, the instances used for the job are released back to the pool. Set Min Idle instance setting to 0 and set the Idle Instance Auto Termination time high enough to ensure that idle instances remain available for subsequent jobs.
# MAGIC 
# MAGIC Using a starter job allows the pool instances to spin up, populate the pool, and remain available for downstream job or interactive clusters.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Min Idle Instances ([AWS](https://docs.databricks.com/clusters/instance-pools/configure.html#minimum-idle-instances)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/instance-pools/configure#minimum-idle-instances))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table below shows the Pools that have a Min Idle instance set to a number bigger than 0.
# MAGIC This is useful to single out Pools where idle resources are incurring in unnecessary expenses.

# COMMAND ----------

sql("""
  SELECT instance_pool_name, min_idle_instances, node_type_id
  FROM pools
  WHERE min_idle_instances > 0
  ORDER BY min_idle_instances DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clusters Using Pools
# MAGIC 
# MAGIC ### Concept
# MAGIC To reduce cluster start time, you can designate predefined pools of idle instances to create worker nodes and the driver node. This is also called attaching the cluster to the pools. The cluster is created using instances in the pools. If a pool does not have sufficient idle resources to create the requested driver node or worker nodes, the pool expands by allocating new instances from the instance provider. When the cluster is terminated, the instances it used are returned to the pool and can be reused by a different cluster.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC You can attach a different pool for the driver node and worker nodes, or attach the same pool for both.
# MAGIC A cluster can be attached to one or more pools.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Use a Pool ([AWS](https://docs.databricks.com/clusters/instance-pools/cluster-instance-pool.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/instance-pools/cluster-instance-pool))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table below shows the clusters that have been configured to use a Databrick Pool to shorten the node provisioning and start up time.

# COMMAND ----------

sql("""
  SELECT driver_instance_source.instance_pool_id as pool_id, collect_set(cluster_id)
  FROM clusters
  GROUP BY 1
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto-termination Minutes
# MAGIC 
# MAGIC ### Concept
# MAGIC The time in minutes that instances above the value set in Minimum Idle Instances can be idle before being terminated by the pool.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Set the Idle Instance Auto Termination time to provide a buffer between when the instance is released from the cluster and when it’s dropped from the pool. Set this to a period that allows you to minimize cost while ensuring the availability of instances for scheduled jobs.
# MAGIC 
# MAGIC For example, job A is scheduled to run at 8:00 AM and takes 40 minutes to complete. Job B is scheduled to run at 9:00 AM and takes 30 minutes to complete. Set the Idle Instance Auto Termination value to 20 minutes to ensure that instances returned to the pool when job A completes are available when job B starts. Unless they are claimed by another cluster, those instances are terminated 20 minutes after job B ends.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Idle Instance Auto Termination ([AWS](https://docs.databricks.com/clusters/instance-pools/configure.html#idle-instance-auto-termination)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/instance-pools/configure#idle-instance-auto-termination))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table below shows the Databricks Pools in descending order of their Idle Instance Auto Termination setting.

# COMMAND ----------

sql("""
  SELECT idle_instance_autotermination_minutes, instance_pool_name
  FROM pools
  ORDER BY idle_instance_autotermination_minutes DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Policies
# MAGIC 
# MAGIC ### Concept
# MAGIC A cluster policy limits the ability to configure clusters based on a set of rules. The policy rules limit the attributes or attribute values available for cluster creation. Cluster policies have ACLs that limit their use to specific users and groups.
# MAGIC 
# MAGIC Cluster policies let you:
# MAGIC - Limit users to create clusters with prescribed settings.
# MAGIC - Simplify the user interface and enable more users to create their own clusters (by fixing and hiding some values).
# MAGIC - Control cost by limiting per cluster maximum cost (by setting limits on attributes whose values contribute to hourly price).
# MAGIC 
# MAGIC Combined with effective onboarding, approval, and chargeback processes, cluster policies can be a foundational component in Databricks platform governance.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Cluster policies support all cluster attributes controlled with the Clusters API 2.0. The specific type of restrictions supported may vary per field (based on their type and relation to the cluster form UI elements).
# MAGIC 
# MAGIC In addition, cluster policies support the following synthetic attributes:
# MAGIC - A “max DBU-hour” metric, which is the maximum DBUs a cluster can use on an hourly basis. This metric is a direct way to control cost at the individual cluster level.
# MAGIC - A limit on the source that creates the cluster: Jobs service (job clusters), Clusters UI, Clusters REST API (all-purpose clusters).
# MAGIC 
# MAGIC ### Documentation
# MAGIC Manage Policies ([AWS](https://docs.databricks.com/administration-guide/clusters/policies.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/administration-guide/clusters/policies))  
# MAGIC Best Practices ([AWS](https://docs.databricks.com/administration-guide/clusters/policies-best-practices.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/administration-guide/clusters/policies-best-practices))  
# MAGIC API ([AWS](https://docs.databricks.com/dev-tools/api/latest/policies.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/policies))

# COMMAND ----------

fetch_and_save_from_API('policies/clusters/list', 'policies', 'policies')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto-termination Minutes
# MAGIC 
# MAGIC ### Concept
# MAGIC An important value to consider for any policy is the auto-termination minutes. This value controls how many minutes a cluster can be idle before being terminated.
# MAGIC 
# MAGIC A value of 0 represents no auto termination. When hidden, removes the auto termination checkbox and value input from the UI.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC There are very few reasons why you would not use auto termination for a cluster. Clusters with streaming jobs and clusters with RStudio are some examples. 
# MAGIC 
# MAGIC ### Documentation
# MAGIC Manage Policies ([AWS](https://docs.databricks.com/administration-guide/clusters/policies.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/administration-guide/clusters/policies))  
# MAGIC Best Practices ([AWS](https://docs.databricks.com/administration-guide/clusters/policies-best-practices.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/administration-guide/clusters/policies-best-practices))  
# MAGIC API ([AWS](https://docs.databricks.com/dev-tools/api/latest/policies.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/policies))  
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table below shows the Auto-Termination minutes values for all of the policies in the workspace.

# COMMAND ----------

sql("""
  SELECT name, definition:autotermination_minutes
  FROM policies
  ORDER BY definition:autotermination_minutes DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom Tags
# MAGIC 
# MAGIC ### Concept
# MAGIC A Policy that enforces Custom Tags is a great way for your organization to make sure that resources are easily recognisable. Tags can be used for reporting of costs, use, and overall governance of resources.
# MAGIC 
# MAGIC The parameter custom_tags.* in a policy controls specific tag values by appending the tag name, for example: custom_tags.< mytag >.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC It is a good practice to tag your clusters for things like cost center/team/admin team/project/etc.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Best Practices ([AWS](https://docs.databricks.com/administration-guide/clusters/policies-best-practices.html#tag-enforcement)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/administration-guide/clusters/policies-best-practices#--tag-enforcement))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The following table shows if a policy contains cutom_tags and if so, what they are.

# COMMAND ----------

sql("""
  SELECT name,
    CASE WHEN definition LIKE '%custom_tags%' THEN True ELSE False END as contains_custom_tags,
    regexp_extract_all(definition, 'custom_tags\.(.*?)\"') as custom_tags,
    definition
  FROM policies
  ORDER BY contains_custom_tags DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single-node
# MAGIC 
# MAGIC ### Concept
# MAGIC Cluster policies simplify cluster configuration for Single Node clusters. A Cluster policy enforcing single-node clusters, allows users to create a Single Node cluster with no worker nodes with Spark enabled in local mode.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Consider the example of a data science team whose members do not have permission to create clusters. An admin can create a cluster policy that authorizes team members to create a maximum number of Single Node clusters, using pools and cluster policies: This enables people to create cheap compute resources to run during the development/exploration stages of their work.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Single-Node Cluster ([AWS](https://docs.databricks.com/clusters/single-node.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/single-node))  
# MAGIC Single-Node Cluster Policy([AWS](https://docs.databricks.com/clusters/single-node.html#single-node-policy)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/single-node#single-node-policy))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table shows the list of Policies that enforce Single Node clusters.

# COMMAND ----------

sql("""
  SELECT custom_tags.ResourceClass, collect_set(cluster_id) as cluster_ids
  FROM clusters
  WHERE custom_tags.ResourceClass = 'SingleNode'
  GROUP BY custom_tags.ResourceClass
""")


# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow
# MAGIC 
# MAGIC ### Concept
# MAGIC An open source platform developed by Databricks to help manage the complete machine learning lifecycle with enterprise reliability, security and scale.
# MAGIC 
# MAGIC It has the following primary components:
# MAGIC - Tracking: Allows you to track experiments to record and compare parameters and results.
# MAGIC - Models: Allow you to manage and deploy models from a variety of ML libraries to a variety of model serving and inference platforms.
# MAGIC - Projects: Allow you to package ML code in a reusable, reproducible form to share with other data scientists or transfer to production.
# MAGIC - Model Registry: Allows you to centralize a model store for managing models’ full lifecycle stage transitions: from staging to production, with capabilities for versioning and annotating.
# MAGIC - Model Serving: Allows you to host MLflow Models as REST endpoints.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Using MLFlow whenever possible and using MLFlow early, this is the easiest way to manage the lifecycle of your machine learning flow with reliability, security and scale.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Blog ([Databricks](https://databricks.com/product/managed-mlflow))  
# MAGIC MLFlow Quickstart ([AWS](https://docs.databricks.com/applications/mlflow/quick-start.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/quick-start))  
# MAGIC MLFlow Guide ([AWS](https://docs.databricks.com/applications/mlflow/index.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiments
# MAGIC 
# MAGIC ### Concept
# MAGIC MLflow tracking is based on two concepts, experiments and runs:
# MAGIC 
# MAGIC - An MLflow experiment is the primary unit of organization and access control for MLflow runs; all MLflow runs belong to an experiment. Experiments let you visualize, search for, and compare runs, as well as download run artifacts and metadata for analysis in other tools.
# MAGIC - An MLflow run corresponds to a single execution of model code. Each run records the following information:
# MAGIC  - Source: Name of the notebook that launched the run or the project name and entry point for the run.
# MAGIC  - Version: Notebook revision if run from a notebook or Git commit hash if run from an MLflow Project.
# MAGIC  - Start & end time: Start and end time of the run.
# MAGIC  - Parameters: Model parameters saved as key-value pairs. Both keys and values are strings.
# MAGIC  - Metrics: Model evaluation metrics saved as key-value pairs. The value is numeric. Each metric can be updated throughout the course of the run (for example, to track how your model’s loss function is converging), and MLflow records and lets you visualize the metric’s history.
# MAGIC  - Tags: Run metadata saved as key-value pairs. You can update tags during and after a run completes. Both keys and values are strings.
# MAGIC  - Artifacts: Output files in any format. For example, you can record images, models (for example, a pickled scikit-learn model), and data files (for example, a Parquet file) as an artifact.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Using MLFlow whenever possible and using MLFlow early, this is the easiest way to manage the lifecycle of your machine learning flow with reliability, security and scale.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Experiment Page ([AWS](https://docs.databricks.com/applications/machine-learning/experiments-page.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/experiments-page))  
# MAGIC Tracking Experiments ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/tracking))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table below shows each of the MLFlow experiments created in your workspace along with their details.

# COMMAND ----------

experiments_df = spark.createDataFrame(mlflow.list_experiments())
(experiments_df.write.mode("overwrite").saveAsTable("workspace_analysis.mlflow_experiments"))

# COMMAND ----------

sql(""""
  SELECT _name, *
  FROM mlflow_experiments
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Databricks AutoML
# MAGIC 
# MAGIC ### Concept
# MAGIC Databricks AutoML is a tool that empowers data teams to quickly build and deploy machine learning models by automating the heavy lifting of preprocessing, feature engineering and model training/tuning.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Databricks AutoML glassbox approach excels at creating baseline models that can be build on top of and improved by Data Scientists. The base code for feature engineering, model training and tuning is autogenerated and is extensively commented so that a Data Scientist can jump in and tweak it to produce better models in shorter times.
# MAGIC 
# MAGIC ### Documentation
# MAGIC AutoML Overview ([Blog](https://databricks.com/blog/2021/05/27/introducing-databricks-automl-a-glass-box-approach-to-automating-machine-learning-development.html))  
# MAGIC AutoML ([AWS](https://docs.databricks.com/applications/machine-learning/automl.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/automl))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table below shows each of the MLFlow experiments created with AutoML in your workspace along with their details.

# COMMAND ----------

sql("""
  SELECT _tags._databricks_automl as using_automl, *
  FROM mlflow_experiments
  WHERE _tags._databricks_automl = "True"
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Tokens
# MAGIC 
# MAGIC ### Concept
# MAGIC 
# MAGIC ### Best Practice
# MAGIC 
# MAGIC ### Documentation
# MAGIC Engineering Guide ([AWS](https://docs.databricks.com/dev-tools/api/latest/authentication.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/authentication))  
# MAGIC Token API ([AWS](https://docs.databricks.com/dev-tools/api/latest/tokens.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/tokens))  
# MAGIC Token Management ([AWS](https://docs.databricks.com/administration-guide/access-control/tokens.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/administration-guide/access-control/tokens))

# COMMAND ----------

fetch_and_save_from_API("token-management/tokens", "token_infos", "tokens")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Expiry Time > 90 days from today or No Expiry Time
# MAGIC 
# MAGIC ### Concept
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Managing the life-cycle of PATs is very important as with any authentication device. Unmanaged tokens (ie tokens that are set to never expire) are a security vulnerability.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Engineering Guide ([AWS](https://docs.databricks.com/dev-tools/api/latest/authentication.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/authentication))  
# MAGIC Token API ([AWS](https://docs.databricks.com/dev-tools/api/latest/tokens.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/tokens))  
# MAGIC Token Management ([AWS](https://docs.databricks.com/administration-guide/access-control/tokens.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/administration-guide/access-control/tokens))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table contains the list of tokens that are either set to expire 90 days from today or to never expire as well as their creator and comment.
# MAGIC This should be useful as a starting point to understand the usage of the tokens and identify vulnerabilities.

# COMMAND ----------

sql("""
  SELECT *
  FROM tokens
  WHERE (datediff(from_unixtime(expiry_time / 1000,'yyyy-MM-dd HH:mm:ss'), current_date()) > 90) OR expiry_time = -1
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recent Tokens in Last 30 Days
# MAGIC 
# MAGIC ### Concept
# MAGIC 
# MAGIC ### Best Practice
# MAGIC 
# MAGIC ### Documentation
# MAGIC Engineering Guide ([AWS](https://docs.databricks.com/dev-tools/api/latest/authentication.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/authentication))  
# MAGIC Token API ([AWS](https://docs.databricks.com/dev-tools/api/latest/tokens.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/tokens))  
# MAGIC Token Management ([AWS](https://docs.databricks.com/administration-guide/access-control/tokens.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/administration-guide/access-control/tokens))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The list shows all of the tokens that were created in the last 30 days.

# COMMAND ----------

sql("""
  SELECT *
  FROM tokens
  WHERE datediff(current_date(), from_unixtime(creation_time / 1000,'yyyy-MM-dd HH:mm:ss')) <= 30
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Delta Live Tables
# MAGIC 
# MAGIC ### Concept
# MAGIC 
# MAGIC ### Best Practice
# MAGIC 
# MAGIC ### Documentation
# MAGIC Stating Guide ([Blog](https://databricks.com/discover/pages/getting-started-with-delta-live-tables?itm_data=deltalivetables-link-gettingstarteddlt))  
# MAGIC DLT Main Page ([AWS](https://docs.databricks.com/data-engineering/delta-live-tables/index.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/data-engineering/delta-live-tables/))  
# MAGIC DLT User Guide ([AWS](https://docs.databricks.com/data-engineering/delta-live-tables/delta-live-tables-user-guide.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/data-engineering/delta-live-tables/delta-live-tables-user-guide))
# MAGIC 
# MAGIC ### Analysis
# MAGIC This table shows a list of all of the Delta Live Tables created on the workspace as well as its creator and other details.

# COMMAND ----------

fetch_and_save_from_API('pipelines', 'statuses', 'pipelines')

# COMMAND ----------

sql("""
  SELECT *
  FROM pipelines
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Users and Groups
# MAGIC 
# MAGIC ### Concept
# MAGIC Databricks Groups are logical buckets to add users, and other groups. They are useful to assign the same entitlements to multiple users.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC 
# MAGIC ### Documentation
# MAGIC Users and Groups ([AWS](https://docs.databricks.com/administration-guide/users-groups/index.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/administration-guide/users-groups/))  
# MAGIC Groups ([AWS](https://docs.databricks.com/administration-guide/users-groups/groups.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/administration-guide/users-groups/groups))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The analysis shows a table of all of the groups created on the workspace.

# COMMAND ----------

fetch_and_save_from_API('groups/list', 'group_names', 'groups', True)

# COMMAND ----------

sql("""
  SELECT *
  FROM groups
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Repos
# MAGIC 
# MAGIC ### Concept
# MAGIC To support best practices for data science and engineering code development, Databricks Repos provides repository-level integration with Git providers. You can develop code in a Databricks notebook and sync it with a remote Git repository. Databricks Repos lets you use Git functionality such as cloning a remote repo, managing branches, pushing and pulling changes, and visually comparing differences upon commit.
# MAGIC 
# MAGIC Databricks Repos also provides an API that you can integrate with your CI/CD pipeline. For example, you can programmatically update a Databricks repo so that it always has the most recent code version.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC 
# MAGIC ### Documentation
# MAGIC Product ([Databricks](https://databricks.com/product/repos))  
# MAGIC User Guide ([AWS](https://docs.databricks.com/repos.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/repos))  
# MAGIC API ([AWS](https://docs.databricks.com/dev-tools/api/latest/repos.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/repos))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table shows the different repos that have been created on the workspace as well as their details.

# COMMAND ----------

fetch_and_save_from_API('repos', 'repos', 'repos')

# COMMAND ----------

sql("""
  SELECT *
  FROM repos
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Roles
# MAGIC This utility is usable only on clusters with credential passthrough enabled.
# MAGIC 
# MAGIC https://docs.databricks.com/dev-tools/databricks-utils.html#credentials-utility-dbutilscredentials
# MAGIC 
# MAGIC ### Concept
# MAGIC 
# MAGIC ### Best Practice
# MAGIC 
# MAGIC ### Documentation
# MAGIC (AWS) (Azure)
# MAGIC 
# MAGIC ### Analysis

# COMMAND ----------

try:
  dbutils.credentials.showRoles()
except:
  pass

# COMMAND ----------

# MAGIC %md
# MAGIC # Secrets
# MAGIC 
# MAGIC ### Concept
# MAGIC A secret is a key-value pair that stores secret material, with a key name unique within a secret scope. Instead of directly entering your credentials into a notebook, use Databricks secrets to store your credentials and reference them in notebooks and jobs. To manage secrets, you can use the Databricks CLI to access the Secrets API 2.0.
# MAGIC 
# MAGIC Databricks Secrets is a managed solution that helps you tackle three essential topics:
# MAGIC - Credential leakage prevention – Accidents happen, and sometimes users can accidentally leak credentials via log messages to notebook outputs. This puts the credentials at risk of exposure, since Databricks notebooks automatically snapshot notebook contents as part of revision-history tracking. These snapshots store both command content and cell output, so if a notebook is shared, another user could revert to an older snapshot and view the printed secret value. Any new system should strive to keep security teams at ease by preventing accidental secret leakage from occurring altogether.
# MAGIC - Access control management – Different teams within an organization interact with credentials for different purposes. For example, an administrator might provision the credentials, but teams that leverage the credentials only need read-only permissions for those credentials. Supporting fine-grained access control allows teams to reason properly about the state of the world.
# MAGIC - Auditability – Leveraging credentials is a sensitive operation. Security teams need historical usage records to meet compliance needs and enable accurate responses to security events.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC You can use Databricks keyvault-backed secret scope or use the keyvaults from your Cloud Service Provider.
# MAGIC 
# MAGIC The most important thing is to always use secrets when dealing with credentials (passwords, tokens, etc). This not only ensures you are not exposing credentials in your code, but also facilitates having one place to manage all your credentials.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Product ([Databricks](https://databricks.com/blog/2018/06/04/securely-managing-credentials-in-databricks.html))  
# MAGIC Secret Management ([AWS](https://docs.databricks.com/security/secrets/index.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/security/secrets/))  
# MAGIC User Guide ([AWS](https://docs.databricks.com/security/secrets/secrets.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/security/secrets/secret-scopes))  
# MAGIC API ([AWS](https://docs.databricks.com/dev-tools/api/latest/secrets.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/secrets))  
# MAGIC CLI ([AWS](https://docs.databricks.com/dev-tools/cli/secrets-cli.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/cli/secrets-cli))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table shows the scopes created in the workspace along with the key of the secrets (not the values).

# COMMAND ----------

scopes = dbutils.secrets.listScopes()

def scope_to_secrets(scope):
  scope_name = scope[0]
  secrets = list(map(lambda secret: secret[0], dbutils.secrets.list(scope_name)))
  return (scope_name, secrets)

secrets_data = list(map(scope_to_secrets, scopes))


simpleSchema = StructType([
  StructField("scope", StringType(),True),
  StructField("secrets", ArrayType(StringType(), True))]
)
rdd = sc.parallelize(secrets_data)
(
  spark.createDataFrame(rdd, simpleSchema)
  .write
  .mode("overwrite")
  .saveAsTable("workspace_analysis.{}".format("secrets"))
)

# COMMAND ----------

sql("""
  SELECT * FROM secrets
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Databricks SQL
# MAGIC 
# MAGIC ### Concept
# MAGIC Databricks SQL provides a simple experience for SQL users who want to run quick ad-hoc queries on their data lake, create multiple visualization types to explore query results from different perspectives, and build and share dashboards. This guide provides getting-started, how-to, and reference information for Databricks SQL users and administrators.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Product ([Databricks](https://databricks.com/product/databricks-sql))  
# MAGIC Guide ([AWS](https://docs.databricks.com/sql/index.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/sql/))  
# MAGIC Developer Guide ([AWS](https://docs.databricks.com/spark/latest/spark-sql/index.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/spark/latest/spark-sql/))  
# MAGIC Language Manual ([AWS](https://docs.databricks.com/sql/language-manual/index.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/sql/language-manual/))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table contains the list of all SQL endpoints created in the workspace as well as their specific configurations.

# COMMAND ----------

fetch_and_save_from_API('sql/endpoints', 'endpoints', 'endpoints')

# COMMAND ----------

sql("""
  SELECT *
  FROM endpoints
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Running Endpoints
# MAGIC 
# MAGIC ### Concept
# MAGIC A SQL endpoint is a computation resource that lets you run SQL commands on data objects within the Databricks environment.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC When you create your first SQL endpoints, Databricks recommends that you accept the defaults as they appear on the New SQL endpoint page. But you have many options that you can configure to meet your specific needs.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Admin Guide ([AWS](https://docs.databricks.com/sql/admin/sql-endpoints.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/sql/admin/sql-endpoints))  
# MAGIC API ([AWS](https://docs.databricks.com/sql/api/sql-endpoints.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/sql/api/sql-endpoints))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table contains the list of all SQL endpoints that are currently running on the workspace as well as their configurations.

# COMMAND ----------

sql("""
  SELECT *
  FROM endpoints
  WHERE state IN ("STARTING", "RUNNING")
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Photon
# MAGIC 
# MAGIC ### Concept
# MAGIC Photon is the native vectorized query engine on Databricks, written to be directly compatible with Apache Spark APIs so it works with your existing code. It is developed in C++ to take advantage of modern hardware, and uses the latest techniques in vectorized query processing to capitalize on data- and instruction-level parallelism in CPUs, enhancing performance on real-world data and applications-—all natively on your data lake. Photon is part of a high-performance runtime that runs your existing SQL and DataFrame API calls faster and reduces your total cost per workload.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC Photon is enabled by default in Databricks SQL endpoints.
# MAGIC 
# MAGIC Keep Photon on to ensure that queries are executed on the Photon native vectorized engine that speeds up query execution. Databricks recommends against disabling Photon and plans to remove the off option in a future release. However, there may be circumstances in which a Databricks support representative recommends that you disable it. In that case, expand Advanced options to turn Photon off.
# MAGIC 
# MAGIC It is important to know that Photon has a few limitations:
# MAGIC - Works on Delta and Parquet tables only for both read and write.
# MAGIC - Does not support window and sort operators
# MAGIC - Does not support Spark Structured Streaming.
# MAGIC - Does not support UDFs.
# MAGIC - Not expected to improve short-running queries (<2 seconds), for example, queries against small amounts of data.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Product ([Databricks](https://databricks.com/product/photon))  
# MAGIC Blog ([Databricks](https://databricks.com/blog/2021/06/17/announcing-photon-public-preview-the-next-generation-query-engine-on-the-databricks-lakehouse-platform.html))  
# MAGIC User Guide ([AWS](https://docs.databricks.com/runtime/photon.html)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/runtime/photon))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table shows the Databricks SQL endpoints that have Photon enabled.

# COMMAND ----------

sql("""
  SELECT enable_photon, name, id
  FROM endpoints
  ORDER BY 1 DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scaling
# MAGIC 
# MAGIC ### Concept
# MAGIC Scaling sets the minimum and maximum number of clusters over which queries sent to the endpoint are distributed.
# MAGIC 
# MAGIC The default is a minimum of one and maximum of one cluster.
# MAGIC 
# MAGIC ### Best Practice
# MAGIC To handle more concurrent users for a given query, increase the cluster count. Databricks recommends a cluster for every ten concurrent queries.
# MAGIC 
# MAGIC ### Documentation
# MAGIC Queuing and Autoscaling ([AWS](https://docs.databricks.com/sql/admin/sql-endpoints.html#queueing-and-autoscaling)) ([Azure](https://docs.microsoft.com/en-us/azure/databricks/sql/admin/sql-endpoints#queueing-and-autoscaling))
# MAGIC 
# MAGIC ### Analysis
# MAGIC The table shows the current scaling of all of the Databricks SQL endpoints configured in the workspace.

# COMMAND ----------

sql("""
  SELECT num_clusters, max_num_clusters, min_num_clusters, state, name, id
  FROM endpoints
  ORDER BY 1, 2, 3 DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Cleanup

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Comment the next line to not delete the Database
# MAGIC 
# MAGIC DROP DATABASE workspace_analysis CASCADE
