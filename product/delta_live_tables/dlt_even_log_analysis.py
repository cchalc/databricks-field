# Databricks notebook source
# MAGIC %sql
# MAGIC DROP DATABASE IF EXISTS cchalc_wiki_demo CASCADE

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS cchalc
# MAGIC     COMMENT "CREATE A DATABASE WITH A LOCATION PATH"
# MAGIC     LOCATION "/Users/christopher.chalcraft@databricks.com/databases/cchalc" --this must be a location on dbfs (i.e. not direct access)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Review the top referrers to Wikipedia's Apache Spark articles
# MAGIC SELECT * FROM cchalc.clickstream_raw

# COMMAND ----------

# MAGIC %fs ls /Users/christopher.chalcraft@databricks.com/databases/cchalc/system/events

# COMMAND ----------

# MAGIC %sql
# MAGIC use cchalc

# COMMAND ----------

system_events_path = "/Users/christopher.chalcraft@databricks.com/databases/cchalc/system/events"
system_events = spark.read.format("delta").load(system_events_path)

# COMMAND ----------

system_events.write.mode("overwrite").saveAsTable("system_events")

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from cchalc.system_events

# COMMAND ----------

# MAGIC %scala
# MAGIC val json_parsed = spark.read.json(spark.table("cchalc.system_events").select("details").as[String])
# MAGIC val json_schema = json_parsed.schema

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.functions._
# MAGIC 
# MAGIC val parsed_event_log = spark.table("cchalc.system_events").withColumn("details_parsed", from_json($"details", json_schema))
# MAGIC 
# MAGIC parsed_event_log.createOrReplaceTempView("event_log")

# COMMAND ----------

# MAGIC %scala
# MAGIC parsed_event_log.write.format("delta").mode("overwrite").option("mergeSchema", "true").option("optimizeWrite", "true").saveAsTable("event_log")

# COMMAND ----------

display(spark.sql("select * from cchalc.event_log"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Lineage

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   details:flow_definition.output_dataset,
# MAGIC   details:flow_definition.input_datasets,
# MAGIC   details:flow_definition.flow_type,
# MAGIC   details:flow_definition.schema,
# MAGIC   details:flow_definition.explain_text,
# MAGIC   details:flow_definition
# MAGIC FROM cchalc.event_log
# MAGIC WHERE details:flow_definition IS NOT NULL
# MAGIC ORDER BY timestamp

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   details:flow_definition.output_dataset,
# MAGIC   details:flow_definition.input_datasets
# MAGIC FROM cchalc.event_log
# MAGIC WHERE details:flow_definition IS NOT NULL

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Flow Progress and Data Quality Results

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   id,
# MAGIC   details:flow_progress.metrics,
# MAGIC   details:flow_progress.data_quality.dropped_records,
# MAGIC   explode(from_json(details:flow_progress:data_quality:expectations
# MAGIC            ,schema_of_json("[{'name':'str', 'dataset':'str', 'passed_records':42, 'failed_records':42}]"))) expectations,
# MAGIC   details:flow_progress
# MAGIC FROM cchalc.event_log
# MAGIC WHERE details:flow_progress.metrics IS NOT NULL
# MAGIC ORDER BY timestamp

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   id,
# MAGIC   expectations.dataset,
# MAGIC   expectations.name,
# MAGIC   expectations.failed_records,
# MAGIC   expectations.passed_records
# MAGIC FROM(
# MAGIC   SELECT 
# MAGIC     id,
# MAGIC     timestamp,
# MAGIC     details:flow_progress.metrics,
# MAGIC     details:flow_progress.data_quality.dropped_records,
# MAGIC     explode(from_json(details:flow_progress:data_quality:expectations
# MAGIC              ,schema_of_json("[{'name':'str', 'dataset':'str', 'passed_records':42, 'failed_records':42}]"))) expectations
# MAGIC   FROM cchalc.event_log
# MAGIC   WHERE details:flow_progress.metrics IS NOT NULL) data_quality

# COMMAND ----------

event_log_expectations = spark.sql("""
SELECT
  id,
  expectations.dataset,
  expectations.name,
  expectations.failed_records,
  expectations.passed_records
FROM(
  SELECT 
    id,
    timestamp,
    details:flow_progress.metrics,
    details:flow_progress.data_quality.dropped_records,
    explode(from_json(details:flow_progress:data_quality:expectations
             ,schema_of_json("[{'name':'str', 'dataset':'str', 'passed_records':42, 'failed_records':42}]"))) expectations
  FROM cchalc.event_log
  WHERE details:flow_progress.metrics IS NOT NULL) data_quality
""")

# COMMAND ----------

display(event_log_expectations)

# COMMAND ----------

# MAGIC %fs ls /Users/christopher.chalcraft@databricks.com/databases/cchalc/system/events

# COMMAND ----------

event_log_expectations.printSchema()

# COMMAND ----------

# %sql
# CREATE TABLE event_log_expectations (
# id STRING,
# name STRING,
# failed_records LONG,
# passed_records LONG)
# USING DELTA
# LOCATION '/Users/christopher.chalcraft@databricks.com/databases/cchalc/system/event_log_expectations'

# COMMAND ----------

# MAGIC %scala
# MAGIC // event_log_expectations.write.format("delta").option("optimizeWrite", "true").saveAsTable("cchalc.event_log_expectations")

# COMMAND ----------

event_log_expectations.createOrReplaceTempView("event_log_expectations_tmpview")

# COMMAND ----------

spark.sql("create table event_log_expectations as select * from event_log_expectations_tmpview")
