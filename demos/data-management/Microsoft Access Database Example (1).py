# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ### Step # 1
# MAGIC
# MAGIC Install `net.sf.ucanaccess:ucanaccess:5.0.1` on the cluster through Maven Central.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step # 2
# MAGIC Copy Microsoft Access Database to DBFS.

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://download.microsoft.com/download/e/9/8/e981203a-d902-4d63-afbf-424027b1e88c/olympicmedals.accdb
# MAGIC mkdir -p /dbfs/FileStore/accessdb/
# MAGIC mv olympicmedals.accdb /dbfs/FileStore/accessdb/

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step # 3
# MAGIC In spark we can create and register custom jdbc dialect for ucanaccess jdbc driver like following:

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.jdbc.{JdbcDialect, JdbcDialects}
# MAGIC
# MAGIC case object MSAccessJdbcDialect extends JdbcDialect {
# MAGIC   override def canHandle(url: String): Boolean = url.startsWith("jdbc:ucanaccess")
# MAGIC   override def quoteIdentifier(colName: String): String = s"[$colName]"
# MAGIC }
# MAGIC
# MAGIC
# MAGIC JdbcDialects.registerDialect(MSAccessJdbcDialect)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step # 4
# MAGIC Query Access Database using Spark.

# COMMAND ----------

# MAGIC %md Get list of tables.

# COMMAND ----------

# MAGIC %python
# MAGIC connectionProperties = {
# MAGIC   "driver" : "net.ucanaccess.jdbc.UcanaccessDriver"
# MAGIC }
# MAGIC
# MAGIC url = "jdbc:ucanaccess:///dbfs/FileStore/accessdb/olympicmedals.accdb"
# MAGIC tables = spark.read.jdbc(url=url, table="information_schema.tables", properties=connectionProperties)
# MAGIC
# MAGIC display(df.filter(df.TABLE_SCHEMA=="PUBLIC"))

# COMMAND ----------

# MAGIC %python
# MAGIC connectionProperties = {
# MAGIC   "driver" : "net.ucanaccess.jdbc.UcanaccessDriver"
# MAGIC }
# MAGIC
# MAGIC url = "jdbc:ucanaccess:///dbfs/FileStore/accessdb/olympicmedals.accdb"
# MAGIC df = spark.read.jdbc(url=url, table="MEDALS", properties=connectionProperties)
# MAGIC
# MAGIC display(df)
