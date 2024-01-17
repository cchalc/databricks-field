# Databricks notebook source
project_name = 'recsys'
catalogs = ['cjc']
schemas = [f'{project_name}'] * len(catalogs)

for catalog, schema in zip(catalogs, schemas):
    # Create the catalog if it does not exist
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    # Create a schema called "my-mlops-project" in the catalog
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.`{schema}`")


# COMMAND ----------

catalog_name = "cjc"
schema_name = "recsys"
spark.sql(f"use catalog {catalog_name}")
spark.sql(f"use schema {schema_name}")

# COMMAND ----------


