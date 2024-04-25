# Databricks notebook source
# MAGIC %md
# MAGIC ### Generate some random data to test

# COMMAND ----------

from pyspark.sql import SparkSession
import random

num_rows = 100000
numerical_vars = ['num_var1', 'num_var2', 'num_var3', 'num_var4']
categorical_vars = ['cat_var1', 'cat_var2', 'cat_var3', 'cat_var4']
target_variables = ['target1', 'target2']

# Generate dummy data
data = []
for _ in range(num_rows):
    row = {
        # Generate random integers for numerical variables
        'num_var1': random.randint(1, 100),
        'num_var2': random.randint(1, 100),
        'num_var3': random.randint(1, 100),
        'num_var4': random.randint(1, 100),
        # Generate random categories for categorical variables
        'cat_var1': random.choice(['A', 'B', 'C']),
        'cat_var2': random.choice(['X', 'Y', 'Z']),
        'cat_var3': random.choice(['D', 'E', 'F']),
        'cat_var4': random.choice(['U', 'V', 'W']),
        # Randomly assign 0 or 1 to target variables
        'target1': random.randint(0, 1),
        'target2': random.randint(0, 1)
    }
    data.append(row)

# Create a Spark DataFrame
risk_app = spark.createDataFrame(data)

risk_app.show(5)


# COMMAND ----------

display(risk_app)

# COMMAND ----------


