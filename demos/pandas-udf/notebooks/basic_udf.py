# Databricks notebook source
# MAGIC %md
# MAGIC # Basic UDF

# COMMAND ----------

import pandas as pd

# Example DataFrame
data = {'Column1': [1, 2, 3, 4, 5], 'Column2': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Transformation function to be applied
def transform(row):
    return row['Column1'] * row['Column2']

# Apply function in a for loop
result = []
for index, row in df.iterrows():
    result.append(transform(row))

print(result)


# COMMAND ----------

sdf = spark.createDataFrame(df)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType

# Define the pandas_udf
@pandas_udf(IntegerType())
def transform_udf(column1, column2):
    return column1 * column2


# COMMAND ----------

from pyspark.sql.functions import col

# Apply UDF and add a new column with results
result_df = sdf.withColumn("Result", transform_udf(col("Column1"), col("Column2")))

result_df.show()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # ApplyInPandas

# COMMAND ----------

from pyspark.sql import Row

# Example data
data = [
    Row(Group="A", Value=1),
    Row(Group="B", Value=2),
    Row(Group="A", Value=3),
    Row(Group="B", Value=4),
    Row(Group="C", Value=5)
]

# Create DataFrame
df = spark.createDataFrame(data)
df.show()


# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pandas as pd

# Define the output schema of the pandas_udf function
schema = StructType([
    StructField("Group", StringType()),
    StructField("TotalValue", IntegerType())
])

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def calculate_group_sum(pdf):
    # Perform the calculation on the pandas DataFrame
    result = pd.DataFrame({
        "Group": [pdf['Group'].iloc[0]],
        "TotalValue": [pdf['Value'].sum()]
    })
    return result

# Group by 'Group' column and apply the calculation
result_df = df.groupby("Group").applyInPandas(calculate_group_sum, schema)

result_df.show()


# COMMAND ----------


