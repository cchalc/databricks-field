# Databricks notebook source
# MAGIC %md
# MAGIC # Pandas User-Defined Functions in Spark 
# MAGIC <img src="https://github.com/billkellett/databricks-demo-pandas-udf/blob/master/images/pandas_icon2.png?raw=true" width=200 />
# MAGIC 
# MAGIC This notebook demonstrates how to use Pandas UDFs in Apache Spark.  This brief demo is based on:
# MAGIC 
# MAGIC - a blog post by Ji Lin, found here: https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html 
# MAGIC - Apache PySpark documentation: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.pandas_udf
# MAGIC - Databricks Pandas documentation: https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html
# MAGIC 
# MAGIC As Ji Lin notes, ordinary Python user-defined functions "operate one-row-at-a-time, and thus suffer from high serialization and invocation overhead. As a result, many data pipelines define UDFs in Java and Scala and then invoke them from Python.  Pandas UDFs built on top of Apache Arrow bring you the best of both worlds—the ability to define low-overhead, high-performance UDFs entirely in Python."
# MAGIC 
# MAGIC Pandas UDFs can provide much higher performance for two reasons:
# MAGIC 
# MAGIC - They use code vectorization.  For a great introduction to vectorization, see https://stackoverflow.com/questions/1422149/what-is-vectorization 
# MAGIC - They use a standard data format provided by Apache Arrow.  For an introduction to Apache Arrow, see https://www.dremio.com/apache-arrow-explained/
# MAGIC 
# MAGIC ## What types of UDFs does Pandas support?
# MAGIC 
# MAGIC As of February 2019, three types of Pandas UDFs are supported.  We'll be looking at each of them in this demo:
# MAGIC 
# MAGIC - __SCALAR__
# MAGIC - __GROUPED_MAP__
# MAGIC - __GROUPED_AGG__
# MAGIC 
# MAGIC <img src="https://github.com/billkellett/databricks-demo-pandas-udf/blob/master/images/caution.jpg?raw=true" width=100/> 
# MAGIC 
# MAGIC Note that this demo will __*not*__ attempt to prove that Pandas UDFs are faster than ordinary Python UDFs.  That sort of performance testing would require:
# MAGIC 
# MAGIC - ...bigger data than we will be using, which would mean longer job run-times
# MAGIC - ...more complex UDFs than we will be using, which would make the demo more confusing.  Pandas UDFs show their value best when the code is complex enough to show the power of vectorization.
# MAGIC 
# MAGIC Instead, this demo will focus on __*how*__ to use Pandas UDFs in Spark.
# MAGIC 
# MAGIC ### Let's get started!
# MAGIC Before we start using UDFs, let's create some data and examine it.
# MAGIC 
# MAGIC The cell below creates a dataframe with ten million rows and two columns:
# MAGIC 
# MAGIC - __id__: an integer that increments every 10,000 rows
# MAGIC - __v__: a randomly-generated floating-point number between 0 and 1
# MAGIC 
# MAGIC We'll also register the dataframe as a temporary SQL View called MY_DATA.

# COMMAND ----------

from pyspark.sql.functions import col, rand

df = spark.range(0, 10 * 1000 * 1000).withColumn('id', (col('id') / 10000).cast('integer')).withColumn('v', rand())
df.cache()

df.createOrReplaceTempView("MY_DATA")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's just take a quick peek at the data
# MAGIC 
# MAGIC SELECT * FROM MY_DATA

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Now verify that we have 10 million rows
# MAGIC 
# MAGIC SELECT COUNT(*) FROM MY_DATA

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Our "id" column should increment every 10,000 rows, so we should have 1,000 distinct ids --- let's verify
# MAGIC 
# MAGIC SELECT COUNT(DISTINCT id) FROM MY_DATA

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify that the lowest value in the "id" column is 0 and the highest is 999
# MAGIC 
# MAGIC SELECT 
# MAGIC   MIN(id) as min,
# MAGIC   MAX(id) AS max FROM MY_DATA

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 1: Scalar UDF
# MAGIC 
# MAGIC Scalar Pandas UDFs are used for vectorizing scalar operations. 
# MAGIC 
# MAGIC To define a scalar Pandas UDF, simply use @pandas_udf to annotate a Python function that takes in pandas.Series as arguments and returns another pandas.Series.
# MAGIC 
# MAGIC Note that the returned pandas.Series must be the __same size__ as the input pandas.Series (because we are affecting every input row).
# MAGIC 
# MAGIC We'll create a very simple UDF that simply increments the number that is passed to it.
# MAGIC 
# MAGIC First we'll create an "ordinary" Python UDF.  Then we'll create the same UDF as a Pandas UDF, so you can see the minor changes that are necessary

# COMMAND ----------

# Here is a plain-old Python UDF that increments a number
# The parameter in the decorator is the return type

@udf('double')
def plus_one(v):
    return v + 1

# COMMAND ----------

# Here is the Pandas version of the above UDF.
# Note the following differences:
# - There are two imports required
# - The decorator is "pandas_udf" instead of "udf"
# - The function accepts a pandas Series and returns a pandas Series

import pandas as pd
from pyspark.sql.functions import pandas_udf 

@pandas_udf('double')
def pandas_plus_one(s: pd.Series) -> pd.Series:
    return s + 1

# COMMAND ----------

# Now let's run the "ordinary" UDF.  The UDF results are placed in a new column, v_plus_one.
# Note that Python precision may affect some of the low-order digits in the new column.

df_temp = df.withColumn('v_plus_one', plus_one(df.v))
display(df_temp)

# COMMAND ----------

# Now run the Pandas version.  
# Notice that the syntax is the same.

df_temp = df.withColumn('v_plus_one', pandas_plus_one(df.v))
display(df_temp)

# COMMAND ----------

# I can also do all of the above in SQL
# I'll register both of my UDFs so I can use them in SQL statements

spark.udf.register("PLUS_ONE", plus_one)
spark.udf.register("PANDAS_PLUS_ONE", pandas_plus_one)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Here we use the "plain-old" UDF
# MAGIC 
# MAGIC SELECT 
# MAGIC   id,
# MAGIC   v,
# MAGIC   PLUS_ONE(v) AS v_plus_one
# MAGIC FROM MY_DATA

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Here is the Pandas version.  Note that the syntax does not change.
# MAGIC 
# MAGIC SELECT 
# MAGIC   id,
# MAGIC   v,
# MAGIC   PANDAS_PLUS_ONE(v) AS v_plus_one
# MAGIC FROM MY_DATA

# COMMAND ----------

# MAGIC %md
# MAGIC ###Example 2: Grouped Map Pandas UDFs
# MAGIC 
# MAGIC *From Ji Lin's blog...*
# MAGIC 
# MAGIC Python users are fairly familiar with the split-apply-combine pattern in data analysis. Grouped map Pandas UDFs are designed for this scenario, and they operate on all the data for some group, e.g., “for each date, apply this operation”.
# MAGIC 
# MAGIC A grouped map Pandas UDFs first splits a Spark DataFrame into groups based on the conditions specified in the groupby operator, applies a user-defined function (pandas.DataFrame -> pandas.DataFrame) to each group, then combines and returns the results as a new Spark DataFrame.
# MAGIC 
# MAGIC Grouped map Pandas UDFs use the same function decorator pandas_udf as scalar Pandas UDFs, but they have a few differences:
# MAGIC 
# MAGIC __Input of the user-defined function:__
# MAGIC - Scalar: pandas.Series
# MAGIC - Grouped map: pandas.DataFrame
# MAGIC 
# MAGIC __Output of the user-defined function:__
# MAGIC - Scalar: pandas.Series
# MAGIC - Grouped map: pandas.DataFrame
# MAGIC 
# MAGIC __Grouping semantics:__
# MAGIC - Scalar: no grouping semantics
# MAGIC - Grouped map: defined by “groupby” clause
# MAGIC 
# MAGIC __Output size:__
# MAGIC - Scalar: same as input size
# MAGIC - Grouped map: any size
# MAGIC 
# MAGIC __Return types in the function decorator:__
# MAGIC - Scalar: a DataType that specifies the type of the returned pandas.Series
# MAGIC - Grouped map: a StructType that specifies each column name and type of the returned pandas.DataFrame
# MAGIC 
# MAGIC __NOTE__: Since we have already examined the differences between "plain-old" Python UDFs and Pandas UDFs, we won't repeat that exercise.  From here on, we'll simply show how to use Pandas UDFs.

# COMMAND ----------

# Let's add some additional columns to our data, for the UDF to use.

from pyspark.sql.functions import lit

df1 = df.withColumn('grp_avg', lit(0).cast('double')).withColumn('variance', lit(0).cast('double'))

# COMMAND ----------

df1.show(3)

# COMMAND ----------

# Here is a Pandas UDF of the GROUPED_MAP type.  
# It works with the same dataframe we used above.
# This UDF leverages the split-apply-combine paradigm to calculate the variance from the mean value for each group of rows with the same id

# Input and output to this UDF are both a pandas.DataFrame
# Note that we add 2 columns to the output pandas.DataFrame:
# - grp_avg: the average value of column "v" for each id group
# - variance: the "v" value of the row, minus the "grp_avg" value

# NOTE that GROUPED_MAP still uses the older syntax, which was deprecated for the other two udf types

from pyspark.sql.functions import PandasUDFType

@pandas_udf(df1.schema, PandasUDFType.GROUPED_MAP)
def pandas_subtract_mean(pdf):
  return pdf.assign(grp_avg=pdf.v.mean(), variance=pdf.v - pdf.v.mean())

# COMMAND ----------

# Run a groupby that calls the UDF

from pyspark.sql.functions import lit

df1 = df1.groupby('id').apply(pandas_subtract_mean)
display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Example 3: Grouped Aggregate Pandas UDFs
# MAGIC 
# MAGIC A grouped aggregate UDF defines a transformation.  The UDF accepts one or more pandas.Series, and returns a Scalar.  
# MAGIC 
# MAGIC The return type can either be a Python primitive type or a numpy data type.
# MAGIC 
# MAGIC See https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.pandas_udf for more details.

# COMMAND ----------

# This Pandas UDF calculates the average value of "v" for each group of rows with the same "id"

@pandas_udf('double')  
def mean_udf(s: pd.Series) -> 'double':
    return s.mean()

# COMMAND ----------

# Let's run it.  This should return 1 row for each unique id value (there should be 1000 rows)

df2 = df.groupby("id").agg(mean_udf(df['v']))
display(df2)

# COMMAND ----------

# Verify that there are 1000 rows

df2.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What just happened?
# MAGIC 
# MAGIC In this demo, we explored several use cases for Pandas UDFs.  Pandas UDFs provide superior performance over "plain-old" Python UDFs, and also provide some options for working with various data shapes. 
# MAGIC 
# MAGIC We examined three specific use cases:
# MAGIC 
# MAGIC - __Scalar UDF__: We modified a column in every row of the data
# MAGIC - __Grouped Map__: We leveraged the split-apply-combine capability of this Pandas UDF type to calculate a deviation from the group mean for every row in each group.
# MAGIC - __Grouped Aggregate__: We leveraged this Pandas UDF type to aggregate values into a single row per group.

# COMMAND ----------

# MAGIC %md
# MAGIC ### If you want to learn more and go deeper...
# MAGIC 
# MAGIC The following resources will help:
# MAGIC 
# MAGIC - a __blog post__ by Ji Lin, found here: https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html 
# MAGIC - Apache __PySpark documentation__: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.pandas_udf
# MAGIC - Databricks __Pandas documentation__: https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html
# MAGIC 
# MAGIC If you want to learn more about foundational technologies behind Pandas UDFs:
# MAGIC 
# MAGIC - __Code vectorization__: For a great introduction to vectorization, see https://stackoverflow.com/questions/1422149/what-is-vectorization 
# MAGIC - __Apache Arrow__: For an introduction to Apache Arrow, see https://www.dremio.com/apache-arrow-explained/
