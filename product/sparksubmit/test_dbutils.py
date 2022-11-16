#from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

from typing import Any, Tuple, Callable

from pyspark.sql import SparkSession, DataFrame
import logging
import IPython as ip
from pyspark.sql.types import StructType, ArrayType
import pyspark.sql.functions as f

#spark = SparkSession.builder.getOrCreate()
def _get_spark() -> SparkSession:
    user_ns = ip.get_ipython().user_ns
    if "spark" in user_ns:
        return user_ns["spark"]
    else:
        spark = SparkSession.builder.getOrCreate()
        user_ns["spark"] = spark
        return spark

def _get_dbutils(spark: SparkSession):
    try:
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
    except ImportError:
        import IPython
        dbutils = IPython.get_ipython().user_ns.get("dbutils")
        if not dbutils:
            log.warning("could not initialise dbutils!")
    return dbutils

def get_dbutils(spark):
    from pyspark.dbutils import DBUtils
    return DBUtils(spark)


# start ipython
#ip.start_ipython()
from IPython import embed

embed()

# Initialize variables
spark: SparkSession = _get_spark()
dbutils = _get_dbutils(spark)
#dbutils = DBUtils(spark)
#dbutils = get_dbutils(spark)

print(dbutils.help())
