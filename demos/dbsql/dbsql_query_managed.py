from databricks import sql
import os

#from pyspark.sql import SparkSession, Row
#spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()

connection = sql.connect(
        server_hostname = os.getenv("DBSQL_CLI_HOST_NAME"),
        http_path = os.getenv("DBSQL_CLI_HTTP_PATH"),
        access_token = os.getenv("DBSQL_CLI_ACCESS_TOKEN")
        )

cursor = connection.cursor()
cursor.execute("SELECT distinct(zone) FROM cjc_h3_prod_api.taxi_zone_explode_h3")
data = cursor.fetchall()
print(data)

# parse Row data
#rdd=spark.sparkContext.parallelize(data)
#print(rdd.collect())

cursor.close()
connection.close()
