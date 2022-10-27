from databricks import sql
import os

with sql.connect(
        server_hostname = os.getenv("DBSQL_CLI_HOST_NAME"),
        http_path = os.getenv("DBSQL_CLI_HTTP_PATH"),
        access_token = os.getenv("DBSQL_CLI_ACCESS_TOKEN")
        ) as connection:

    with connection.cursor() as cursor:
        cursor.execute("SELECT distinct(zone) FROM cjc_h3_prod_api.taxi_zone_explode_h3")
        result = cursor.fetchall()

        for row in result:
            print(row)
