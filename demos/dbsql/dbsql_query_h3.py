from databricks import sql
import os


query = """
  select
    cast(properties.location_id as int) as locationid,
    properties.borough,
    properties.zone,
    explode(h3_polyfillash3(geometry, 12)) as cell
  from
    cjc_h3_prod_api.taxi_zone_explode
  limit 20
"""

with sql.connect(
        server_hostname = os.getenv("DBSQL_CLI_HOST_NAME"),
        http_path = os.getenv("DBSQL_CLI_HTTP_PATH"),
        access_token = os.getenv("DBSQL_CLI_ACCESS_TOKEN")
        ) as connection:

    with connection.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchall()

        for row in result:
            print(row)
