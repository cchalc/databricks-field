def move_catalog_to(catalog_to_move, catalog_destination, schema_prefix = None):
    tables = spark.sql(f"SELECT * FROM system.information_schema.tables where table_catalog='{catalog_to_move}' and table_schema != 'information_schema'").collect()
    for table in tables:
        new_schema_name = table['table_schema']
        if schema_prefix is not None:
            new_schema_name = schema_prefix+"_"+new_schema_name
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog_destination}`.`{new_schema_name}`")
        sql = f"CREATE OR REPLACE TABLE `{catalog_destination}`.`{new_schema_name}`.`{table['table_name']}` DEEP CLONE `{catalog_to_move}`.`{table['table_schema']}`.`{table['table_name']}`"
        print(sql)
        spark.sql(sql)

# move_catalog_to("forrester_demo", "shared", schema_prefix="forrester_demo")