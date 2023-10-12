SELECT * FROM system.information_schema.catalogs catalogs
  where catalog_name not in (SELECT distinct(catalog_name) FROM system.information_schema.catalog_tags WHERE lower(tag_name) like "remove%" )
  and catalog_owner = current_user();
