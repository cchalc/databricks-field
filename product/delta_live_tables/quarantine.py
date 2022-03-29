# Databricks notebook source
# https://docs.databricks.com/data-engineering/delta-live-tables/delta-live-tables-cookbook.html#quarantine-invalid-data

import dlt

rules = {}
quarantine_rules = {}

rules["valid_website"] = "(Website IS NOT NULL)"
rules["valid_location"] = "(Location IS NOT NULL)"

# concatenate inverse rules
quarantine_rules["invalid_record"] = "NOT({0})".format(" AND ".join(rules.values()))

@dlt.table(
  name="raw_farmers_market"
)
def get_farmers_market_data():
  return (
    spark.read.format('csv').option("header", "true")
      .load('/databricks-datasets/data.gov/farmers_markets_geographic_data/data-001/')
  )

@dlt.table(
  name="valid_farmers_market"
)
@dlt.expect_all_or_drop(rules)
def get_valid_farmers_market():
  return (
    dlt.read("raw_farmers_market")
      .select("MarketName", "Website", "Location", "State",
              "Facebook", "Twitter", "Youtube", "Organic", "updateTime")
  )

@dlt.table(
  name="invalid_farmers_market"
)
@dlt.expect_all_or_drop(quarantine_rules)
def get_invalid_farmers_market():
  return (
    dlt.read("raw_farmers_market")
      .select("MarketName", "Website", "Location", "State",
              "Facebook", "Twitter", "Youtube", "Organic", "updateTime")
  )
