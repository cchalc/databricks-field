import dlt
from pyspark.sql.functions import expr, col

def get_rules(tag):
  """
    loads data quality rules from csv file
    :param tag: tag to match
    :return: dictionary of rules that matched the tag
  """
  rules = {}
  df = spark.read.format("csv").option("header", "true").load("/path/to/rules.csv")
  for row in df.filter(col("tag") == tag).collect():
    rules[row['name']] = row['constraint']
  return rules

@dlt.table(
  name="raw_farmers_market"
)
@dlt.expect_all_or_drop(get_rules('validity'))
def get_farmers_market_data():
  return (
    spark.read.format('csv').option("header", "true")
      .load('/databricks-datasets/data.gov/farmers_markets_geographic_data/data-001/')
  )

@dlt.table(
  name="organic_farmers_market"
)
@dlt.expect_all_or_drop(get_rules('maintained'))
def get_organic_farmers_market():
  return (
    dlt.read("raw_farmers_market")
      .filter(expr("Organic = 'Y'"))
      .select("MarketName", "Website", "State",
        "Facebook", "Twitter", "Youtube", "Organic",
        "updateTime"
      )
  )
