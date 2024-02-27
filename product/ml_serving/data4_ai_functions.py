# Databricks notebook source
CATALOG_NAME = "cjc"
SCHEMA_NAME = "ml_serv"

spark.sql("CREATE CATALOG IF NOT EXISTS cjc")
spark.sql("CREATE SCHEMA IF NOT EXISTS cjc.ml_serv")
spark.sql("CREATE VOLUME IF NOT EXISTS cjc.ml_serv.myc")
spark.sql("use cjc.ml_serv")

# COMMAND ----------

import pandas as pd

pii_sentences = [
    "Alice Smith's email is alice.smith@email.com and lives at 123 Maple St, Springfield.",
    "Bob Johnson's phone number is 555-1234, residing at 456 Pine Lane, Lakeside.",
    "Carol White mentioned her SSN is 123-45-6789, currently at 789 Oak Ave, Rivertown.",
    "David Brown's license plate is ABC1234 and his address is 101 Birch Rd, Hilltown.",
    "Eve Davis shared her passport number, G12345678, while living at 202 Cedar St, Coastcity.",
    "Frank Moore's credit card number is 1234 5678 9012 3456, billing to 303 Elm St, Greentown.",
    "Grace Lee's driver's license is L123-4567-8901, with a domicile at 404 Aspen Way, Frostville.",
    "Henry Wilson's bank account number is 123456789, banking at 505 Walnut St, Sunnyvale.",
    "Ivy Young disclosed her birthdate, 01/02/1990, alongside her residence at 606 Pinecone Rd, Raincity.",
    "Jack Taylor's employee ID is 7890, working at 707 Redwood Blvd, Cloudtown.",
    "Kathy Green's insurance policy is INS-123456, covered at 808 Maple Grove, Windyville.",
    "Leo Carter mentioned his membership number, MEM-789123, frequenting 909 Cherry Lane, Starcity.",
    "Mia Ward's patient ID is PAT-456789, consulting at 1010 Willow Path, Moonville.",
    "Nathan Ellis's booking reference is REF1234567, staying at 1111 Ivy Green Rd, Sunnyside.",
    "Olivia Sanchez's pet's name is Whiskers, living together at 1212 Magnolia St, Petville.",
    "Peter Gomez's library card number is 1234567, a patron of 1313 Lilac Lane, Booktown.",
    "Quinn Torres is registered under the voter ID VOT-7890123, residing at 1414 Oakdale St, Voteville.",
    "Rachel Kim mentioned her alumni number, ALU-123789, belonging to 1515 Pine St, Gradtown.",
    "Steve Adams's gym membership is GYM-456123, exercising at 1616 Fir Ave, Muscleville.",
    "Tina Nguyen's loyalty card is LOY-789456, shopping at 1717 Spruce Way, Marketcity."
]

df = spark.createDataFrame(
    pd.DataFrame(
        {
            "unmasked_text": pii_sentences
        }
    )
)
display(df)

# COMMAND ----------

_ = (
  df.write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.pii_data")
)

# COMMAND ----------


