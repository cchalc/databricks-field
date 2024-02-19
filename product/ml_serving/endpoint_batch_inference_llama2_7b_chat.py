# Databricks notebook source
# MAGIC %md
# MAGIC # Query model from endpoint
# MAGIC
# MAGIC [docs](https://docs.databricks.com/en/machine-learning/foundation-models/query-foundation-model-apis.html)

# COMMAND ----------

!pip install databricks-genai-inference

dbutils.library.restartPython()


# COMMAND ----------

# Test it is working
from databricks_genai_inference import ChatCompletion


response = ChatCompletion.create(model="llama-2-70b-chat",
                                 messages=[{"role": "system", "content": "You are a helpful assistant."},
                                           {"role": "user","content": "Knock knock."}],
                                 max_tokens=128)
print(f"response.message:{response.message}")

# COMMAND ----------

# prompt complex

# system
SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. Always answer as helpfully as possible, while being safe. The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction}
[/INST]
"""

# additional instruction
USER_INSTRUCTION_PROFILE_PII = """You only consider what is in the below text to analyze. 
You reply in valid JSON only. Return a valid JSON object only. No other text. Do not explain your response. 

<start of text to analyze>
{text_to_analyse}
<end of text to analyze>

Defining PII:
Personally identifiable information (PII) refers to any data that can be used to identify, contact, or locate a single person, either directly or indirectly.
PII includes but not limited to: valid email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver's licence numbers (LICENCE), MAC addresses (MAC), etc.
Pay careful attention for a person's name: use context clues, such as capitalisation, proper nouns (e.g. June), words following greetings, words following honorifics, and surrounding text, to help identify people's names. 
Brand names and product names are not PII.

Your task:
In the text to analyze meticulously and thoroughly detect all instances of PII.
Account for any spelling and grammatical errors.
Label each instance using format [PII_TYPE].
You do not provide examples or sample data. Do not provide instances that is not in the text to analyze.
There can be multiple of the same PII type in the text. Return every instance detected.

Instructions for your JSON response:
You return an empty set if there is no PII detected.
If you found PII, verify your answer. If a PII type and value is not in the text to analyze, remove it from your answer.
Return a valid JSON object only. No other text. Do not explain your response. 
JSON: 
{{"pii_detected": [{{"pii_type": "", "value": ""}}]}}
Example: 
{{"pii_detected": [{{"pii_type": "IP_ADDRESS", "value": "23.23.12.0"}}]}}"""

# COMMAND ----------

# prompt basic

DEFAULT_SYSTEM_PROMPT = """\
You are only a bot in charge of redacting personally identifiable information (PII) such as names, addresses, phone numbers, social security numbers, etc., You only only take an instruction an repeat it back replacing the PII with [REDACTED] in the response. An example would be: Jimmy John lives in White Rock at 1234 Beach St and loves to play pickleball. Call for a game at 789-555-1234. This would be replaced with: [REDACTED] lives in [REDACTED] at [REDACTED] and loves to play pickleball. Call for a game at [REDACTED].
"""

PROMPT_FOR_GENERATION_FORMAT = """
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>

Repeat the following text back with the PII redacted: {instruction}
[/INST]
""".format(
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    instruction="{instruction}"
)

# COMMAND ----------

# Import the necessary modules
from pyspark.sql.functions import pandas_udf
import pandas as pd
import requests

# Then execute the rest of the code
api_root = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
MODEL_SERVING_ENDPOINT_NAME = "databricks-llama-2-70b-chat"

@pandas_udf("string")
def pii_bot(prompts: pd.Series) -> pd.Series:
    """
    Executes batch inference on a model with given prompts, formatted with a system prompt,
    and returns the primary text output for each prompt in a new pandas Series.

    Parameters:
    - prompts (pd.Series): Text prompts to be processed.

    Returns:
    - pd.Series: Primary text output from the model for each input prompt.

    Raises:
    - Exception: If the HTTP request to the model serving endpoint fails.
    """
    # Wrap system prompt around input prompts
    # formatted_prompts = prompts.map(
    #     lambda prompt: system_prompt.format(instruction=prompt)
    # )
    
    # Wrap system prompt around input prompts
    formatted_prompts = prompts.map(
        lambda prompt: PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
    )

    # Batch inference
    inference_data = {
        "inputs": {"prompt": formatted_prompts.apply(str).tolist()},
        "params": {"max_tokens": 2048, "temperature": 0.01},
    }

    headers = {
        "Context-Type": "text/json",
        "Authorization": f"Bearer {api_token}",
    }

    response = requests.post(
        url=f"{api_root}/serving-endpoints/{MODEL_SERVING_ENDPOINT_NAME}/invocations",
        json=inference_data,
        headers=headers,
    )

    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    response_json = response.json()

    results = []
    for prediction in response_json["predictions"]:
        results.append(prediction["candidates"][0]["text"])

    return pd.Series(results)

# COMMAND ----------

@pandas_udf("string")
def create_profiling_prompt(data: pd.Series) -> pd.Series:
    """
    Generate a profiling prompt for each element in the data series.

    Uses a predefined user instruction template (`user_instruction_profile_pii`)
    to format the text entries in the data series.

    Parameters:
    - data (pd.Series): A pandas Series containing text entries to be formatted.

    Returns:
    - pd.Series: A pandas Series containing the formatted profiling prompts.
    """

    prompts = data.map(
lambda text: USER_INSTRUCTION_PROFILE_PII.format(text_to_analyse=text)    )
    return prompts

# COMMAND ----------

CATALOG_NAME = "cjc"
SCHEMA_NAME = "ml_serv"

# COMMAND ----------

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

# Get a list of all tables in the specific schema
# display(spark.sql("SHOW TABLES IN cjc.ml_serv"))

# COMMAND ----------

# _ = spark.sql(
#     f"""
#       CREATE OR REPLACE TABLE {CATALOG_NAME}.{SCHEMA_NAME}.bronze_documents_pii_handled
#       (
#           message_id STRING COMMENT 'Unique identifier for the message',
#           unmasked_text STRING COMMENT 'Raw unmasked text content',
#           pii_detected {SCHEMA_PII_PROFILE_EXTRACTED} COMMENT 'List of PII detected with type and value'
#           masked_text STRING COMMENT 'Raw unmasked text content',

#       )
#       COMMENT "Messages with profiling information generated from PII Bot"
#       TBLPROPERTIES
#       (
#           "sensitivity" = "high",
#           "quality" = "bronze"
#       )
# """
# )

# COMMAND ----------

from pyspark.sql.functions import concat, lit, col

redact_instruction = "Given the following text, identify and replace any Personally Identifiable Information (PII) such as names, email addresses, phone numbers, physical addresses, and any other specific details that can identify an individual, with [REDACTED]. Ensure that the modified text maintains its original meaning as much as possible without revealing any PII: "

df_profile = df.withColumn("basic_profile", concat(lit(redact_instruction), df["unmasked_text"]))
display(df_profile)

# COMMAND ----------

df_id_pii_type = df.withColumn("prompt_profiling", create_profiling_prompt(col("unmasked_text")))
display(df_id_pii_type)

# COMMAND ----------

# Save the DataFrame "df_id_pii_type" as a table in Unified Catalog
df_id_pii_type.write.saveAsTable("input_data")

# COMMAND ----------

input_data = spark.read.table("input_data")
display(input_data)

# COMMAND ----------

df_categorised = input_data.select(
    col("unmasked_text"),
    pii_bot(col("prompt_profiling")).alias("resp"),
)

# COMMAND ----------

display(df_categorised)

# COMMAND ----------

# Test it is working
from databricks_genai_inference import ChatCompletion


response = ChatCompletion.create(model="llama-2-70b-chat",
                                 messages=[{"role": "system", "content": "You are only a bot in charge of redacting personally identifiable information (PII) such as names, addresses, phone numbers, social security numbers, etc., You only only take an instruction an repeat it back replacing the PII with [REDACTED] in the response. An example would be: Jimmy John lives in White Rock at 1234 Beach St and loves to play pickleball. Call for a game at 789-555-1234. This would be replaced with: [REDACTED] lives in [REDACTED] at [REDACTED] and loves to play pickleball. Call for a game at [REDACTED]."},
                                           {"role": "user","content": "Alice Smith's email is alice.smith@email.com and lives at 123 Maple St, Springfield."}],
                                 max_tokens=128)
print(f"response.message:{response.message}")

# COMMAND ----------

@pandas_udf("string")
def redact_pii_udf(text_series: pd.Series) -> pd.Series:
    responses = []
    for text in text_series:
        try:
            response = ChatCompletion.create(
                model="llama-2-70b-chat",
                messages=[
                    {"role": "system", "content": "You are only a bot in charge of redacting personally identifiable information (PII) such as names, addresses, phone numbers, social security numbers, etc., You only only take an instruction an repeat it back replacing the PII with [REDACTED] in the response. An example would be: Jimmy John lives in White Rock at 1234 Beach St and loves to play pickleball. Call for a game at 789-555-1234. This would be replaced with: [REDACTED] lives in [REDACTED] at [REDACTED] and loves to play pickleball. Call for a game at [REDACTED]."},
                    {"role": "user", "content": text}
                ],
                max_tokens=128
            )
            responses.append(response.message)
        except Exception as e:
            responses.append("Error: " + str(e))
    return pd.Series(responses)



# COMMAND ----------

# try on a pandas dataframe

def redact_pii(row):
    # Assuming the column containing the text to be redacted is named "text"
    user_content = row["unmasked_text"]
    
    try:
        response = ChatCompletion.create(
            model="llama-2-70b-chat",
            messages=[
                {"role": "system", "content": "You are only a bot in charge of redacting personally identifiable information (PII) such as names, addresses, phone numbers, social security numbers, etc., You only take an instruction and repeat it back replacing the PII with [REDACTED] in the response. An example would be: Jimmy John lives in White Rock at 1234 Beach St and loves to play pickleball. Call for a game at 789-555-1234. This would be replaced with: [REDACTED] lives in [REDACTED] at [REDACTED] and loves to play pickleball. Call for a game at [REDACTED]."},
                {"role": "user", "content": user_content}
            ],
            max_tokens=128
        )
        return response.message
    except Exception as e:
        # Handle exceptions or errors in the API call
        print(f"Error processing row: {e}")
        return "Error processing text"



# COMMAND ----------

pdf = df.toPandas()
pdf.head()

# COMMAND ----------

pdf['masked_text'] = pdf.apply(redact_pii, axis=1)

# COMMAND ----------

pdf.unmasked_text[4]

# COMMAND ----------

# df_redacted = df.limit(1).withColumn("masked_text", redact_pii_udf(df.unmasked_text))
# display(df_redacted)

## ERROR
# [Monitor - Restart - Reason: exceeded memory limit of 500000000
# SparkRuntimeException: [UDF_ERROR.INTERNAL_] Execution of function redact_pii_udf(unmasked_text#79005) failed 

# COMMAND ----------



# COMMAND ----------


