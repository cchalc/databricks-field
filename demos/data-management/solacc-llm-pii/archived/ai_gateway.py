# Databricks notebook source
import mlflow
from mlflow.gateway import set_gateway_uri, create_route, search_routes, query

set_gateway_uri("databricks")

# COMMAND ----------

import mlflow
from mlflow.gateway import set_gateway_uri, create_route, search_routes, query

set_gateway_uri("databricks")

# COMMAND ----------

print(search_routes())

# COMMAND ----------

notebook_context  = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
databricks_url    = notebook_context.apiUrl().getOrElse(None)
workspace_url     = notebook_context.browserHostName().getOrElse(None)
print(workspace_url)

# COMMAND ----------

import json

print(json.dumps(
   json.loads(
     dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()), 
     indent=2
   )
)

# COMMAND ----------

print(search_routes())

# COMMAND ----------

pat = "xxxx"
route_name = "vinnyv-llama2-13b-chat-hf"

create_route(
   name=route_name,
   route_type="llm/v1/completions",
   model={
       "name": "vinnyv-llama2-13b-chat-hf",
       "provider": "databricks-model-serving",
        "databricks_model_serving_config": {
            "databricks_workspace_url": f"https://{workspace_url}",
            "databricks_api_token": pat,
        },
   }
)


# COMMAND ----------

notebook_context  = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
databricks_url    = notebook_context.apiUrl().getOrElse(None)
workspace_url     = notebook_context.browserHostName().getOrElse(None)
print(workspace_url)

# COMMAND ----------

route_name = "vinnyv-llama2-13b-chat-hf"
route_name = "vinnyv-mosaicml-inference"

response = query(
  route=route_name,
  data={"prompt": "What is MLflow?"},
)

print("executing")
print(response)
response_text = response["candidates"][0]["text"]
# assert response_text.startswith("MLflow is an open source ML platform")


# COMMAND ----------

import json

print(json.dumps(
   json.loads(
     dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()), 
     indent=2
   )
)

# COMMAND ----------

route_name = "vinnyv-mosaicml-inference"
mosaicml_api_key = "uGBIJ5.HpswLNIWb+RSD58cs0_z5zNw8QGZGVYHcd7nX4_5"

create_route(
    name=route_name,
    route_type="llm/v1/completions",
    model={
        "name": "llama2-70b-chat",
        "provider": "mosaicml",
        "mosaicml_config": {
            "mosaicml_api_key": mosaicml_api_key,
        },
    },
)


# COMMAND ----------

pat = "xxxx"
route_name = "vinnyv-llama2-13b-chat-hf"

create_route(
   name=route_name,
   route_type="llm/v1/completions",
   model={
       "name": "vinnyv-llama2-13b-chat-hf",
       "provider": "databricks-model-serving",
        "databricks_model_serving_config": {
            "databricks_workspace_url": f"https://{workspace_url}",
            "databricks_api_token": pat,
        },
   }
)


# COMMAND ----------

response = query(
  route_name,
  data={
    "prompt": """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. The primary purpose of your job is to meticulously identify privacy risks in any given documentation. You are thorough. If you don't know the answer to a question, please don't share false information.
<</SYS>>

You reply in valid JSON only. 
Personally identifiable information (PII) refers to any data that can be used to identify, contact, or locate a single person, either directly or indirectly.
PII includes but not limited to: valid email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver's licence numbers (LICENCE), etc.

Your task:
In the text to analyse below thoroughly detect all instances of PII. Do not ignore any potential PII.
Account for any spelling and grammatical errors.
Pay careful attention for person names. Use context clues, such as capitalisation, proper nouns (e.g. June), words following greetings (e.g. Hello), and surrounding text, to help identify person names. 
Brand and product names are not PII.
There can be multiple of the same PII type in the text. Return every instance detected.
Label each instance using format [PII_TYPE]. Do not return examples. Do not return empty or null values. 
Accuracy: Double-check your work to ensure that all PII is accurately detected and labeled.
Return valid JSON syntax only. Do not provide explanations. JSON:
{"pii_detected": [{"pii_type": "", "value": ""}]}
(e.g. {"pii_detected": [{"pii_type": "NAME", "value": "Harry},{"pii_type": "NAME", "value": "Susan"}]})


Text to analyse:
Hello Helena, your phone should be set up with the number 1-966-757-3629 x2936. If there are any issues with connectivity, please report them to IT by providing them with the phone's IMEI number, which is 27-544713-582252-7.
[/INST]
""",
  "temperature": 0.01,
  "max_tokens": 2048,
  "top_k": 10,
  }
)

response_text = response["candidates"][0]["text"]
print(response_text)

# COMMAND ----------

print(response_text)

# COMMAND ----------

import json

print(json.dumps(json.loads(response_text)))

# COMMAND ----------

route_name = "vinnyv-llama2-13b-chat-hf"
route_name = "vinnyv-mosaicml-inference"

response = query(
  route=route_name,
  data={"prompt": "What is MLflow?"},
)

print("executing")
print(response)
response_text = response["candidates"][0]["text"]
# assert response_text.startswith("MLflow is an open source ML platform")


# COMMAND ----------

response = query(
  route_name,
  data={
    "prompt": """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer assistant. Always answer as helpfully as possible, while being safe. Be succinct. The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. 

If you don't know the answer to a question, please don't share false information.
<</SYS>>

My previous instructions:
-----
YYou reply in valid JSON only. 
Personally identifiable information (PII) refers to any data that can be used to identify, contact, or locate a single person, either directly or indirectly.
PII includes but not limited to: valid email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver's licence numbers (LICENCE), etc.

Your task:
In the text to analyse below thoroughly detect all instances of PII. Do not ignore any text that is potentially PII.
Use the text's context clues to determine if a word is a name. Consider words that are capitalized, after greetings (e.g. Dear), or proper nouns as potential names. Names are PII.
Brand and product names are not PII.
There can be multiple of the same PII type in the text. Return every instance detected.
Label each instance using format [PII_TYPE]. Do not return examples. Do not return empty or null values. 
Accuracy: Double-check your work to ensure that all PII is accurately detected and labeled.
Return valid JSON syntax only. Do not provide explanations. JSON:
{"pii_detected": [{"pii_type": "", "value": ""}]}
(e.g. {"pii_detected": [{"pii_type": "NAME", "value": "Harry"},{"pii_type": "NAME", "value": "Susan"}]})


Text to analyse:
Attn: August, we need you to conduct a comprehensive study analyzing the impact of industrial culture on employee motivation. Please use your account Savings Account for any necessary purchases and keep track of all transactions for auditing purposes. Kind regards, Vinny - vinny.vijeyakumaar@databricks.com

Hello Helena, your phone should be set up with the number 1-966-757-3629 x2936. If there are any issues with connectivity, please report them to IT by providing them with the phone's IMEI number, which is 27-544713-582252-7.
-----

Your previous response:
{"pii_detected": [{"pii_type": "NAME", "value": "August"}, {"pii_type": "NAME", "value": "Vinny"}, {"pii_type": "EMAIL", "value": "vinny.vijeyakumaar@databricks.com"}, {"pii_type": "PHONE", "value": "1-966-757-3629 x2936"}, {"pii_type": "IMEI", "value": "27-544713-582252-7"}]}
-----

Your task: How can I improve my previous instruction? I previously asked you to detect PII in given text. My previous instructions are below. Helena was not detected as a name. How can I refine my instructions so you reliably identify people's names?
[/INST]
""",
    "temperature": 1,
    "max_tokens": 2048
  }
)

response_text = response["candidates"][0]["text"]
print(response_text)

# COMMAND ----------

route_name = "vinnyv-mosaicml-inference"
mosaicml_api_key = "uGBIJ5.HpswLNIWb+RSD58cs0_z5zNw8QGZGVYHcd7nX4_5"

create_route(
    name=route_name,
    route_type="llm/v1/completions",
    model={
        "name": "llama2-70b-chat",
        "provider": "mosaicml",
        "mosaicml_config": {
            "mosaicml_api_key": mosaicml_api_key,
        },
    },
)


# COMMAND ----------

response = query(
  route_name,
  data={
    "prompt": """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. The primary purpose of your job is to meticulously identify privacy risks in any given documentation. You are thorough. If you don't know the answer to a question, please don't share false information.
<</SYS>>

You reply in valid JSON only. 
Personally identifiable information (PII) refers to any data that can be used to identify, contact, or locate a single person, either directly or indirectly.
PII includes but not limited to: valid email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver's licence numbers (LICENCE), etc.

Your task:
In the text to analyse below thoroughly detect all instances of PII. Do not ignore any potential PII.
Account for any spelling and grammatical errors.
Pay careful attention for person names. Use context clues, such as capitalisation, proper nouns (e.g. June), words following greetings (e.g. Hello), and surrounding text, to help identify person names. 
Brand and product names are not PII.
There can be multiple of the same PII type in the text. Return every instance detected.
Label each instance using format [PII_TYPE]. Do not return examples. Do not return empty or null values. 
Accuracy: Double-check your work to ensure that all PII is accurately detected and labeled.
Return valid JSON syntax only. Do not provide explanations. JSON:
{"pii_detected": [{"pii_type": "", "value": ""}]}
(e.g. {"pii_detected": [{"pii_type": "NAME", "value": "Harry"},{"pii_type": "NAME", "value": "Susan"}]})


Text to analyse:
Hello Helena, your phone should be set up with the number 1-966-757-3629 x2936. If there are any issues with connectivity, please report them to IT by providing them with the phone's IMEI number, which is 27-544713-582252-7.
[/INST]
""",
    "temperature": 0.01,
    "max_tokens": 2048,
    "top_k": 10,
    "top_p": 0.65,
  }
)

response_text = response["candidates"][0]["text"]
# print(response_text)
print(json.dumps(json.loads(response_text)))
print(json.dumps(json.loads(response_text), indent=2))

# COMMAND ----------

response = query(
  route_name,
  data={
    "prompt": """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. The primary purpose of your job is to meticulously identify privacy risks in any given documentation. You are thorough. If you don't know the answer to a question, please don't share false information.
<</SYS>>

You reply in valid JSON only. 
Personally identifiable information (PII) refers to any data that can be used to identify, contact, or locate a single person, either directly or indirectly.
PII includes but not limited to: valid email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver's licence numbers (LICENCE), etc.

Your task:
In the text to analyse below thoroughly detect all instances of PII. Do not ignore any potential PII.
Account for any spelling and grammatical errors.
Pay careful attention for person names. Use context clues, such as capitalisation, proper nouns (e.g. June), words following greetings (e.g. Hello), and surrounding text, to help identify person names. 
Brand and product names are not PII.
There can be multiple of the same PII type in the text. Return every instance detected.
Label each instance using format [PII_TYPE]. Do not return examples. Do not return empty or null values. 
Accuracy: Double-check your work to ensure that all PII is accurately detected and labeled.
Return valid JSON syntax only. Do not provide explanations. JSON:
{"pii_detected": [{"pii_type": "", "value": ""}]}
(e.g. {"pii_detected": [{"pii_type": "NAME", "value": "Harry},{"pii_type": "NAME", "value": "Susan"}]})


Text to analyse:
Hello Helena, your phone should be set up with the number 1-966-757-3629 x2936. If there are any issues with connectivity, please report them to IT by providing them with the phone's IMEI number, which is 27-544713-582252-7.
[/INST]
""",
  "temperature": 0.01,
  "max_tokens": 2048,
  "top_k": 10,
  }
)

response_text = response["candidates"][0]["text"]
print(response_text)

# COMMAND ----------

route_name = "vinnyv-mosaicml-inference"
response = query(
  route_name,
  data={
    "prompt": """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Below is some text to analyse and mask.
We also have a list of items to mask in the text.
Here is the original list, with each item being a pair of string mask and text to mask:
-POSTAL_ADDRESS: Elise Run 8153
-ZIP_CODE: 00575-5476
-MOBILE_DEVICE_IMEI: 27-503440-983044-2

Step 1: create new working list
Create a new list by looking through the original list and remove any string masks or text that resembles the following: addresses, zip codes.
This is the new list you will work off. Replace the original list with this new list.

Step 2: redact text
Mask the below text using only the information in the new list using a string mask of format [STRING_MASK]. 
(e.g. if your new list has "EMAIL: joe@example.com", all occurences of "joe@example.com" would be replaced by "[EMAIL]").
Do not mask anything not in the new list.
Do not add any new words or characters into the original text besides the string mask.

Step 3: return your results
Carefully check your work ensuring you only masked what is in the new working list.
Finally, Respond with the new list and redacted text. Explain your redactions.


Text to analyse and mask using items in new list:
Hello, You requested us to update your bank details with the new address Elise Run 8153 with the zip code 00575-5476. To confirm this action, could you please respond to the security question we sent to your mobile device with IMEI number 27-503440-983044-2? Kind regards, Vinny - vinny.vijeyakumaar@databricks.com
[/INST]
""",
  "temperature": 0.01,
  "max_tokens": 2048
  }
)

response_text = response["candidates"][0]["text"]
print(response_text)

# COMMAND ----------

print(response_text)

# COMMAND ----------



# COMMAND ----------

import json

print(json.dumps(json.loads(response_text)))

# COMMAND ----------

response = query(
  route_name,
  data={
    "prompt": """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer assistant. Always answer as helpfully as possible, while being safe. Be succinct. The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. 

If you don't know the answer to a question, please don't share false information.
<</SYS>>

My previous instructions:
-----
YYou reply in valid JSON only. 
Personally identifiable information (PII) refers to any data that can be used to identify, contact, or locate a single person, either directly or indirectly.
PII includes but not limited to: valid email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver's licence numbers (LICENCE), etc.

Your task:
In the text to analyse below thoroughly detect all instances of PII. Do not ignore any text that is potentially PII.
Use the text's context clues to determine if a word is a name. Consider words that are capitalized, after greetings (e.g. Dear), or proper nouns as potential names. Names are PII.
Brand and product names are not PII.
There can be multiple of the same PII type in the text. Return every instance detected.
Label each instance using format [PII_TYPE]. Do not return examples. Do not return empty or null values. 
Accuracy: Double-check your work to ensure that all PII is accurately detected and labeled.
Return valid JSON syntax only. Do not provide explanations. JSON:
{"pii_detected": [{"pii_type": "", "value": ""}]}
(e.g. {"pii_detected": [{"pii_type": "NAME", "value": "Harry"},{"pii_type": "NAME", "value": "Susan"}]})


Text to analyse:
Attn: August, we need you to conduct a comprehensive study analyzing the impact of industrial culture on employee motivation. Please use your account Savings Account for any necessary purchases and keep track of all transactions for auditing purposes. Kind regards, Vinny - vinny.vijeyakumaar@databricks.com

Hello Helena, your phone should be set up with the number 1-966-757-3629 x2936. If there are any issues with connectivity, please report them to IT by providing them with the phone's IMEI number, which is 27-544713-582252-7.
-----

Your previous response:
{"pii_detected": [{"pii_type": "NAME", "value": "August"}, {"pii_type": "NAME", "value": "Vinny"}, {"pii_type": "EMAIL", "value": "vinny.vijeyakumaar@databricks.com"}, {"pii_type": "PHONE", "value": "1-966-757-3629 x2936"}, {"pii_type": "IMEI", "value": "27-544713-582252-7"}]}
-----

Your task: How can I improve my previous instruction? I previously asked you to detect PII in given text. My previous instructions are below. Helena was not detected as a name. How can I refine my instructions so you reliably identify people's names?
[/INST]
""",
    "temperature": 1,
    "max_tokens": 2048
  }
)

response_text = response["candidates"][0]["text"]
print(response_text)

# COMMAND ----------

response = query(
  route_name,
  data={
    "prompt": """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. The primary purpose of your job is to meticulously identify privacy risks in any given documentation. You are thorough. If you don't know the answer to a question, please don't share false information.
<</SYS>>

You reply in valid JSON only. 
Personally identifiable information (PII) refers to any data that can be used to identify, contact, or locate a single person, either directly or indirectly.
PII includes but not limited to: valid email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver's licence numbers (LICENCE), etc.

Your task:
In the text to analyse below thoroughly detect all instances of PII. Do not ignore any potential PII.
Account for any spelling and grammatical errors.
Pay careful attention for person names. Use context clues, such as capitalisation, proper nouns (e.g. June), words following greetings (e.g. Hello), and surrounding text, to help identify person names. 
Brand and product names are not PII.
There can be multiple of the same PII type in the text. Return every instance detected.
Label each instance using format [PII_TYPE]. Do not return examples. Do not return empty or null values. 
Accuracy: Double-check your work to ensure that all PII is accurately detected and labeled.
Return valid JSON syntax only. Do not provide explanations. JSON:
{"pii_detected": [{"pii_type": "", "value": ""}]}
(e.g. {"pii_detected": [{"pii_type": "NAME", "value": "Harry"},{"pii_type": "NAME", "value": "Susan"}]})


Text to analyse:
Hello Helena, your phone should be set up with the number 1-966-757-3629 x2936. If there are any issues with connectivity, please report them to IT by providing them with the phone's IMEI number, which is 27-544713-582252-7.
[/INST]
""",
    "temperature": 0.01,
    "max_tokens": 2048,
    "top_k": 10,
    "top_p": 0.65,
  }
)

response_text = response["candidates"][0]["text"]
# print(response_text)
print(json.dumps(json.loads(response_text)))
print(json.dumps(json.loads(response_text), indent=2))

# COMMAND ----------

route_name = "vinnyv-mosaicml-inference"
response = query(
  route_name,
  data={
    "prompt": """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Below is some text to analyse and mask.
We also have a list of items to mask in the text.
Here is the original list, with each item being a pair of string mask and text to mask:
-POSTAL_ADDRESS: Elise Run 8153
-ZIP_CODE: 00575-5476
-MOBILE_DEVICE_IMEI: 27-503440-983044-2

Step 1: create new working list
Create a new list by looking through the original list and remove any string masks or text that resembles the following: addresses, zip codes.
This is the new list you will work off. Replace the original list with this new list.

Step 2: redact text
Mask the below text using only the information in the new list using a string mask of format [STRING_MASK]. 
(e.g. if your new list has "EMAIL: joe@example.com", all occurences of "joe@example.com" would be replaced by "[EMAIL]").
Do not mask anything not in the new list.
Do not add any new words or characters into the original text besides the string mask.

Step 3: return your results
Carefully check your work ensuring you only masked what is in the new working list.
Finally, Respond with the new list and redacted text. Explain your redactions.


Text to analyse and mask using items in new list:
Hello, You requested us to update your bank details with the new address Elise Run 8153 with the zip code 00575-5476. To confirm this action, could you please respond to the security question we sent to your mobile device with IMEI number 27-503440-983044-2? Kind regards, Vinny - vinny.vijeyakumaar@databricks.com
[/INST]
""",
  "temperature": 0.01,
  "max_tokens": 2048
  }
)

response_text = response["candidates"][0]["text"]
print(response_text)

# COMMAND ----------


