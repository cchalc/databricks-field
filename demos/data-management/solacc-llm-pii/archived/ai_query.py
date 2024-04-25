# Databricks notebook source
# DBTITLE 1,AI_QUERY()
spark.sql(
    f"""
  CREATE OR REPLACE FUNCTION {CATALOG_NAME}.{SCHEMA_NAME}.pii_bot_categorise_department(message STRING)
  RETURNS STRUCT<candidates:ARRAY<STRUCT<text:STRING, metadata:STRUCT<finish_reason:STRING>>>, metadata:STRUCT<input_tokens:float, output_tokens:float, total_tokens:float>>
  COMMENT "Categorises departments a message may belong to"
  RETURN SELECT AI_QUERY("{MODEL_SERVING_ENDPOINT_NAME}",
      NAMED_STRUCT(
        "prompt", CONCAT("<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer. Always answer as helpfully as possible, while being safe. The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. If you don't know the answer to a question, please don't share false information.
<</SYS>>


", 'In the message provided below, determine which of our corporate departments the message should be routed to. 
It is possible that the message can belong to multiple departments.
Below are descriptions of our departments and their primary roles.
Pick only departments from this list. If you cannot determine the department, respond with UNCATEGORIZED.

- CUSTOMER_SUPPORT: support and aid our external customers with questions and issues
- CYBERSECURITY: protects our organisation against cybersecurity risks
- SECURITY: protects our organisation against security risks that are not cybersecurity risks
- TECHNICAL_SUPPORT: provides IT support for our staff and offices
- PEOPLE_OPS: support our staff with day to day HR (human resources) needs
- LEGAL: handle all legal and contractual matters
- FINANCE: handle all finance and finance operations

Return JSON only. Do not provide explanations or commentary.
JSON:
{{ "departments": [""] }}

Message to analyse:
', 
          message, "[/INST]"
        ),
        "temperature", 0.01,
        "max_tokens", 2048),
        "returnType", "STRUCT<candidates:ARRAY<STRUCT<text:STRING, metadata:STRUCT<finish_reason:STRING>>>, metadata:STRUCT<input_tokens:float, output_tokens:float, total_tokens:float>>"
  ) as generated_text
"""
)

