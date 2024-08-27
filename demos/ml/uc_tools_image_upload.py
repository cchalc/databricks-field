# Databricks notebook source
# MAGIC %pip install openai slack_sdk dspy-ai databricks-sql-connector sqlalchemy  --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG users;
# MAGIC USE abhilash_r;

# COMMAND ----------

import dspy
import pandas as pd

# COMMAND ----------

text_to_sql_name="databricks-dbrx-instruct"
# text_to_sql_name="databricks-meta-llama-3-70b-instruct" 
# text_to_sql_name="shj-llama3-8b-sqlcoder"

# sql_to_answer_name="databricks-meta-llama-3-70b-instruct"
sql_to_answer_name="databricks-dbrx-instruct"

db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

text_to_sql = dspy.Databricks(model=text_to_sql_name, 
                              max_tokens=1024,
                              temperature=0.1,
                              api_key=db_token,
                              api_base="https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/")
                              
sql_to_answer = dspy.Databricks(model=sql_to_answer_name,
                                max_tokens=1024,
                                api_key=db_token,
                                api_base="https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/")

# COMMAND ----------

# DSPy signature for converting text to SQL query
class TextToSQLAnswer(dspy.Signature):
    """Convert natural language text to SQL using suitable schema(s)."""

    question:str = dspy.InputField(desc="natural language input which will be converted to SQL")
    relevant_table_schemas:str = dspy.InputField(desc="These are the DDLs for the tables with description, column names and columm data types.")
    sql:str = dspy.OutputField(desc="Generate syntactically correct query with correct column names using suitable tables(s).\nAlways use table aliases in the SQL.\nAlways use LOWER and LIKE clause with wildcards for string type columns\n syntax for date_sub function is date_sub (current_date(), n days). \n DON'T OUTPUT anything else other than the query")

# DSPy signature for converting SQL query and question to natural language text
class SQLReturnToAnswer(dspy.Signature):
    """Summarise the answer from the query result to give one consisce output"""

    question:str = dspy.InputField()
    sql:str = dspy.InputField(desc="sql query that generated the rows")
    relevant_rows:str = dspy.InputField(desc="relevant rows to answer the question")
    answer:str = dspy.OutputField(desc="answer to the question using relevant rows and the sql query")

# If there is an SQLError, then rectify the error by trying again
class SQLRectifier(dspy.Signature):
    """Correct the SQL query to resolve the error using the proper table names, columns and rows"""

    input_sql:str = dspy.InputField(desc="sql query that needs to be fixed")
    error_str: str = dspy.InputField(desc="error that needs to be resolved")
    relevant_table_schemas:str = dspy.InputField(desc="Multiple possible tables which has table name and corresponding columns, along with relevant rows from the table (values in the same order as columns above)")
    sql:str = dspy.OutputField(desc="corrected sql query to resolve the error and remove and any invalid syntax in the query.\n Don't output anything else other than the sql query")

# COMMAND ----------

sql_generator = dspy.ChainOfThought(TextToSQLAnswer)
# sql_generator = dspy.Predict(TextToSQLAnswer)

sql_result_summarizer = dspy.Predict(SQLReturnToAnswer)

sql_rectifier = dspy.ChainOfThought(SQLRectifier, rationale_type=dspy.OutputField(
            prefix="Reasoning: Let's think step by step in order to",
            desc="${produce the answer}. We ..."
        ))

# COMMAND ----------

from databricks import sql

def get_ddl_statements(schema_name: str) -> str:
    # Retrieve the list of tables in the schema
    tables = spark.sql(f"SHOW TABLES IN {schema_name}").collect()
    table_names = []
    table_create_stmts = []
    table_desc_list = []
    for table in tables:
        table_name = table["tableName"]
        table_names.append(table_name)
        table_create_stmts.append(spark.sql(f"SHOW CREATE TABLE {schema_name}.{table_name}").collect()[0][0])
        table_desc_list.append(spark.sql(f"DESCRIBE DETAIL {schema_name}.{table_name}").collect()[0].description)

    data_dict = {"TableName":table_names, "CreateTableStatement":table_create_stmts, "TableDescription":table_desc_list}
    data_pdf = pd.DataFrame(data_dict)
    ddl_statements = []
    for key, value in data_pdf.iterrows():
        create_statement = value["CreateTableStatement"]
        tblprop_idx = create_statement.index('TBLPROPERTIES')
        create_statement = create_statement[:tblprop_idx]
        create_statement = create_statement.replace("COMMENT ''", "")
        ddl_statements.append(create_statement)

    ddl_statements = '\n\n'.join(ddl_statements)
    return ddl_statements

def process_sql_str(sql_str:str):
    sql_str = sql_str.replace("```","")
    sql_str = sql_str.replace("sql","")
    sql_str = sql_str.strip()
    return sql_str

def execute_sql_statement(sql_statement: str, warehouse_id: str, host_name: str, db_token: str):
    connection = sql.connect(
        server_hostname=host_name,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        access_token=db_token,
    )

    result_str = ""
    with connection.cursor() as cursor:
        cursor.execute(sql_statement)
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            for col, value in zip(columns, row):
                result_str += f" {col}={value},"
            result_str = result_str[:-1]
            result_str += "\n"
    return result_str

# COMMAND ----------

def retriable_execution(question:str, max_retries:int=3, 
                        warehouse_id:str="5ab5dda58c1ea16b", 
                        uc_catalog_schema="users.abhilash_r"):
  
  ddl_input = get_ddl_statements(uc_catalog_schema)
  sql_query = sql_generator(question=question, relevant_table_schemas=ddl_input)
  sql_statement = sql_query.sql
  num_try=0
  while num_try <= max_retries:
      try:
        sql_str = process_sql_str(sql_statement)
        print('\nSQL STATEMENT-\n', sql_str)
        results_string = execute_sql_statement(sql_str, warehouse_id, db_host, db_token)
        print('\nSQL RESULTS-\n', results_string)
        break
      except Exception as error:
        print(('ERROR-\n', str(error)))
        sql_query = sql_rectifier(input_sql=sql_query.sql, 
                                  error_str=str(error),
                                  relevant_table_schemas=ddl_input)
        sql_statement = sql_query.sql
        num_try = num_try+1
        if num_try == max_retries:
          print('MAX RETRIES REACHED', num_try)
          raise error

  with dspy.context(lm=sql_to_answer):
    final_answer = sql_result_summarizer(question=question, sql=sql_statement, relevant_rows=results_string)
    return final_answer.answer

# COMMAND ----------

question = "what are the top 3 products sold in the last 1 month"

dspy.settings.configure(lm=text_to_sql)
product_summ = retriable_execution(question)

print("\nLLM OUTPUT-\n", product_summ)

# COMMAND ----------

question = "Give me a summary based on top 3 positive reviews"

dspy.settings.configure(lm=text_to_sql)
review_summ = retriable_execution(question)

print("\nLLM OUTPUT-\n", product_summ)

# COMMAND ----------

base64_image= "PHN2ZyB3aWR0aD0iMTMyIiBoZWlnaHQ9IjIyIiB2aWV3Qm94PSIwIDAgMTMyIDIyIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xOC4zMTc2IDkuMjc0NzlMOS42ODY2MyAxNC4xMzM5TDAuNDQ0NTExIDguOTQyMjNMMCA5LjE4MjQxVjEyLjk1MTVMOS42ODY2MyAxOC4zODMzTDE4LjMxNzYgMTMuNTQyN1YxNS41MzgxTDkuNjg2NjMgMjAuMzk3MkwwLjQ0NDUxMSAxNS4yMDU1TDAgMTUuNDQ1N1YxNi4wOTIzTDkuNjg2NjMgMjEuNTI0MkwxOS4zNTQ3IDE2LjA5MjNWMTIuMzIzM0wxOC45MTAyIDEyLjA4MzFMOS42ODY2MyAxNy4yNTYzTDEuMDM3MTkgMTIuNDE1N1YxMC40MjAzTDkuNjg2NjMgMTUuMjYwOUwxOS4zNTQ3IDkuODI5MDZWNi4xMTU0NUwxOC44NzMyIDUuODM4MzFMOS42ODY2MyAxMC45OTNMMS40ODE3IDYuNDExMDZMOS42ODY2MyAxLjgxMDYxTDE2LjQyODQgNS41OTgxM0wxNy4wMjExIDUuMjY1NTdWNC44MDM2N0w5LjY4NjYzIDAuNjgzNTk0TDAgNi4xMTU0NVY2LjcwNjY3TDkuNjg2NjMgMTIuMTM4NUwxOC4zMTc2IDcuMjc5NDJWOS4yNzQ3OVoiIGZpbGw9IiNFRTNEMkMiPjwvcGF0aD48cGF0aCBkPSJNMzcuNDQ5IDE4LjQ0MjdWMS44NTE1NkgzNC44OTMxVjguMDU5NEMzNC44OTMxIDguMTUxNzcgMzQuODM3NSA4LjIyNTY4IDM0Ljc0NDkgOC4yNjI2M0MzNC42NTIzIDguMjk5NTggMzQuNTU5NyA4LjI2MjYzIDM0LjUwNDEgOC4yMDcyQzMzLjYzMzYgNy4xOTEwNCAzMi4yODE2IDYuNjE4MjkgMzAuNzk5OSA2LjYxODI5QzI3LjYzMjcgNi42MTgyOSAyNS4xNTA5IDkuMjc4NzkgMjUuMTUwOSAxMi42NzgzQzI1LjE1MDkgMTQuMzQxMSAyNS43MjUgMTUuODc0NiAyNi43ODA4IDE3LjAwMTZDMjcuODM2NSAxOC4xMjg3IDI5LjI2MjYgMTguNzM4MyAzMC43OTk5IDE4LjczODNDMzIuMjYzMSAxOC43MzgzIDMzLjYxNTEgMTguMTI4NyAzNC41MDQxIDE3LjA3NTVDMzQuNTU5NyAxNy4wMDE2IDM0LjY3MDggMTYuOTgzMiAzNC43NDQ5IDE3LjAwMTZDMzQuODM3NSAxNy4wMzg2IDM0Ljg5MzEgMTcuMTEyNSAzNC44OTMxIDE3LjIwNDlWMTguNDQyN0gzNy40NDlaTTMxLjM1NTUgMTYuNDI4OUMyOS4zMTgyIDE2LjQyODkgMjcuNzI1MyAxNC43ODQ1IDI3LjcyNTMgMTIuNjc4M0MyNy43MjUzIDEwLjU3MjEgMjkuMzE4MiA4LjkyNzc1IDMxLjM1NTUgOC45Mjc3NUMzMy4zOTI4IDguOTI3NzUgMzQuOTg1NyAxMC41NzIxIDM0Ljk4NTcgMTIuNjc4M0MzNC45ODU3IDE0Ljc4NDUgMzMuMzkyOCAxNi40Mjg5IDMxLjM1NTUgMTYuNDI4OVoiIGZpbGw9ImJsYWNrIj48L3BhdGg+PHBhdGggZD0iTTUxLjExOCAxOC40NDM1VjYuODk2Mkg0OC41ODA2VjguMDYwMTdDNDguNTgwNiA4LjE1MjU0IDQ4LjUyNSA4LjIyNjQ1IDQ4LjQzMjQgOC4yNjM0QzQ4LjMzOTggOC4zMDAzNSA0OC4yNDcyIDguMjYzNCA0OC4xOTE2IDguMTg5NUM0Ny4zMzk3IDcuMTczMzMgNDYuMDA2MSA2LjYwMDU5IDQ0LjQ4NzQgNi42MDA1OUM0MS4zMjAyIDYuNjAwNTkgMzguODM4NCA5LjI2MTA5IDM4LjgzODQgMTIuNjYwNkMzOC44Mzg0IDE2LjA2MDEgNDEuMzIwMiAxOC43MjA2IDQ0LjQ4NzQgMTguNzIwNkM0NS45NTA2IDE4LjcyMDYgNDcuMzAyNiAxOC4xMTA5IDQ4LjE5MTYgMTcuMDM5NEM0OC4yNDcyIDE2Ljk2NTUgNDguMzU4MyAxNi45NDcgNDguNDMyNCAxNi45NjU1QzQ4LjUyNSAxNy4wMDI0IDQ4LjU4MDYgMTcuMDc2MyA0OC41ODA2IDE3LjE2ODdWMTguNDI1SDUxLjExOFYxOC40NDM1Wk00NS4wNjE1IDE2LjQyOTdDNDMuMDI0MiAxNi40Mjk3IDQxLjQzMTQgMTQuNzg1MyA0MS40MzE0IDEyLjY3OTFDNDEuNDMxNCAxMC41NzI5IDQzLjAyNDIgOC45Mjg1MiA0NS4wNjE1IDguOTI4NTJDNDcuMDk4OSA4LjkyODUyIDQ4LjY5MTcgMTAuNTcyOSA0OC42OTE3IDEyLjY3OTFDNDguNjkxNyAxNC43ODUzIDQ3LjA5ODkgMTYuNDI5NyA0NS4wNjE1IDE2LjQyOTdaIiBmaWxsPSJibGFjayI+PC9wYXRoPjxwYXRoIGQ9Ik03Mi44NDI2IDE4LjQ0MzVWNi44OTYySDcwLjMwNTJWOC4wNjAxN0M3MC4zMDUyIDguMTUyNTQgNzAuMjQ5NiA4LjIyNjQ1IDcwLjE1NyA4LjI2MzRDNzAuMDY0NCA4LjMwMDM1IDY5Ljk3MTggOC4yNjM0IDY5LjkxNjIgOC4xODk1QzY5LjA2NDMgNy4xNzMzMyA2Ny43MzA3IDYuNjAwNTkgNjYuMjEyIDYuNjAwNTlDNjMuMDI2MyA2LjYwMDU5IDYwLjU2MyA5LjI2MTA5IDYwLjU2MyAxMi42NzkxQzYwLjU2MyAxNi4wOTcxIDYzLjA0NDggMTguNzM5MSA2Ni4yMTIgMTguNzM5MUM2Ny42NzUyIDE4LjczOTEgNjkuMDI3MiAxOC4xMjk0IDY5LjkxNjIgMTcuMDU3OEM2OS45NzE4IDE2Ljk4MzkgNzAuMDgyOSAxNi45NjU1IDcwLjE1NyAxNi45ODM5QzcwLjI0OTYgMTcuMDIwOSA3MC4zMDUyIDE3LjA5NDggNzAuMzA1MiAxNy4xODcyVjE4LjQ0MzVINzIuODQyNlpNNjYuNzg2MSAxNi40Mjk3QzY0Ljc0ODggMTYuNDI5NyA2My4xNTYgMTQuNzg1MyA2My4xNTYgMTIuNjc5MUM2My4xNTYgMTAuNTcyOSA2NC43NDg4IDguOTI4NTIgNjYuNzg2MSA4LjkyODUyQzY4LjgyMzUgOC45Mjg1MiA3MC40MTYzIDEwLjU3MjkgNzAuNDE2MyAxMi42NzkxQzcwLjQxNjMgMTQuNzg1MyA2OC44MjM1IDE2LjQyOTcgNjYuNzg2MSAxNi40Mjk3WiIgZmlsbD0iYmxhY2siPjwvcGF0aD48cGF0aCBkPSJNNzcuNDkyMiAxNy4wNzU1Qzc3LjUxMDcgMTcuMDc1NSA3Ny41NDc4IDE3LjA1NzEgNzcuNTY2MyAxNy4wNTcxQzc3LjYyMTggMTcuMDU3MSA3Ny42OTU5IDE3LjA5NCA3Ny43MzMgMTcuMTMxQzc4LjYwMzUgMTguMTQ3MSA3OS45NTU1IDE4LjcxOTkgODEuNDM3MiAxOC43MTk5Qzg0LjYwNDQgMTguNzE5OSA4Ny4wODYyIDE2LjA1OTQgODcuMDg2MiAxMi42NTk4Qzg3LjA4NjIgMTAuOTk3IDg2LjUxMjEgOS40NjM1NSA4NS40NTY0IDguMzM2NTNDODQuNDAwNiA3LjIwOTUxIDgyLjk3NDUgNi41OTk4MiA4MS40MzcyIDYuNTk5ODJDNzkuOTc0MSA2LjU5OTgyIDc4LjYyMiA3LjIwOTUxIDc3LjczMyA4LjI2MjYzQzc3LjY3NzQgOC4zMzY1MyA3Ny41ODQ4IDguMzU1MDEgNzcuNDkyMiA4LjMzNjUzQzc3LjM5OTYgOC4yOTk1OCA3Ny4zNDQgOC4yMjU2OCA3Ny4zNDQgOC4xMzMzVjEuODUxNTZINzQuNzg4MVYxOC40NDI3SDc3LjM0NFYxNy4yNzg4Qzc3LjM0NCAxNy4xODY0IDc3LjM5OTYgMTcuMTEyNSA3Ny40OTIyIDE3LjA3NTVaTTc3LjIzMjkgMTIuNjc4M0M3Ny4yMzI5IDEwLjU3MjEgNzguODI1NyA4LjkyNzc1IDgwLjg2MzEgOC45Mjc3NUM4Mi45MDA0IDguOTI3NzUgODQuNDkzMiAxMC41NzIxIDg0LjQ5MzIgMTIuNjc4M0M4NC40OTMyIDE0Ljc4NDUgODIuOTAwNCAxNi40Mjg5IDgwLjg2MzEgMTYuNDI4OUM3OC44MjU3IDE2LjQyODkgNzcuMjMyOSAxNC43NjYxIDc3LjIzMjkgMTIuNjc4M1oiIGZpbGw9ImJsYWNrIj48L3BhdGg+PHBhdGggZD0iTTk0LjQ3NjYgOS4yNjMyOEM5NC43MTczIDkuMjYzMjggOTQuOTM5NiA5LjI4MTc1IDk1LjA4NzggOS4zMTg3VjYuNjk1MTVDOTQuOTk1MSA2LjY3NjY4IDk0LjgyODUgNi42NTgyIDk0LjY2MTggNi42NTgyQzkzLjMyODIgNi42NTgyIDkyLjEwNTggNy4zNDE4IDkxLjQ1NzYgOC40MzE4N0M5MS40MDIgOC41MjQyNSA5MS4zMDk0IDguNTYxMiA5MS4yMTY4IDguNTI0MjVDOTEuMTI0MiA4LjUwNTc3IDkxLjA1MDEgOC40MTMzOSA5MS4wNTAxIDguMzIxMDJWNi44OTgzOUg4OC41MTI3VjE4LjQ2NDJIOTEuMDY4NlYxMy4zNjQ5QzkxLjA2ODYgMTAuODMzNyA5Mi4zNjUxIDkuMjYzMjggOTQuNDc2NiA5LjI2MzI4WiIgZmlsbD0iYmxhY2siPjwvcGF0aD48cGF0aCBkPSJNOTkuMjkxNyA2Ljg5NzQ2SDk2LjY5ODdWMTguNDYzMkg5OS4yOTE3VjYuODk3NDZaIiBmaWxsPSJibGFjayI+PC9wYXRoPjxwYXRoIGQ9Ik05Ny45NTc2IDEuODcwMTJDOTcuMDg3MSAxLjg3MDEyIDk2LjM4MzMgMi41NzIxOSA5Ni4zODMzIDMuNDQwNTVDOTYuMzgzMyA0LjMwODkxIDk3LjA4NzEgNS4wMTA5OSA5Ny45NTc2IDUuMDEwOTlDOTguODI4MSA1LjAxMDk5IDk5LjUzMTkgNC4zMDg5MSA5OS41MzE5IDMuNDQwNTVDOTkuNTMxOSAyLjU3MjE5IDk4LjgyODEgMS44NzAxMiA5Ny45NTc2IDEuODcwMTJaIiBmaWxsPSJibGFjayI+PC9wYXRoPjxwYXRoIGQ9Ik0xMDYuODg2IDYuNjAwNTlDMTAzLjMzIDYuNjAwNTkgMTAwLjc1NSA5LjE1MDIzIDEwMC43NTUgMTIuNjc5MUMxMDAuNzU1IDE0LjM5NzMgMTAxLjM2NyAxNS45MzA4IDEwMi40NTkgMTcuMDM5NEMxMDMuNTcxIDE4LjE0NzkgMTA1LjEyNiAxOC43NTc2IDEwNi44NjcgMTguNzU3NkMxMDguMzEyIDE4Ljc1NzYgMTA5LjQyMyAxOC40ODA1IDExMS41MzUgMTYuOTI4NUwxMTAuMDcyIDE1LjM5NUMxMDkuMDM0IDE2LjA3ODYgMTA4LjA3MSAxNi40MTEyIDEwNy4xMjcgMTYuNDExMkMxMDQuOTc4IDE2LjQxMTIgMTAzLjM2NyAxNC44MDM4IDEwMy4zNjcgMTIuNjc5MUMxMDMuMzY3IDEwLjU1NDQgMTA0Ljk3OCA4Ljk0NyAxMDcuMTI3IDguOTQ3QzEwOC4xNDUgOC45NDcgMTA5LjA5IDkuMjc5NTYgMTEwLjAzNSA5Ljk2MzE2TDExMS42NjQgOC40Mjk2OEMxMDkuNzU3IDYuODAzODIgMTA4LjAzNCA2LjYwMDU5IDEwNi44ODYgNi42MDA1OVoiIGZpbGw9ImJsYWNrIj48L3BhdGg+PHBhdGggZD0iTTExNi4wMzUgMTMuMzYyQzExNi4wNzIgMTMuMzI1IDExNi4xMjggMTMuMzA2NiAxMTYuMTg0IDEzLjMwNjZIMTE2LjIwMkMxMTYuMjU4IDEzLjMwNjYgMTE2LjMxMyAxMy4zNDM1IDExNi4zNjkgMTMuMzgwNUwxMjAuNDYyIDE4LjQ0MjhIMTIzLjYxMUwxMTguMzE0IDEyLjA1MDJDMTE4LjIzOSAxMS45NTc4IDExOC4yMzkgMTEuODI4NSAxMTguMzMyIDExLjc1NDZMMTIzLjIwMyA2Ljg5NTUxSDEyMC4wNzNMMTE1Ljg2OSAxMS4xMDhDMTE1LjgxMyAxMS4xNjM0IDExNS43MjEgMTEuMTgxOSAxMTUuNjI4IDExLjE2MzRDMTE1LjU1NCAxMS4xMjY0IDExNS40OTggMTEuMDUyNSAxMTUuNDk4IDEwLjk2MDJWMS44NzAxMkgxMTIuOTI0VjE4LjQ2MTNIMTE1LjQ4VjEzLjk1MzJDMTE1LjQ4IDEzLjg5NzggMTE1LjQ5OCAxMy44MjM5IDExNS41NTQgMTMuNzg2OUwxMTYuMDM1IDEzLjM2MloiIGZpbGw9ImJsYWNrIj48L3BhdGg+PHBhdGggZD0iTTEyNy43NzYgMTguNzM5QzEyOS44NjkgMTguNzM5IDEzMS45OTkgMTcuNDY0MiAxMzEuOTk5IDE1LjA0MzlDMTMxLjk5OSAxMy40NTUgMTMwLjk5OSAxMi4zNjQ5IDEyOC45NjIgMTEuNjk5OEwxMjcuNTcyIDExLjIzNzlDMTI2LjYyOCAxMC45MjM4IDEyNi4xODMgMTAuNDgwNCAxMjYuMTgzIDkuODcwN0MxMjYuMTgzIDkuMTY4NjMgMTI2LjgxMyA4LjY4ODI2IDEyNy43MDIgOC42ODgyNkMxMjguNTU0IDguNjg4MjYgMTI5LjMxMyA5LjI0MjUzIDEyOS43OTUgMTAuMjAzM0wxMzEuODUxIDkuMDk0NzNDMTMxLjA5MiA3LjU0Mjc3IDEyOS41MTcgNi41ODIwMyAxMjcuNzAyIDYuNTgyMDNDMTI1LjQwNSA2LjU4MjAzIDEyMy43MzkgOC4wNjAwOSAxMjMuNzM5IDEwLjA3MzlDMTIzLjczOSAxMS42ODEzIDEyNC43MDIgMTIuNzUyOSAxMjYuNjgzIDEzLjM4MTFMMTI4LjExIDEzLjg0M0MxMjkuMTEgMTQuMTU3MSAxMjkuNTM2IDE0LjU2MzUgMTI5LjUzNiAxNS4yMTAyQzEyOS41MzYgMTYuMTg5NCAxMjguNjI4IDE2LjU0MDQgMTI3Ljg1IDE2LjU0MDRDMTI2LjgxMyAxNi41NDA0IDEyNS44ODcgMTUuODc1MyAxMjUuNDQzIDE0Ljc4NTJMMTIzLjM1IDE1Ljg5MzhDMTI0LjAzNSAxNy42NDkgMTI1LjcyIDE4LjczOSAxMjcuNzc2IDE4LjczOVoiIGZpbGw9ImJsYWNrIj48L3BhdGg+PHBhdGggZD0iTTU4LjIzMDQgMTguNjI4QzU5LjA0NTMgMTguNjI4IDU5Ljc2NzcgMTguNTU0MSA2MC4xNzUxIDE4LjQ5ODZWMTYuMjgxNUM1OS44NDE4IDE2LjMxODUgNTkuMjQ5MSAxNi4zNTU0IDU4Ljg5NzIgMTYuMzU1NEM1Ny44NiAxNi4zNTU0IDU3LjA2MzYgMTYuMTcwNyA1Ny4wNjM2IDEzLjkzNTFWOS4xODY4N0M1Ny4wNjM2IDkuMDU3NTQgNTcuMTU2MiA4Ljk2NTE3IDU3LjI4NTggOC45NjUxN0g1OS43ODYyVjYuODc3NDFINTcuMjg1OEM1Ny4xNTYyIDYuODc3NDEgNTcuMDYzNiA2Ljc4NTAzIDU3LjA2MzYgNi42NTU3VjMuMzMwMDhINTQuNTA3NlY2LjY3NDE4QzU0LjUwNzYgNi44MDM1MSA1NC40MTUgNi44OTU4OSA1NC4yODU0IDYuODk1ODlINTIuNTA3M1Y4Ljk4MzY0SDU0LjI4NTRDNTQuNDE1IDguOTgzNjQgNTQuNTA3NiA5LjA3NjAyIDU0LjUwNzYgOS4yMDUzNVYxNC41ODE4QzU0LjUwNzYgMTguNjI4IDU3LjIxMTcgMTguNjI4IDU4LjIzMDQgMTguNjI4WiIgZmlsbD0iYmxhY2siPjwvcGF0aD48L3N2Zz4="

# COMMAND ----------

theme = "the delivery person is dressed like an astronout and is riding a motorbike in an urban street"

# COMMAND ----------

prompt = """
Given the theme, products and review feedback. Generate a marketing image for instagram that embededs these elements for a grocery delivery business. 

theme:
{theme}

products:
{product_details}

review summary:
{summary}

Cold color palette, muted colors, detailed, 8k

"""

# COMMAND ----------

print(prompt.format(theme=theme,product_details=product_summ,summary=review_summ))

# COMMAND ----------

use_shutterstock_imageai = True

import base64

if use_shutterstock_imageai:
    from openai import OpenAI
    import os

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
    DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

    client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints"
    )

    images = client.images.generate(
    prompt=prompt.format(theme=theme,product_details=product_summ,summary=review_summ),
    model="databricks-shutterstock-imageai"
    # model='databricks-llama-2-70b-chat'
    )

    print(images.data[0].b64_json)
    with open("myimage.jpg", "wb") as fh:
        fh.write(base64.b64decode(images.data[0].b64_json))
else:
    with open("myimage.jpg", "wb") as fh:
        fh.write(base64.b64decode(base64_image))

# COMMAND ----------

token_c= "<token>"

import base64
slack_token = base64.b64decode(token_c).decode("utf-8")

# COMMAND ----------

import logging
import os
# Import WebClient from Python SDK (github.com/slackapi/python-slack-sdk)
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# WebClient instantiates a client that can call API methods
# When using Bolt, you can use either `app.client` or the `client` passed to listeners.
client = WebClient(token=token)
logger = logging.getLogger(__name__)

# COMMAND ----------

file_name = "myimage.jpg"
# ID of channel that you want to upload file to
channel_id = "C07BEGG7YGH"

try:
    # Call the files.upload method using the WebClient
    # Uploading files requires the `files:write` scope

    result = client.files_upload_v2(
        channel=channel_id,
        file=file_name
    )
    print(result)
    
    # Log the result
    # ts = result['files'][0]['timestamp']

    # result = client.chat_update(
    #     channel=channel_id,
    #     ts=ts,
    #     text="Here's my file :smile:",
    # )

    logger.info(result)

except SlackApiError as e:
    logger.error("Error uploading file: {}".format(e))

# COMMAND ----------

# DBTITLE 1,Sample data generator
import random
from datetime import datetime, timedelta

# Define the items and member IDs
items = ["Apples", "Bananas", "Bread", "Milk", "Curd","Eggs", "Cheese", "Butter", "Sugarcane juice", "Mosambi juice", "Yogurt", "Tea powder","Wheat flour","Potatos","Carrots","Spices","Biscuits"]
gender = ['male','female','unknown']
age_group = ['young','adult','senior']
order_c = ['Bengaluru','Mumbai','Chennai','Hyderabad','Pune']
member_ids = [1, 2, 3, 4, 5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

# Generate the records
records = []
for _ in range(20000):
    member_id = random.choice(member_ids)
    date = datetime.now() - timedelta(days=random.randint(1, 60))
    item = random.choice(items)
    order_city = random.choice(order_c)
    records.append((member_id, date.strftime("%Y-%m-%d"), item, order_city))

# creating a dataframe 
dataframe = spark.createDataFrame(records, ["Member_number", "Date", "itemDescription","orderCity"]) 
# display(dataframe)
dataframe.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("users.abhilash_r.groceries_dataset")

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

os.environ['DATABRICKS_TOKEN']  = dbutils.secrets.get('abhir_agent_studio','databricks_token')

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/abhir_sample_agent_llama/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    # ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(dataset, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

# COMMAND ----------

data = {
  "dataframe_split": {
    "columns": [
      "messages"
    ],
    "data": [
      [
        [
          {
            "role": "user",
            "content": "what are the top items ordered in the past week"
          }
        ]

      ]
    ]
  }
}

response = score_model(data)

# COMMAND ----------

print(response)

# COMMAND ----------

import dbutils

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC users.abhilash_r.generate_image_and_share (
# MAGIC   prompt STRING COMMENT 'Prompt to the model'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'Generate an image based on the prompt'
# MAGIC AS $$
# MAGIC   import requests
# MAGIC   import json
# MAGIC   import base64
# MAGIC   import io
# MAGIC   from PIL import Image
# MAGIC   import datetime
# MAGIC
# MAGIC   CLIENT_ID = "5670bdd6-0210-4b3d-8eda-62558be712ce"
# MAGIC   CLIENT_SECRET = "dosef82b413c613730d113c61ead61573696"
# MAGIC
# MAGIC   url = "https://e2-demo-field-eng.cloud.databricks.com/oidc/v1/token"
# MAGIC
# MAGIC   payload = {
# MAGIC     "grant_type": "client_credentials",
# MAGIC     "scope": "all-apis",
# MAGIC     "expires_in":60
# MAGIC   }
# MAGIC
# MAGIC   response = json.loads(requests.post(url, data=payload, auth=(CLIENT_ID, CLIENT_SECRET)).text)
# MAGIC
# MAGIC   token = response["access_token"]
# MAGIC
# MAGIC   url = "https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/databricks-shutterstock-imageai/invocations"
# MAGIC
# MAGIC   j = json.loads(requests.post(url, json={"prompt": f"{prompt}.Resolution 256K"}, headers={"Authorization": f"Bearer {token}"}).content)
# MAGIC
# MAGIC   base64_str = j["data"][0]["b64_json"]
# MAGIC
# MAGIC   buffer = io.BytesIO()
# MAGIC   imgdata = base64.b64decode(base64_str)
# MAGIC   img = Image.open(io.BytesIO(imgdata))
# MAGIC   new_img = img.resize((512, 512))  # x, y
# MAGIC   new_img.save(buffer, format="PNG")
# MAGIC   
# MAGIC   now = datetime.datetime.now()
# MAGIC   time = now.strftime("%Y%m%d%H%M%S")
# MAGIC   
# MAGIC   url = f"https://e2-demo-field-eng.cloud.databricks.com/api/2.0/fs/files/Volumes/users/abhilash_r/test_files/shutterstock_image_{time}.png"
# MAGIC
# MAGIC   s = requests.put(url, data=buffer.getvalue(), headers={"Authorization": f"Bearer {token}"})
# MAGIC   
# MAGIC   return f"Image uploaded to /Volumes/users/abhilash_r/test_files/shutterstock_image_{time}.png"
# MAGIC $$

# COMMAND ----------

# MAGIC %sql
# MAGIC select users.abhilash_r.generate_image_and_share('astronout on the moon')

# COMMAND ----------

import requests

CLIENT_ID = "5670bdd6-0210-4b3d-8eda-62558be712ce"
CLIENT_SECRET = "<secret>"

url = "https://e2-demo-field-eng.cloud.databricks.com/oidc/v1/token"

# Assuming the API expects the client credentials in the body
payload = {
    "grant_type": "client_credentials",
    "scope": "all-apis"
}

# If the API requires Basic Auth, use the auth parameter
response = json.loads(requests.post(url, data=payload, auth=(CLIENT_ID, CLIENT_SECRET)).text)

# print(response["access_token"])
token = response["access_token"]

# COMMAND ----------


url = "https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/databricks-shutterstock-imageai/invocations"

j = json.loads(requests.post(url, json={

      
        "prompt": "hello. Resolution 256K"
      
    
  }, headers={"Authorization": f"Bearer {token}"}).content)

print (j)

# COMMAND ----------

import io
import base64
from PIL import Image

base64_str = j["data"][0]["b64_json"]

buffer = io.BytesIO()
imgdata = base64.b64decode(base64_str)
img = Image.open(io.BytesIO(imgdata))
new_img = img.resize((128, 128))  # x, y
new_img.save(buffer, format="PNG")
img_b64 = base64.b64encode(buffer.getvalue())
print(str(img_b64)[2:-1])

# COMMAND ----------

print(buffer.getvalue())

# COMMAND ----------

url = "https://e2-demo-field-eng.cloud.databricks.com/api/2.0/fs/files/Volumes/users/abhilash_r/test_files/sample.png"

s = requests.put(url, data=buffer.getvalue(), headers={"Authorization": f"Bearer {token}"})

print (s)


# COMMAND ----------

url = "https://slack.com/api/chat.postMessage"

s = json.loads(requests.post(url, data={"channel": "C07BEGG7YGH","text":"hello"}, headers={"Authorization": f"Bearer {slack_token}"}).content)

print (s)

