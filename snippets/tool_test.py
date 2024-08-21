# Databricks notebook source
# MAGIC %pip install --upgrade --quiet databricks-sdk langchain-community mlflow

# COMMAND ----------

from langchain_community.chat_models.databricks import ChatDatabricks

#llm = ChatDatabricks(endpoint="databricks-dbrx-instruct")
llm = ChatDatabricks(endpoint="DBRX-Provisioned-Throughput")

# COMMAND ----------

# MAGIC %sql
# MAGIC select benmackenzie_catalog.default.python_exec('print(232323 * 23232.0)');

# COMMAND ----------

from langchain_community.tools.databricks import UCFunctionToolkit

tools = (
    UCFunctionToolkit(
        # You can find the SQL warehouse ID in its UI after creation.
        warehouse_id="e2-demo-field-eng.cloud.databricks.com"
    )
    .include(
        # Include functions as tools using their qualified names.
        # You can use "{catalog_name}.{schema_name}.*" to get all functions in a schema.
        "benmackenzie_catalog.default.python_exec",
    )
    .get_tools()
)

# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Make sure to use tool for information.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)

# COMMAND ----------

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "36939 * 8922.4"})
