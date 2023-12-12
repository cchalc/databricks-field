# Databricks notebook source
query = dbutils.widgets.get("Prompt")
q_and_a(query)

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC #### What is Retrieval Augmented Generation (RAG) for LLMs?
# MAGIC
# MAGIC RAG is a powerful and efficient GenAI technique that allows you to improve model performance by leveraging your own data (e.g., documentation specific to your business), without the need to fine-tune the model.
# MAGIC This is done by providing your custom information as context to the LLM. This reduces hallucination and allows the LLM to produce results that provide company-specific data, without making any changes to the original LLM.
# MAGIC RAG has shown success in chatbots and Q&A systems that need to maintain up-to-date information or access domain-specific knowledge.
# MAGIC
# MAGIC ##### RAG and Vector Search
# MAGIC
# MAGIC To be able to provide additional context to our LLM, we need to search for documents/articles where the answer to our user question might be.
# MAGIC To do so,  a common solution is to deploy a vector database. This involves the creation of document embeddings (vectors of fixed size, computed by a model).<br/>
# MAGIC The vectors will then be used to perform realtime similarity search during inference.
# MAGIC
# MAGIC ##### Implementing RAG with Databricks Lakehouse AI and a MosaicML endpoint
# MAGIC
# MAGIC In this demo, we will show you how to build and deploy your custom chatbot, answering questions on any custom or private information.
# MAGIC As an example, we will specialize this chatbot to answer questions over Databricks, feeding databricks.com documentation articles to the model for accurate answers.
# MAGIC Here is the flow we will implement:
# MAGIC
# MAGIC - Download databricks.com documentation articles
# MAGIC - Prepare the articles for our model (split into chunks)
# MAGIC - Create a Vector Search Index using an Embedding endpoint
# MAGIC - Deploy an AI gateway as a proxy to MosaicML
# MAGIC - Build and deploy our RAG chatbot
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-full.png?raw=true" style="float: right; margin-left: 10px"  width="1100px;">

# COMMAND ----------



def display_answer(question, answer):
  prompt = answer
  answer = answer['choices'][0]['message']['content']
  
  #Tune the message with the user running the notebook. In real workd example we'd have a table with the customer details. 
  displayHTML(f"""
              <div style="float: right; width: 45%;">
                <h3>Debugging:</h3>
                <div style="border-radius: 10px; background-color: #ebebeb; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; color: #363636"><strong>Prompt sent to the model:</strong><br/><i>{prompt}</i></div>
              </div>
              <h3>Chatbot:</h3>
              <div style="border-radius: 10px; background-color: #c2efff; padding: 10px; width: 45%; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; font-size: 14px">{question}</div>
                <div style="border-radius: 10px; background-color: #e3f6fc; padding: 10px;  width: 45%; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; margin-left: 40px; font-size: 14px">
                <img style="float: left; width:40px; margin: -10px 5px 0px -10px" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/robot.png?raw=true"/> {answer}
                </div>
        """)
  
from openai.embeddings_utils import get_embedding, cosine_similarity 

import openai

openaikey = dbutils.secrets.get("tokens", "canadaeh-openaikey")
openai.api_key = openaikey
openai.api_type = "azure"
openai.api_base = "https://canada-eh-openai.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
aoai_model = "gpt-35-deployment"

def display_answer(question, answer):
  prompt = answer
  answer = answer['choices'][0]['message']['content']
  
  #Tune the message with the user running the notebook. In real workd example we'd have a table with the customer details. 
  displayHTML(f"""
              <div style="float: right; width: 45%;">
                <h3>Debugging:</h3>
                <div style="border-radius: 10px; background-color: #ebebeb; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; color: #363636"><strong>Prompt sent to the model:</strong><br/><i>{prompt}</i></div>
              </div>
              <h3>Chatbot:</h3>
              <div style="border-radius: 10px; background-color: #c2efff; padding: 10px; width: 45%; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; font-size: 14px">{question}</div>
                <div style="border-radius: 10px; background-color: #e3f6fc; padding: 10px;  width: 45%; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; margin-left: 40px; font-size: 14px">
                <img style="float: left; width:40px; margin: -10px 5px 0px -10px" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/robot.png?raw=true"/> {answer}
                </div>
        """)

pdf_subset2 = spark.read.table("cjc_cap_markets.capm_data.ada_embeddings").toPandas()

def search_docs(df, user_query, top_n=3):
  embedding = get_embedding(
      user_query,
      engine="cjc-text-embedding-ada-002"
  )
  df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

  res = (
      df.sort_values("similarities", ascending=False)
      .head(top_n)
  )
  return res

def combine_nested_list(nested_list):
  """helper function to combine list"""
  combined_text = ' '.join([''.join(sublist) for sublist in nested_list])
  return combined_text


def split_string_with_overlap(input_string, chunk_size=500, overlap=20):
  """helper function to (re)split into specific chunk sizes. Overlap refers to number of shared words across sliding windows to prevent context loss when chunking. chunk size should be mostly based on model context window size, and to a lesser extent, desired number of chunks returned per query and considerations around context window loss"""
  words = input_string.split()
  result = []
  start = 0

  while start < len(words):
    end = min(start + chunk_size, len(words))
    chunk = words[start:end]

    if start > 0:
        chunk = words[start-overlap:end]

    result.append(" ".join(chunk))
    start += chunk_size - overlap

  return result

def q_and_a(query):
  res = search_docs(pdf_subset2, query, top_n=8)['text'].tolist()
  context = combine_nested_list(res)

  prompt = [{"role": "system", "content": "You are a helpful assistant."},
            {'role': 'user','content':'test'}]
  prompt[1] = {'role': 'user','content': f'Please answer the question: {query}. If the answer is not in the below context, feel free to attempt to answer based on your internal knowledge, but be clear that you are doing so. \n\n provided context: {context}' }

  result = openai.ChatCompletion.create(
      engine=aoai_model,
      messages=prompt,
      max_tokens=300,
      temperature=0.0
  )
  
  display_answer(query, result)

dbutils.widgets.text("Prompt", "")

# COMMAND ----------

# who is the primary company within this 10-K Form?
# where is bank of america headquartered?
# what were bank of america's total assets and liabilities
# what were bank of america's total revenues
# what are bank of america's positions on diversity and inclusion?
# how did bank of america's net earnings improve year over year
# what are bank of america's top assets
# what were the q4 earnings of bank of america?

# COMMAND ----------

# dbutils.widgets.removeAll()
