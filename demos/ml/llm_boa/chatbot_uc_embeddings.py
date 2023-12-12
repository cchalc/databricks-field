# Databricks notebook source
# MAGIC %run ./resources/prettify

# COMMAND ----------

from openai.embeddings_utils import get_embedding, cosine_similarity 

import openai

openaikey = dbutils.secrets.get("tokens", "canadaeh-openaikey")
openai.api_key = openaikey
openai.api_type = "azure"
openai.api_base = "https://canada-eh-openai.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
aoai_model = "gpt-35-deployment"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Search for relevant documents
# MAGIC
# MAGIC We define a search function below to first vectorize our query text, and then search for the vectors with the closest distance. 

# COMMAND ----------

pdf_subset2 = spark.read.table("cjc_cap_markets.capm_data.ada_embeddings").toPandas()

# COMMAND ----------

# search through the reviews for a specific product
def search_docs(df, user_query, top_n=3):
  embedding = get_embedding(
      user_query,
      engine="cjc-text-embedding-ada-002" # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
  )
  df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

  res = (
      df.sort_values("similarities", ascending=False)
      .head(top_n)
  )
  return res

res = search_docs(pdf_subset2, "Can I get information on cable company tax revenue?", top_n=4)
display(res['text'])

# COMMAND ----------

# MAGIC %md
# MAGIC Tada! Now you can query for similar content! Notice that you did not have to configure any database networks beforehand nor pass in any credentials. FAISS works locally with your code.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt engineering for question answering 
# MAGIC
# MAGIC Now that we have identified documents about space from the news dataset, we can pass these documents as additional context for a language model to generate a response based on them! 
# MAGIC
# MAGIC We first need to pick a `text-generation` model. Below, we use a Hugging Face model. You can also use OpenAI as well, but you will need to get an Open AI token and [pay based on the number of tokens](https://openai.com/pricing). 

# COMMAND ----------

# MAGIC %md
# MAGIC ## OpenAI

# COMMAND ----------

#Set Model
aoai_model = "gpt-35-deployment"

# COMMAND ----------

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

# COMMAND ----------

# DBTITLE 0,User QnA
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

query = input("Hey! I'm your Databricks assistant. How can I help?")
q_and_a(query)

# COMMAND ----------


