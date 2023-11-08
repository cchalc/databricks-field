# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # 2/ Creating a Vector Search Index on top of our Delta Lake table
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-3.1.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC We now have our knowledge base ready, and saved as a Delta Lake table within Unity Catalog (including permission, lineage, audit logs and all UC features).
# MAGIC
# MAGIC Typically, deploying a production-grade Vector Search index on top of your knowledge base is a difficult task. You need to maintain a process to capture table changes, index the model, provide a security layer, and all sorts of advanced search capabilities.
# MAGIC
# MAGIC Databricks Vector Search removes those painpoints.
# MAGIC
# MAGIC ## Databricks Vector Search
# MAGIC Databricks Vector Search is a new production-grade service that allows you to store a vector representation of your data, including metadata. It will automatically sync with the source Delta table and keep your index up-to-date without you needing to worry about underlying pipelines or clusters. 
# MAGIC
# MAGIC It makes embeddings highly accessible. You can query the index with a simple API to return the most similar vectors, and can optionally include filters or keyword-based queries.
# MAGIC
# MAGIC Vector Search is currently in Private Preview; you can [*Request Access Here*](https://docs.google.com/forms/d/e/1FAIpQLSeeIPs41t1Ripkv2YnQkLgDCIzc_P6htZuUWviaUirY5P5vlw/viewform)
# MAGIC
# MAGIC *If you still do not have access to Databricks Vector Search, you can leverage [Chroma](https://docs.trychroma.com/getting-started) (open-source embedding database for building LLM apps). For an example end-to-end implementation with Chroma, please see [this demo](https://www.dbdemos.ai/minisite/llm-dolly-chatbot/).*

# COMMAND ----------

print('')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Document Embeddings 
# MAGIC
# MAGIC The first step is to create embeddings from the documents saved in our Delta Lake table. To do so, we need an LLM model specialized in taking a text of arbitrary length, and turning it into an embedding (vector of fixed size representing our document). 
# MAGIC
# MAGIC Embedding creation is done through LLMs, and many options are available: from public APIs to private models fine-tuned on your datasets.
# MAGIC
# MAGIC *Note: It is critical to ensure that the model is always the same for both embedding index creation and real-time similarity search. Remember that if your embedding model changes, you'll have to re-index your entire set of vectors, otherwise similarity search won't return relevant results.*

# COMMAND ----------

# DBTITLE 1,Install vector search package
# MAGIC %pip install databricks-vectorsearch mlflow==2.8.0 databricks-sdk==0.12.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=cjc $db=chatbot $reset_all_data=false

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating and registring our embedding model in UC
# MAGIC
# MAGIC Let's create an embedding model and save it in Unity Catalog. We'll then deploy it as serverless model serving endpoint. Vector Search will call this endpoint to create embeddings from our documents, and then index them.
# MAGIC
# MAGIC The model will also be used during realtime similarity search to convert the queries into vectors. This will be taken care of by Databricks Vector Search.
# MAGIC
# MAGIC #### Choosing an embeddings model
# MAGIC There are multiple choices for the embeddings model:
# MAGIC
# MAGIC * **SaaS API embeddings model**:
# MAGIC Starting simple with a SaaS API is a good option. If you want to avoid vendor dependency as a result of proprietary SaaS API solutions (e.g. OpenAI), you can build with a SaaS API that is pointing to an OSS model. You can use the new [MosaicML Embedding](https://docs.mosaicml.com/en/latest/inference.html) endpoint: `/instructor-large/v1`. See more in [this blogpost](https://www.databricks.com/blog/using-ai-gateway-llama2-rag-apps)
# MAGIC * **Deploy an OSS embeddings model**: On Databricks, you can deploy a custom copy of any OSS embeddings model behind a production-grade Model Serving endpoint.
# MAGIC * **Fine-tune an embeddings model**: On Databricks, you can use AutoML to fine-tune an embeddings model to your data. This has shown to improve relevance of retrieval. AutoML is in Private Preview - [Request Access Here](https://docs.google.com/forms/d/1MZuSBMIEVd88EkFj1ehN3c6Zr1OZfjOSvwGu0FpwWgo/edit)
# MAGIC
# MAGIC Because we want to keep this demo simple, we'll directly leverage MosaicML endpoint, an external SaaS API.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Deploying an AI gateway to MosaicML Endpoint
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-3.png?raw=true" style="float: right; margin-left: 10px"  width="600px;">
# MAGIC
# MAGIC
# MAGIC With MLflow, Databricks introduced the concept of AI Gateway ([documentation](https://mlflow.org/docs/latest/gateway/index.html)).
# MAGIC
# MAGIC AI Gateway acts as a proxy between your application and LLM APIs. It offers:
# MAGIC
# MAGIC - API key management
# MAGIC - Unified access point to easily switch the LLM backend without having to change your implementation
# MAGIC - Throughput control
# MAGIC - Logging and retries
# MAGIC - Format prompt for your underlying model
# MAGIC
# MAGIC *Note: if you don't have a MosaicML key, you can also deploy an OpenAI gateway route:*
# MAGIC
# MAGIC ```
# MAGIC gateway.create_route(
# MAGIC     name=mosaic_embeddings_route_name,
# MAGIC     route_type="llm/v1/embeddings",
# MAGIC     model={
# MAGIC         "name": "text-embedding-ada-002",
# MAGIC         "provider": "openai",
# MAGIC         "openai_config": {
# MAGIC             "openai_api_key": dbutils.secrets.get("dbdemos", "openai"),
# MAGIC         }
# MAGIC     }
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC ### This demo requires secrets:
# MAGIC
# MAGIC * You'll need to setup the Databricks CLI in your laptop or using this cluster terminal:
# MAGIC `%pip install databricks-cli`<br/>
# MAGIC * Configure the CLI. You'll need your workspace URL and a PAT token from [your profile page](/settings/user/developer/access-tokens/)<br/>
# MAGIC `databricks configure`<br/>
# MAGIC * Create the dbdemos scope:<br/>
# MAGIC `databricks secrets create-scope --scope dbdemos`<br/>
# MAGIC * Define your Mosaic ML secret for your MosaicML key. *If you still don't have access to mosaicML, you can consider using Azure OpenAI (in this case, setup your AI gateway with the example we gave)*<br/>
# MAGIC `databricks secrets put --scope dbdemos --key mosaic_ml_api_key`<br/>
# MAGIC * Define your service principal secret. It will be used by the Model Endpoint to autenticate your AI gateway. If this is a demo/test, you can use one of your PAT token.<br/>
# MAGIC `databricks secrets put --scope dbdemos --key sp_secret_key`

# COMMAND ----------

# DBTITLE 1,Creating the AI Gateway with MosaicML embedding 
#init MLflow experiment
import mlflow
from mlflow import gateway
init_experiment_for_batch("cjc-llm-chatbot-rag", "cjc-rag-model")

gateway.set_gateway_uri(gateway_uri="databricks")
#define our embedding route name, this is the endpoint we'll call for our embeddings
mosaic_embeddings_route_name = "mosaicml-instructor-xl-embeddings"

try:
    route = gateway.get_route(mosaic_embeddings_route_name)
except:
    # Create a route for embeddings with MosaicML
    print(f"Creating the route {mosaic_embeddings_route_name}")
    print(gateway.create_route(
        name=mosaic_embeddings_route_name,
        route_type="llm/v1/embeddings",
        model={
            "name": "instructor-xl",
            "provider": "mosaicml",
            "mosaicml_config": {
                "mosaicml_api_key": dbutils.secrets.get(scope="cjc", key="mosaic_ml_api_key")#Don't have a MosaicML Key ? Try with AzureOpenAI instead!
            }
        }
    ))

# COMMAND ----------

# DBTITLE 1,Testing our AI Gateway
print(f"calling AI gateway {gateway.get_route(mosaic_embeddings_route_name).route_url}")

r = gateway.query(route=mosaic_embeddings_route_name, data={"text": "What is Databricks Lakehouse?"})

print(r)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Creating an embedding model serving endpoint
# MAGIC
# MAGIC <div style="background-color: #def2ff; padding: 15px;  border-radius: 30px; ">
# MAGIC   <strong>Information</strong><br/>
# MAGIC   Your Vector Search Index can directly be using the AI gateway to compute the embeddings. However, the AI Gateway API will shortly be improved for better usability. Therefore, we'll temporary introduce a serverless model endpoint doing the bridge between the Vector Search Index and your AI Gateway.
# MAGIC
# MAGIC   The flow will be the following:<br/>
# MAGIC   `Vector Search index => Model Serving Embedding endpoint => AI Gateway => MosaicML instructorXL endpoint`
# MAGIC
# MAGIC   The demo will be updated in a few weeks to remove the need for the embedding endpoint:<br/>
# MAGIC   `Vector Search index => AI Gateway => MosaicML instructorXL endpoint`
# MAGIC
# MAGIC   The Model Serving Embedding endpoint is just a proxy formatting the result for the vector search. Open [_resources/00-init]($./_resources/00-init)
# MAGIC
# MAGIC   *Remember that this is a temporary setup, we'll update the content to remove this part and the model serving creation in the future, so that you only need an AI gateway for your Vector Search index.*
# MAGIC </div>
# MAGIC

# COMMAND ----------

proxy_model_name="dbdemos_mosaic_ml_proxy"
print(f"{catalog}.{db}.{proxy_model_name}")

# COMMAND ----------

# Name of the proxy model
proxy_model_name="dbdemos_mosaic_ml_proxy"
# Where the model will be saved in Unity Catalog
proxy_model_full_name = f"{catalog}.{db}.{proxy_model_name}"
# The model serving embedding endpoint name
proxy_endpoint_name="cjc_dbdemos_embeddings_proxy"

# Create the model, register it in UC and deploy the serverless endpoint.
# The Service Principal (sp) secret is a PAT token used by the model serving endpoint to authenticate when quering the AI gateway.
launch_embedding_model_proxy_endpoint(proxy_model_full_name, proxy_endpoint_name, mosaic_embeddings_route_name, sp_secret_scope = "dbdemos", sp_secret_key = "ai_gateway_service_principal")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Creating the Vector Search Index using our AI Gateway
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-4.png?raw=true" style="float: right; margin-left: 10px" width="600px">
# MAGIC
# MAGIC
# MAGIC Now that our embedding endpoint is up and running, we can use it in our Vector Search index definition.
# MAGIC
# MAGIC Every time a new row is added in our Delta Lake table, Databricks will automatically capture the change, call the embedding endpoint with the row content, and index the embedding.
# MAGIC
# MAGIC *Note: Databricks Vector Search can also use a custom Model Serving endpoint if you wish to host your own fine-tuned embedding model instead of querying the AI Gateway. This lets you deploy fine-tuned embedding models.*

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Creating our Vector Search Endpoint
# MAGIC Vector Search endpoint provide a REST api that you can use to send low latencies, realtime search queries against embedding indexes.
# MAGIC
# MAGIC They're available in the [Compute](#setting/clusters/vector-search) menu. 
# MAGIC
# MAGIC Let's create our first Vector Search endpoint using the API.
# MAGIC

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
#Create our Vector Search client
vsc = VectorSearchClient()

vs_endpoint_name="dbdemos_vs_endpoint"
if not vs_endpoint_exists(vsc, vs_endpoint_name):
    vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating our Index
# MAGIC
# MAGIC Now that our endpoint is ready, we can create an index. The index will contain our embeddings. It requires:
# MAGIC
# MAGIC * A vector search endpoint, which will be used as REST endpoint to query our index
# MAGIC * A Delta Lake table, where our documents are stored
# MAGIC * An embedding model or an AI gateway to be used to compute the embeddings during indexation and request time
# MAGIC
# MAGIC As reminder, we want to add the index in the `databricks_documentation` table, indexing the column `content`. Let's review our `databricks_documentation` table:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM databricks_pdf_documentation

# COMMAND ----------

# MAGIC %md
# MAGIC Vector search will capture all changes in your table, including updates and deletions, to synchronize your embedding index.
# MAGIC
# MAGIC To do so, make sure the `delta.enableChangeDataFeed` option is enabled in your Delta Lake table. Databricks Vector Index will use it to automatically propagate the changes. See the [Change Data Feed docs](https://docs.databricks.com/en/delta/delta-change-data-feed.html#enable-change-data-feed) to learn more, including how to set this property at the time of table creation

# COMMAND ----------

# MAGIC %sql 
# MAGIC ALTER TABLE databricks_pdf_documentation SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# DBTITLE 1,Creating the index
#The table we'd like to index
source_table_fullname = f"{catalog}.{db}.databricks_pdf_documentation"
#Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.databricks_pdf_documentation_vs_index"

#if not index_exists(vs_index_fullname, vs_endpoint_name):
print(f'Creating a vector search index `{vs_index_fullname}` against the table `{source_table_fullname}`, using the endpoint {vs_endpoint_name}')
print(f'The index will use the serverless model serving embedding_model_endpoint_name={proxy_endpoint_name}')
#vsc.delete_index(vs_index_fullname, vs_endpoint_name)

if not index_exists(vsc, vs_index_fullname, vs_endpoint_name):
    print(f"creating endpoint {vs_endpoint_name}...")
    i=vsc.create_delta_sync_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_fullname,
        source_table_name=source_table_fullname,
        pipeline_type="CONTINUOUS",
        primary_key="id",
        embedding_model_endpoint_name=proxy_endpoint_name,
        embedding_source_column="content"
    )

    #sleep(3) #Set permission so that all users can access the demo index (shared)
    set_index_permission(vs_index_fullname, "ALL_PRIVILEGES", "account users")
    print(i)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Waiting for the index to build
# MAGIC That's all we have to do. Under the hood, Databricks will maintain a [Delta Live Tables](https://docs.databricks.com/en/delta-live-tables/index.html) (DLT) job to refresh our pipeline.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, this can take several minutes.
# MAGIC
# MAGIC For more details, you can access the DLT pipeline from the link you get in the index definition.

# COMMAND ----------

# %sql 
# SELECT * FROM event_log("0c2424b3-3f96-45fd-8660-8c1818a5e7b1")

# Error in SQL statement: AnalysisException: [EVENT_LOG_REQUIRES_SHARED_COMPUTE] Cannot query event logs from an Assigned or No Isolation Shared cluster, please use a Shared cluster or a Databricks SQL warehouse instead.

# COMMAND ----------

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, vs_index_fullname, vs_endpoint_name)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Your index is now ready and will be automatically synchronized with your table.
# MAGIC
# MAGIC Databricks will capture all changes made to the `databricks_documentation` Delta Lake table, and update the index accordingly. You can run your ingestion pipeline and update your documentations table, the index will automatically reflect these changes and get in synch with the best latencies.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Searching for similar content
# MAGIC
# MAGIC Our index is ready, and our Delta Lake table is now fully synchronized!
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC *Note: `similarity_search` also support a filters parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call.*
# MAGIC
# MAGIC *Note: Make sure that what you search is similar to one of the documents you indexed! Check your document table if in doubt.*

# COMMAND ----------

question = "How can I track billing usage on my workspaces?"

results = vsc.get_index(vs_index_fullname, vs_endpoint_name).similarity_search(
  query_text=question,
  columns=["url", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating a Direct Index for realtime / API updates
# MAGIC
# MAGIC The index we created is synchronized with an existing Delta Lake table. To update the index, you'll have to update your Delta table. This works well for ~sec latencies workloads and Analytics needs.
# MAGIC
# MAGIC However, some use-cases requires sub-second update latencies. Databricks let you create real-time indexes using Direct Index. This let you run instant insert/update on your rows.
# MAGIC
# MAGIC ```
# MAGIC vsc.create_direct_vector_index(
# MAGIC   index_name=index_name,
# MAGIC   endpoint_name= endpoint_name,
# MAGIC   primary_key="id",
# MAGIC   embedding_dimension=1024,
# MAGIC   embedding_column="text_vector",
# MAGIC   schema={"id": "integer", "text": "string", "text_vector": "array<float>", "bool_val": "boolean"}
# MAGIC ```
# MAGIC   
# MAGIC You can then do instant updates: `vsc.get_index(index_name, endpoint_name).upsert(new_rows)`

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Next step: Deploy our chatbot model with RAG
# MAGIC
# MAGIC We've seen how Databricks Lakehouse AI makes it easy to ingest and prepare your documents, and deploy a Vector Search index on top of it with just a few lines of code and configuration.
# MAGIC
# MAGIC This simplifies and accelerates your data projects so that you can focus on the next step: creating your realtime chatbot endpoint with well-crafted prompt augmentation.
# MAGIC
# MAGIC Open the [03-Deploy-RAG-Chatbot-Model]($./03-Deploy-RAG-Chatbot-Model) notebook to create and deploy a chatbot endpoint.
