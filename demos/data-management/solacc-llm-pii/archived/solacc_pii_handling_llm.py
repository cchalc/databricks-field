# Databricks notebook source
import accelerate
import pandas as pd
import mlflow
import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download

from config import HF_MODEL_NAME, HF_MODEL_REVISION_ID, CACHE_PATH, DOWNLOAD_PATH

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the PyFunc Model
# MAGIC
# MAGIC You'll notice that we have `trust_remote_code=True`. This is because:
# MAGIC > Note: This model requires that trust_remote_code=True be passed to the from_pretrained method. This is because we use a custom MPT model architecture that is not yet part of the Hugging Face transformers package
# MAGIC
# MAGIC ### Prompt template
# MAGIC
# MAGIC The MPT 7B Instruct model has been fine tuned with a specific [prompt template](https://huggingface.co/mosaicml/mpt-7b-instruct#formatting). You can think about the prompt template as the "right way to ask a question to the model". On the model's webpage, they specifically note that the model should be instructed in this way. 
# MAGIC
# MAGIC A prompt may look like
# MAGIC ```
# MAGIC Below is an instruction that describes a task.
# MAGIC Write a response that appropriately completes the request.
# MAGIC ### Instruction:
# MAGIC <your instruction goes here>
# MAGIC ### Response:
# MAGIC ```
# MAGIC
# MAGIC It can be cumbersome to remember this format and difficult to educate our business' users on it. They should only worry about the questions they want to ask. 
# MAGIC
# MAGIC Luckily, with a PyFunc model, we can abstract away this complexity by taking user prompts and massaging it into the appropriate format. This is exactly what is being done in the `_build_prompt()` method
# MAGIC

# COMMAND ----------

class MPT(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
          context.artifacts['repository'], padding_side="left")

        config = transformers.AutoConfig.from_pretrained(
            context.artifacts['repository'], 
            trust_remote_code=True,
            init_device = 'cuda', # For fast initialization directly on GPU
        )
        # [QFB]: why does switching on Triton cause CUDA OOM errors?
        # config.attn_config['attn_impl'] = 'triton'
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            config=config,
            device_map="auto",
            torch_dtype=torch.bfloat16, # Load model weights in bfloat16 # TODO: G5s, or float 16
            trust_remote_code=True)
        # [QFB]: does this interfere with init_device='cuda'? Or is it redundant?
        # self.model.to(device='cuda')
        
        self.model.eval()


    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        INTRO_BLURB = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
        )

        return f"""{INTRO_BLURB}
        {INSTRUCTION_KEY}
        {instruction}
        {RESPONSE_KEY}
        """


    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        generated_text = []
        for index, row in model_input.iterrows():
          prompt = row["prompt"]
          # You can add other parameters here
          temperature = model_input.get("temperature", [0.1])[0]
          max_new_tokens = model_input.get("max_new_tokens", [1000])[0]
          full_prompt = self._build_prompt(prompt)
          encoded_input = self.tokenizer.encode(full_prompt, return_tensors="pt").to("cuda")
          output = self.model.generate(
            encoded_input, 
            do_sample=True, 
            temperature=temperature,
            max_new_tokens=max_new_tokens)
          prompt_length = len(encoded_input[0])
          generated_text.append(
            self.tokenizer.batch_decode(
              output[:,prompt_length:], 
              skip_special_tokens=True
            )
          )
        return pd.Series(generated_text)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to MLflow

# COMMAND ----------

# DBTITLE 1,Retrieve the base model from Hugging Face
# If the model has been downloaded previously in the same session, this will not repetitively download large model files, 
#   but only the remaining files in the repo

model_location = snapshot_download(
  repo_id=HF_MODEL_NAME,
  cache_dir=CACHE_PATH,
  local_dir=DOWNLOAD_PATH,
  local_dir_use_symlinks=False,
  revision=HF_MODEL_REVISION_ID,
)

print(model_location)

# COMMAND ----------

# DBTITLE 1,Configure and log MLflow Model
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["what is ML?"], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 5 minutes to complete
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=MPT(),
        artifacts={'repository' : model_location},
        pip_requirements=[
          "accelerate", 
          "einops", 
          "sentencepiece",
          "torch", 
          "transformers",
          ],
        # pip_requirements=[
        #   # f"torch=={torch.__version__}", 
        #   f"transformers=={transformers.__version__}", 
        #   f"accelerate=={accelerate.__version__}", 
        #   "einops", 
        #   "sentencepiece"],
        input_example=input_example,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the Model in MLflow Model Registry
# MAGIC
# MAGIC This may take about 6 minutes to complete.
# MAGIC
# MAGIC A new model registry entry will be created if one doesn't already exist for the given model name. If an entry already exists, a new version of the model is registered

# COMMAND ----------

result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    name=MLFLOW_MODEL_NAME,
    await_registration_for=1000,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the model from Model Registry
# MAGIC
# MAGIC Now that we've packaged and registered our model, let's load it from model registry, and test some prompts
# MAGIC
# MAGIC Assume that the below code is run separately or after the memory cache is cleared.
# MAGIC You may need to cleanup the GPU memory.

# COMMAND ----------

import mlflow
import pandas as pd

from config import MLFLOW_MODEL_NAME

loaded_model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}/latest")

# Make a prediction using the loaded model
input_example=pd.DataFrame({"prompt":["what is ML?", "Name 10 colors."], "temperature": [0.5, 0.2],"max_tokens": [100, 200]})
print(loaded_model.predict(input_example))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint
# MAGIC
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the MPT-7B-Instruct model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. For more information on GPU model serving, contact the Databricks team or sign up [here](https://docs.google.com/forms/d/1-GWIlfjlIaclqDz6BPODI2j1Xg4f4WbFvBXyebBpN-Y/edit)
# MAGIC
# MAGIC At this point, model serving management is automated only via Databricks REST APIs

# COMMAND ----------

import requests
import json

# Provide a name to the serving endpoint
ENDPOINT_NAME = 'vinnyv-mpt-7b-instruct'

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'
update_url = f"{databricks_url}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/config"

model_version = result  # the returned result of mlflow.register_model
endpoint_config = {
  "name": ENDPOINT_NAME,
  "config": {
    "served_models": [{
      "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
      "model_name": model_version.name,
      "model_version": model_version.version,
      "workload_type": "GPU_MEDIUM",
      "workload_size": "Small",
      # Currently, GPU serving models cannot be scaled down to zero
      "scale_to_zero_enabled": "False" 
    }]
  }
}
endpoint_json = json.dumps(endpoint_config, indent='  ')

# Send a POST request to the API
response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)

if response.status_code != 200:
  print(response.text)
  if "RESOURCE_ALREADY_EXISTS" in response.text:
    response = requests.request(method="PUT", headers=deploy_headers, url=update_url, data=endpoint_json)
    if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  else:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

# Show the response of the POST request
# When first creating the serving endpoint, it should show that the state 'ready' is 'NOT_READY'
# You can check the status on the Databricks model serving endpoint page, it is expected to take ~35 min for the serving endpoint to become ready
print(response.json())


# COMMAND ----------

link_to_served_model = f"/#mlflow/endpoints/{ENDPOINT_NAME}"
print(link_to_served_model)

displayHTML(f'<a href="{link_to_served_model}">View the model monitoring page</a>')

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 - Applying the Model to our Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model
# MAGIC
# MAGIC We'll first load our model from the model registry. 
# MAGIC
# MAGIC Then we'll make it callable via a user-defined function (UDF). A UDF allows us to apply the model to our data in an efficient and distributed manner

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import pandas as pd
import re
from pyspark.sql.functions import col, lit, pandas_udf, PandasUDFType

from config import CATALOG_NAME, SCHEMA_NAME, MLFLOW_MODEL_NAME, USE_UC, USE_VOLUMES

model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

model_info = client.get_registered_model(MLFLOW_MODEL_NAME)
latest_version = model_info.latest_versions[0]
latest_run_id = latest_version.run_id
experiment_id = re.search(r"mlflow-tracking/([a-f0-9]{32})", latest_version.source).group(1)

link_to_model_registry = f"#mlflow/models/{MLFLOW_MODEL_NAME}"
link_to_latest_run = f"#mlflow/experiments/{experiment_id}/runs/{latest_run_id}/artifactPath/model"

displayHTML(f'''
  <a href="{link_to_model_registry}">View the model in Model Registry</a><br/>
  <a href="{link_to_latest_run}">View the latest run artifacts</a>
''')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt Engineering
# MAGIC
# MAGIC Let's start with a simple prompt
# MAGIC
# MAGIC ```plaintext
# MAGIC You are a security governance expert. 
# MAGIC Mask all PII in the given text using format [PII Type] (e.g. [Email], [Phone]). 
# MAGIC Product names are not PII. 
# MAGIC If there is no PII, the masked text is the same as the original text. 
# MAGIC Return JSON only. 
# MAGIC Do not provide explanations. 
# MAGIC
# MAGIC JSON format:
# MAGIC {
# MAGIC     "masked_text": <masked text; same as original text if no PII>,
# MAGIC     "pii_detected": [
# MAGIC         {
# MAGIC             "pii_type": <PII type>, 
# MAGIC             "value": <PII value>
# MAGIC         }
# MAGIC     ]
# MAGIC }
# MAGIC
# MAGIC Text to analyse:
# MAGIC I recently purchased the ElectroMax Pro and I am extremely satisfied with its performance. 
# MAGIC The battery life is amazing and it charges quickly. The customer service provided by the 
# MAGIC company was also excellent. I highly recommend this product to anyone looking for a 
# MAGIC reliable and efficient electronic device. Please contact me at john.doe@example.com for 
# MAGIC any further questions.
# MAGIC ```

# COMMAND ----------

prompt = """
You are a security governance expert. 
Detect all PII in the given text only using format [PII_TYPE].
PII includes, but is not restricted to: EMAIL, FIRST_NAME, LAST_NAME, PHONE, PERSON_NAME, ADDRESS, CREDIT_CARD, CVV, IP_ADDRESS
Product names are not PII. 
Return JSON only. If there is no PII, the "pii_detected" value is an empty array.
Do not provide any other content. 

JSON format:
{
    "pii_detected": [
        {
            "pii_type": <PII type>, 
            "value": <PII value>
        }
    ]
}

Text to analyse:
I recently purchased the ElectroMax Pro and I am extremely satisfied with its performance. 
The battery life is amazing and it charges quickly. The customer service provided by the 
company was also excellent. I highly recommend this product to anyone looking for a 
reliable and efficient electronic device. Please contact me at john.doe@example.com for 
any further questions.
"""

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(model_uri)

# Make a prediction using the loaded model
# input_example=pd.DataFrame({"prompt":["what is ML?", "Name 10 colors."], "temperature": [0.5, 0.2],"max_tokens": [100, 200]})
# print(loaded_model.predict(input_example))

# mpt_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# COMMAND ----------

input_example=pd.DataFrame({"prompt":[prompt], "temperature": [0.1],"max_tokens": [10000]})
prediction = loaded_model.predict(input_example)
print(prediction)

df = spark.createDataFrame(prediction)
display(df)

# COMMAND ----------

df = spark.sql(f"SELECT * FROM {CATALOG_NAME}.{SCHEMA_NAME}.ai4privacy_pii_unmasked_text_sample")
display(df)

# COMMAND ----------

PROMPT_PREFIX = """
You are an expert privacy officer.
In the text to analyse below, detect all PII using format [PII_TYPE].
PII includes, but is not restricted to: EMAIL, FIRST_NAME, LAST_NAME, PHONE, PERSON_NAME, ADDRESS, CREDIT_CARD, CVV, IP_ADDRESS
Product names are not PII.
Do NOT make up any PII; only use what is in the given text. Do NOT hallucinate.
Return JSON only. If there is no PII, the "pii_detected" value is an empty array.
Do not provide any other content. 

JSON format:
{"pii_detected": [{"pii_type": <PII type>, "value": <PII value>}]}

Text to analyse:
"""

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType

@pandas_udf(StringType())
def add_prefix_to_column(data: pd.Series) -> pd.Series:
    return PROMPT_PREFIX + data


# COMMAND ----------

df_prompts = df.withColumn("prompt", add_prefix_to_column(df["unmasked_text"]))
display(df_prompts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define a Spark UDF with our PyFunc Model
# MAGIC
# MAGIC We'll use MLflow's `spark_udf()` method to create a Spark UDF that can be invoked in our code. We configure two specific parameters:
# MAGIC - `model_uri`: pointer to our registerd model
# MAGIC - `env_manager`: `virtualenv` to recreate the software environment that was used to train the model
# MAGIC
# MAGIC [Documentation](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf)

# COMMAND ----------

 mpt_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# COMMAND ----------

# TODO
mlflow.pyfunc.load_model()

def udfxxxx:
  

# COMMAND ----------

mpt_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri,
                                  # [QFB]: This gives similar errors to what model serving was showing.
                                  #   Could this be the culprit?
                                  env_manager="conda"
                                  )

# COMMAND ----------

df_evaluated = df_prompts.withColumn(
  "evaluation", mpt_udf(col("prompt"), lit(0.1), lit(10000))
)
display(df_evaluated)

# COMMAND ----------

# [QFB]: 1 prompt takes 3 mins. Any suggestions for speeding up?
# TODO: check if using GPU 

prompt01 = """
In the text to analyse below (enclosed by >>> <<<), detect PII ONLY in that text block using format [PII_TYPE].
Do NOT make up any PII; only use what is in the given text. Do not include examples.
PII types include: EMAIL, FIRST_NAME, LAST_NAME, PHONE, PERSON_NAME, ADDRESS, CREDIT_CARD, CVV, IP_ADDRESS
The following are not PII: product names, brand names
Return JSON only. JSON:
{"pii_detected": [{"pii_type": <PII type detected in provided text>, "value": <PII value detected in provided text>}]}

Text to analyse:>>>
Attn: August, we need you to conduct a comprehensive study analyzing the impact of industrial culture on employee motivation. Please use your account Savings Account for any necessary purchases and keep track of all transactions for auditing purposes.
<<<"""

data = pd.DataFrame({"prompt":[prompt01], "temperature": [0.01],"max_tokens": [5000]})
df_test = spark.createDataFrame(data)
df_test_result = df_test.withColumn("result", mpt_udf(col("prompt"), col("temperature"), col("max_tokens")))

display(df_test_result)

# COMMAND ----------

(
  df_evaluated
  .write
  .mode("overwrite")
  .option("overwriteSchema", "true")
  .saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.ai4privacy_pii_unmasked_text_sample_evaluated")
)

# COMMAND ----------


