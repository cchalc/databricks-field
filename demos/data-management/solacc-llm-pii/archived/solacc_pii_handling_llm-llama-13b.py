# Databricks notebook source
# MAGIC %md
# MAGIC # Solution Accelerator: PII Handling with LLMs
# MAGIC
# MAGIC ---
# MAGIC **Cluster configurations**
# MAGIC
# MAGIC **TODO**
# MAGIC **EDA / ETL**
# MAGIC - Runtime: 13.3 ML LTS
# MAGIC - Machine: XX CPU + XX GB RAM (for Driver and Worker)
# MAGIC   - XX workers
# MAGIC
# MAGIC **TODO**
# MAGIC **Model Development**
# MAGIC - Runtime: 13.3 LTS ML (GPU)
# MAGIC - Machine: Single Node (1 GPU, 256GB ~128GB~ RAM)
# MAGIC   - AWS: `g4dn.16xlarge` ~`g4dn.8xlarge`~
# MAGIC   - Azure: XXX
# MAGIC   - GCP: XXX
# MAGIC

# COMMAND ----------

# from datasets import load_dataset

# from config import CATALOG_NAME, SCHEMA_NAME, MLFLOW_MODEL_NAME, USE_UC, USE_VOLUMES, VOLUME_NAME, VOLUME_PATH, CACHE_PATH, DOWNLOAD_PATH

# USE_TRITON = True # Set to False on Azure

# Model serving parameters
# MLFLOW_MODEL_NAME = "vinnyv-mpt-7b-instruct"

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 - Data Preparation
# MAGIC
# MAGIC Refer to [Notebook: `01_Data_Download_and_Prep`]($./01_Data_Download_and_Prep)

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 - Model Selection
# MAGIC
# MAGIC Refer to [Notebook: `02_Model_Selection`]($./02_Model_Selection)

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 - Customise & Log model to MLflow
# MAGIC
# MAGIC Now that we know the model works, we need to get it ready to be served. In the below flow we'll do the following:
# MAGIC
# MAGIC - Wrap the model in an [MLflow PyFunc model](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html). This provides us with the flexibility to include custom logic for pre-processing prompts and post-processing responses
# MAGIC - Log the model to [MLflow Model Registry](https://docs.databricks.com/en/mlflow/model-registry.html) for easy maintenance of the full lifecycle of the model
# MAGIC - Serve the model with [Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)

# COMMAND ----------

# MAGIC %pip install -Uq accelerate==0.23.0 bitsandbytes==0.41.1 mlflow transformers==4.33.2 xformers==0.0.21 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python 
# MAGIC
# MAGIC # Previous versions
# MAGIC # bitsandbytes==0.40.0
# MAGIC # transformers==4.31.0
# MAGIC
# MAGIC # Restart the Python kernel
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import accelerate
import pandas as pd
import mlflow
import numpy as np
import re
import sentencepiece
import torch
import transformers

from huggingface_hub import snapshot_download, login, notebook_login
# from langchain import HuggingFacePipeline, LLMChain, PromptTemplate
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from config import HF_MODEL_NAME, HF_MODEL_REVISION_ID, CACHE_PATH, DOWNLOAD_PATH, MLFLOW_MODEL_NAME

# Disable progress bars for cleaner output
transformers.utils.logging.disable_progress_bar()

# MLFLOW_MODEL_NAME = "vinnyv-llama2-13b-chat-hf"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define prompt parameters

# COMMAND ----------

SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer assistant. Always answer as helpfully as possible, while being succinct and safe. The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. 

If you don't know the answer to a question, please don't share false information.
<</SYS>>

"""

USER_INSTRUCTION_PREFIX = """In the text to analyse below detect instances of personally identifiable information (PII).
Personally identifiable information (PII) is information that can be used to identify, contact, or locate a single person, either alone or in combination with other pieces of information. 
You will label each found instance using format [PII_TYPE]. 
PII includes but not limited to: email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver's licence numbers (LICENCE).
Use the context of the text to verify whether proper nouns (e.g. June) in the text is a person's name.
Return JSON only. 
JSON:
{"pii_detected": [{"pii_type": "<PII type detected>", "value": "<PII value detected>"}]}

Text to analyse:
>>>"""

PROMPT_TEMPLATE = f"""{SYSTEM_PROMPT}{USER_INSTRUCTION_PREFIX}"""
# {{text_to_analyse}}
# >>>
# [/INST]"""
# .format(
#   system_prompt=SYSTEM_PROMPT,
#   user_instruction_prefix=USER_INSTRUCTION_PREFIX,
#   text_to_analyse="{text_to_analyse}"
# )

print(f"Prompt template: {PROMPT_TEMPLATE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hugging Face Login
# MAGIC
# MAGIC Because we needed to first register to gain access to Llama 2 models, we need to authenticate with Hugging Face to verify we are able to access the model. 
# MAGIC
# MAGIC You need to provide your Hugging Face token. There are two ways to login:
# MAGIC - `notebook_login()`: UI-driven login
# MAGIC - `login()`: programatically login (it is recommended your token is saved in Databricks Secrets)

# COMMAND ----------

# notebook_login()
login(token="<token>")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create LLM Pipeline
# MAGIC
# MAGIC We utilise our code from the previous Notebook to load our LLM pipeline

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, padding_side="left")

model_config = AutoConfig.from_pretrained(
  HF_MODEL_NAME,
  revision=HF_MODEL_REVISION_ID,
  trust_remote_code=True, # this can be needed if we reload from cache
)

model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME,
        revision=HF_MODEL_REVISION_ID,
        trust_remote_code=True,
        config=model_config,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        # device_map='auto',
        # pad_token_id=tokenizer.eos_token_id,
)
model.tie_weights()

pipe = transformers.pipeline(
  "text-generation", 
  model=model, 
  tokenizer=tokenizer,
  device_map='auto',
  return_full_text=False,
  # framework="pt",
  # clean_up_tokenization_spaces=True,
  # prefix=f"{SYSTEM_PROMPT}{USER_INSTRUCTION_PREFIX}"
)


# COMMAND ----------

text = """
Attn: August, we need you to conduct a comprehensive study analyzing the impact of industrial culture on employee motivation. Please use your account Savings Account for any necessary purchases and keep track of all transactions for auditing purposes. Kind regards, Vinny - vinny.vijeyakumaar@databricks.com
"""

# COMMAND ----------

print(PROMPT_TEMPLATE.format(text_to_analyse=text))

# COMMAND ----------

# DBTITLE 1,Test the pipeline
generate_kwargs = {
  'max_new_tokens': 2048,
  'temperature': 0.1,
  'top_p': 0.65,
  'top_k': 20,
  'repetition_penalty': 1.2,
  'no_repeat_ngram_size': 0,
  'use_cache': False,
  'do_sample': True,
  'eos_token_id': tokenizer.eos_token_id,
  'pad_token_id': tokenizer.eos_token_id,
}

print(f"{PROMPT_TEMPLATE}{text}")

# print(pipe(PROMPT_TEMPLATE.format(text_to_analyse=text), **generate_kwargs))
print(pipe(f"{PROMPT_TEMPLATE}{text}>>>[/INST]", **generate_kwargs))

# COMMAND ----------

# MAGIC %md
# MAGIC ### mlflow.transformers
# MAGIC
# MAGIC We'll now log the model with MLflow using the Transformers flavour. We want to log the model for several reasons:
# MAGIC - It's the precursor to having the model registered in the Model Registry
# MAGIC - It allows us to log various iterations of the model and compare performance across iterations
# MAGIC
# MAGIC Note, this can take up to 10 minutes as MLflow retrieves all the relevant assets from Hugging Face

# COMMAND ----------

# import bitsandbytes
import sentencepiece

MLFLOW_MODEL_NAME = "vv-transformers-llama2-13b"

with mlflow.start_run() as run:  
    mlflow.transformers.log_model(
        pipe,
        "model",
        task="text-generation",
        registered_model_name=MLFLOW_MODEL_NAME,
        inference_config = {
          # "max_length": 2048,
          "include_prompt": False,
          "collapse_whitespace": True,
          'max_new_tokens': 2048,
          'temperature': 0.1,
          'top_p': 0.65,
          'top_k': 20,
          'repetition_penalty': 1.2,
          'no_repeat_ngram_size': 0,
          'use_cache': False,
          'do_sample': True,
          'eos_token_id': tokenizer.eos_token_id,
          'pad_token_id': tokenizer.eos_token_id,
        },
        pip_requirements={
          f"accelerate=={accelerate.__version__}",
          "bitsandbytes==0.41.1", # package doesn't expose __version__
          f"sentencepiece=={sentencepiece.__version__}",
          f"torch=={torch.__version__}"
          f"transformers=={transformers.__version__}",
        },
        await_registration_for=1000,
    )


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
# MAGIC ### Load the model
# MAGIC
# MAGIC Now that we've logged and registered the model, let's load it from the model registry. Notice that we used `mlflow.transformers` to log the model, but now are loading it with `mlflow.pyfunc` as a PyFunc flavour model.
# MAGIC
# MAGIC This is because XXXX
# MAGIC
# MAGIC **If you receive an out of memory error**: 
# MAGIC You may receive an out of memory error trying to run the below code. This is because in the previous steps we've already loaded the model into memory. The easiest way to clear the memory is to `Detach & re-attach` the cluster from this Notebook

# COMMAND ----------

# MAGIC %pip install -Uq accelerate==0.23.0 bitsandbytes==0.41.1 mlflow transformers==4.33.2 xformers==0.0.21 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python 
# MAGIC
# MAGIC # Previous versions
# MAGIC # bitsandbytes==0.40.0
# MAGIC # transformers==4.31.0
# MAGIC
# MAGIC # Restart the Python kernel
# MAGIC dbutils.library.restartPython()
# MAGIC
# MAGIC import mlflow
# MAGIC import pandas as pd

# COMMAND ----------

SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are an expert, helpful, respectful and honest privacy officer assistant. Always answer as helpfully as possible, while being succinct and safe. The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. 

If you don't know the answer to a question, please don't share false information.
<</SYS>>

"""

USER_INSTRUCTION_PREFIX = """In the text to analyse below detect instances of personally identifiable information (PII).
Personally identifiable information (PII) is information that can be used to identify, contact, or locate a single person, either alone or in combination with other pieces of information. 
You will label each found instance using format [PII_TYPE]. 
PII includes but not limited to: email addresses (EMAIL), first names (FIRST_NAME), last names (LAST_NAME), phone numbers (PHONE), person names (NAME), residential addresses (ADDRESS), credit card numbers (CREDIT_CARD), credit card CVVs (CVV), IP addresses (IP_ADDRESS), social security numbers (SSN), date of birth (DOB), driver's licence numbers (LICENCE).
Use the context of the text to verify whether proper nouns (e.g. June) in the text is a person's name.
Return JSON only. 
JSON:
{"pii_detected": [{"pii_type": "<PII type detected>", "value": "<PII value detected>"}]}

Text to analyse:
>>>"""

PROMPT_TEMPLATE = f"""{SYSTEM_PROMPT}{USER_INSTRUCTION_PREFIX}"""

# COMMAND ----------

MLFLOW_MODEL_NAME = "vv-transformers-llama2-13b"

loaded_model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}/latest")
print(f"Loaded model: {MLFLOW_MODEL_NAME}")

# COMMAND ----------

text = """
Attn: August, we need you to conduct a comprehensive study analyzing the impact of industrial culture on employee motivation. Please use your account Savings Account for any necessary purchases and keep track of all transactions for auditing purposes. Kind regards, Vinny - vinny.vijeyakumaar@databricks.com
"""

generate_kwargs = {
  'max_new_tokens': 2048,
  'temperature': 0.1,
  'top_p': 0.65,
  'top_k': 20,
  'repetition_penalty': 1.2,
  'no_repeat_ngram_size': 0,
  'use_cache': False,
  'do_sample': True,
}

print(f"{PROMPT_TEMPLATE}{text}")

# print(pipe(PROMPT_TEMPLATE.format(text_to_analyse=text), **generate_kwargs))

# COMMAND ----------

loaded_model.predict(f"{PROMPT_TEMPLATE}{text}[/INST]", params=generate_kwargs)

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf

@pandas_udf("string")
def profile_pii(texts: pd.Series) -> pd.Series:
  texts_list = texts.to_list()
  texts_prompt_list = [f"{PROMPT_TEMPLATE}{text}[/INST]" for text in texts_list]

  pipe = loaded_model.predict(texts_prompt_list, batch_size=1)
  results = [out[0]["generated_text"] for out in pipe]
  return pd.Series(results)

  # pipe = summarizer(texts.to_list(), truncation=True, batch_size=1)
  # summaries = [summary['summary_text'] for summary in pipe]
  # return pd.Series(summaries)


# COMMAND ----------

# MAGIC %sh nvidia-smi

# COMMAND ----------

# MAGIC %sh nvidia-smi

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import StringType

data = ["Attn: August, we need you to conduct a comprehensive study analyzing the impact of industrial culture on employee motivation. Please use your account Savings Account for any necessary purchases and keep track of all transactions for auditing purposes. Kind regards, Vinny - vinny.vijeyakumaar@databricks.com"]

df = spark.createDataFrame(data, StringType()).withColumnRenamed("value", "text")
result_df = df.withColumn("result", profile_pii(col("text")))

display(result_df)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import StringType

data = ["Attn: August, we need you to conduct a comprehensive study analyzing the impact of industrial culture on employee motivation. Please use your account Savings Account for any necessary purchases and keep track of all transactions for auditing purposes. Kind regards, Vinny - vinny.vijeyakumaar@databricks.com"]

df = spark.createDataFrame(data, StringType()).withColumnRenamed("value", "text")
result_df = df.withColumn("result", profile_pii(col("text")))

display(result_df)

# COMMAND ----------

# Make a prediction using the loaded model
input_example=pd.DataFrame({"prompt":["what is ML?", "Name 10 colors."], "temperature": [0.5, 0.2],"max_tokens": [100, 200]})
print(loaded_model.predict(input_example))

# COMMAND ----------

# Make a prediction using the loaded model
input_example=pd.DataFrame({"prompt":["what is ML?", "Name 10 colors."], "temperature": [0.5, 0.2],"max_tokens": [100, 200]})
print(loaded_model.predict(input_example))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Langchain
# MAGIC
# MAGIC We have several options for packaging up our model for MLflow model registry. For example we can register the following model flavours:
# MAGIC - `transformers`: XXXX
# MAGIC - `pyfunc`: XXXXX
# MAGIC - `langchain`: XXXX
# MAGIC
# MAGIC Because we want to control our prompting logic through prompt templates, `pyfunc` or `langchain` are best suited for us.
# MAGIC
# MAGIC In this example we'll utilise `langchain`. Langchain gives us greater flexibility if we decide to expand our model's use (e.g. for conversational chains, testing out other base models, etc) in future
# MAGIC
# MAGIC ### Resources
# MAGIC - [LLMChain definition](https://docs.langchain.com/docs/components/chains/llm-chain)
# MAGIC - [LLMChain documentation](https://python.langchain.com/docs/modules/chains/foundational/llm_chain)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Langchain
# MAGIC
# MAGIC We have several options for packaging up our model for MLflow model registry. For example we can register the following model flavours:
# MAGIC - `transformers`: XXXX
# MAGIC - `pyfunc`: XXXXX
# MAGIC - `langchain`: XXXX
# MAGIC
# MAGIC Because we want to control our prompting logic through prompt templates, `pyfunc` or `langchain` are best suited for us.
# MAGIC
# MAGIC In this example we'll utilise `langchain`. Langchain gives us greater flexibility if we decide to expand our model's use (e.g. for conversational chains, testing out other base models, etc) in future
# MAGIC
# MAGIC ### Resources
# MAGIC - [LLMChain definition](https://docs.langchain.com/docs/components/chains/llm-chain)
# MAGIC - [LLMChain documentation](https://python.langchain.com/docs/modules/chains/foundational/llm_chain)

# COMMAND ----------


print(PROMPT_TEMPLATE)

# Define Langchain prompt template
prompt_template = PromptTemplate(
  input_variables=["text_to_analyse"],
  template=PROMPT_TEMPLATE
)

# COMMAND ----------


print(PROMPT_TEMPLATE)

# Define Langchain prompt template
prompt_template = PromptTemplate(
  input_variables=["text_to_analyse"],
  template=PROMPT_TEMPLATE
)

# COMMAND ----------

llm = HuggingFacePipeline(pipeline=pipe)

llama2_chain = LLMChain(
  llm=llm,
  prompt=prompt_template,
  verbose=True, # for debugging
)

# COMMAND ----------

llm = HuggingFacePipeline(pipeline=pipe)

llama2_chain = LLMChain(
  llm=llm,
  prompt=prompt_template,
  verbose=True, # for debugging
)

# COMMAND ----------



result = llama2_chain({"text_to_analyse": text})
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### mlflow.langchain

# COMMAND ----------

MLFLOW_MODEL_NAME = "vv-langchain-llama2-13b"

with mlflow.start_run() as run:  
    mlflow.langchain.log_model(
        llama2_chain,
        "model",
        registered_model_name=MLFLOW_MODEL_NAME,
        await_registration_for=1000,
        # loader_fn=retriever,
        # persist_dir=chroma_cache
        # pip_requirements=["torch", "transformers", "accelerate"],
        # input_example=input_example,
        # signature=signature,
    )

# COMMAND ----------

import re
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

MLFLOW_MODEL_NAME = "vv-pyfunc-llama2-13b"

# COMMAND ----------

# If the model has been downloaded previously in the same session, this will not repetitively download large model files, 
#   but only the remaining files in the repo

snapshot_location = snapshot_download(
  repo_id=HF_MODEL_NAME, 
  revision=HF_MODEL_REVISION_ID, 
  ignore_patterns="*.safetensors"
)

print(f"Model snapshot saved to: {snapshot_location}")

# COMMAND ----------

# DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being succinct and safe. Do not provide explanations. If you don't know the answer to a question, please don't share false information."

# PROMPT_TEMPLATE

class Llama2(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts['repository'], 
            padding_side="left"
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            load_in_8bit=True,
            trust_remote_code=True,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id
        )
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        # return f"""<s>[INST]<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n\n{instruction}[/INST]\n"""
        return PROMPT_TEMPLATE.format(text_to_analyse=instruction)

    def _generate_response(self, prompt, temperature, max_new_tokens):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(encoded_input, 
                                     do_sample=True, 
                                     temperature=temperature, 
                                     max_new_tokens=max_new_tokens)
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return generated_response
      
    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """

        outputs = []

        for i in range(len(model_input)):
          prompt = model_input["prompt"][i]
          temperature = model_input.get("temperature", [0.01])[i]
          max_new_tokens = model_input.get("max_new_tokens", [2048])[i]

          outputs.append(self._generate_response(prompt, temperature, max_new_tokens))
      
        # {"candidates": [...]} is the required response format for MLflow AI gateway -- see 07_ai_gateway for example
        return {"candidates": outputs}


# COMMAND ----------

# class MPT(mlflow.pyfunc.PythonModel):
#     def load_context(self, context):
#         """
#         This method initializes the tokenizer and language model
#         using the specified model repository.
#         """
#         # Initialize tokenizer and language model
#         self.tokenizer = transformers.AutoTokenizer.from_pretrained(
#           context.artifacts['repository'], padding_side="left")

#         config = transformers.AutoConfig.from_pretrained(
#             context.artifacts['repository'], 
#             trust_remote_code=True,
#             init_device = 'cuda', # For fast initialization directly on GPU
#         )
#         # [QFB]: why does switching on Triton cause CUDA OOM errors?
#         # config.attn_config['attn_impl'] = 'triton'
        
#         self.model = transformers.AutoModelForCausalLM.from_pretrained(
#             context.artifacts['repository'], 
#             config=config,
#             device_map="auto",
#             torch_dtype=torch.bfloat16, # Load model weights in bfloat16 # TODO: G5s, or float 16
#             trust_remote_code=True)
#         # [QFB]: does this interfere with init_device='cuda'? Or is it redundant?
#         # self.model.to(device='cuda')
        
#         self.model.eval()


#     def _build_prompt(self, instruction):
#         """
#         This method generates the prompt for the model.
#         """
#         INSTRUCTION_KEY = "### Instruction:"
#         RESPONSE_KEY = "### Response:"
#         INTRO_BLURB = (
#             "Below is an instruction that describes a task. "
#             "Write a response that appropriately completes the request."
#         )

#         return f"""{INTRO_BLURB}
#         {INSTRUCTION_KEY}
#         {instruction}
#         {RESPONSE_KEY}
#         """


#     def predict(self, context, model_input):
#         """
#         This method generates prediction for the given input.
#         """
#         generated_text = []
#         for index, row in model_input.iterrows():
#           prompt = row["prompt"]
#           # You can add other parameters here
#           temperature = model_input.get("temperature", [0.1])[0]
#           max_new_tokens = model_input.get("max_new_tokens", [1000])[0]
#           full_prompt = self._build_prompt(prompt)
#           encoded_input = self.tokenizer.encode(full_prompt, return_tensors="pt").to("cuda")
#           output = self.model.generate(
#             encoded_input, 
#             do_sample=True, 
#             temperature=temperature,
#             max_new_tokens=max_new_tokens)
#           prompt_length = len(encoded_input[0])
#           generated_text.append(
#             self.tokenizer.batch_decode(
#               output[:,prompt_length:], 
#               skip_special_tokens=True
#             )
#           )
#         return pd.Series(generated_text)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the model to MLflow

# COMMAND ----------

# DBTITLE 1,Configure and log MLflow Model
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

import pandas as pd

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_new_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["what is ML?"], 
            "temperature": [0.5],
            "max_new_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 5 minutes to complete
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Llama2(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the Model in MLflow Model Registry
# MAGIC
# MAGIC This may take about 6 minutes to complete.
# MAGIC
# MAGIC A new model registry entry will be created if one doesn't already exist for the given model name. If an entry already exists, a new version of the model is registered.
# MAGIC
# MAGIC If the below API call times out, you can view the progress of model registration in the [Registered Models UI](#mlflow/models)

# COMMAND ----------

result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    name=MLFLOW_MODEL_NAME,
    await_registration_for=1000,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the model from Model Registry
# MAGIC
# MAGIC Now that we've packaged and registered our model, let's load it from model registry, and test some prompts
# MAGIC
# MAGIC Assume that the below code is run separately or after the memory cache is cleared.
# MAGIC You may need to cleanup the GPU memory.

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}/latest")
print(f"Loaded model: {MLFLOW_MODEL_NAME}")

# COMMAND ----------

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

from config import CATALOG_NAME, SCHEMA_NAME, MLFLOW_MODEL_NAME_P, USE_UC, USE_VOLUMES, TEST_DATA, PROMPT_TEMPLATE_V2

model_uri = f"models:/{MLFLOW_MODEL_NAME_P}/latest"

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

model_info = client.get_registered_model(MLFLOW_MODEL_NAME_P)
latest_version = model_info.latest_versions[0]
latest_run_id = latest_version.run_id
experiment_id = re.search(r"mlflow-tracking/([a-f0-9]{32})", latest_version.source).group(1)

link_to_model_registry = f"#mlflow/models/{MLFLOW_MODEL_NAME_P}"
link_to_latest_run = f"#mlflow/experiments/{experiment_id}/runs/{latest_run_id}/artifactPath/model"

displayHTML(f'''
  <a href="{link_to_model_registry}">View the model in Model Registry</a><br/>
  <a href="{link_to_latest_run}">View the latest run artifacts</a>
''')


# COMMAND ----------

# MAGIC %sh rm -rf /tmp/tmp*

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

# COMMAND ----------

# input_example=pd.DataFrame({"prompt":[TEST_DATA[0]], "temperature": [0.1],"max_new_tokens": [2048]})
input_example=pd.DataFrame({"prompt":[TEST_DATA[0]]})
prediction = loaded_model.predict(input_example)
print(prediction)

df = spark.createDataFrame([prediction])
display(df)

# COMMAND ----------

df.createOrReplaceTempView("inference")

# COMMAND ----------

# MAGIC %sql SELECT * FROM inference

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   from_json('{
# MAGIC "pii_detected": [
# MAGIC {
# MAGIC "pii_type": "EMAIL",
# MAGIC "value": "vinny.vijeyakumaar@databricks.com"
# MAGIC },
# MAGIC {
# MAGIC "pii_type": "NAME",
# MAGIC "value": "August"
# MAGIC }
# MAGIC ]
# MAGIC }', 'struct<pii_detected:array<struct<pii_type:string, value:string>>>' 
# MAGIC   ) as parsed_json
# MAGIC FROM inference

# COMMAND ----------

df = spark.createDataFrame([prediction])
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

loaded_model = mlflow.pyfunc.load_model(model_uri)

@pandas_udf("string")
def predict(texts: pd.Series) -> pd.Series:
  results = loaded_model.predict(texts)
  return pd.Series(results)

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


