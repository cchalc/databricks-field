# Databricks notebook source
# MAGIC %md
# MAGIC # Model Selection
# MAGIC
# MAGIC Now that we have our data, we can turn our attention to selecting an LLM for our use case and defining the prompts.
# MAGIC
# MAGIC ## Considerations for model selection
# MAGIC
# MAGIC First we need to consider which LLM is going to handle our use case. There are a number of factors we need to consider:
# MAGIC - **License**: there are many open-source LLMs that are open for commercial usage
# MAGIC - **Use case**: you can utilise general conversational models but also specific models tailored for tasks such as translation, summarisation, etc.
# MAGIC - **Speed**: do we require a fast response (e.g. for real-time interactions) or can we tolerate latency (e.g. for batch data processing)?
# MAGIC - **Quality**
# MAGIC
# MAGIC Databricks has a [list of recommended models](https://www.databricks.com/product/machine-learning/large-language-models-oss-guidance) based on the use case, desired speed and quality.
# MAGIC
# MAGIC <img src="https://github.com/vinoaj/databricks-resources/blob/main/assets/img/databricks-large-language-models-oss-guidance.png?raw=true" width="600px">
# MAGIC
# MAGIC You can also refer to the [Hugging Face Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for up-to-date evaluation rankings for open source LLMs and chatbots.
# MAGIC
# MAGIC ## Working with Llama 2 13B
# MAGIC
# MAGIC For the purposes of this Solution Accelerator we select the [`Llama-2-13b-chat-hf`](meta-llama/Llama-2-13b-chat-hf). We've done so for the following reasons:
# MAGIC
# MAGIC - 13B models are a good compromise between the needs for speed and quality
# MAGIC   - While testing some 7B models we encounted some false positives. 13B minimised these false positives.
# MAGIC - Utilising an instruction-following model allows us to adapt it for a variety of use cases
# MAGIC - The licence permits commercial use
# MAGIC
# MAGIC The model it self is very robust, has good performance and can take on language tasks easily. It features a 70B parameter version as well, but that would probably be an overkill for our use case
# MAGIC
# MAGIC Meta's Llama 2 13B Base model is a pre-trained model, however it has not been fine-tuned for a specific task. It would be a great candidate if we wanted to fine-tune it for a specific task which we have training data for. Where as the Chat model has been trained on a Instructions dataset, and is more ready to follow instructions
# MAGIC
# MAGIC ## Alternative models
# MAGIC
# MAGIC Of course, Llama 2 is one of many models we considered. You may wish to experiment with other models as well.
# MAGIC
# MAGIC Some other models you can also check out are:
# MAGIC * [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)
# MAGIC * [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)
# MAGIC * [Llama-2-7B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
# MAGIC
# MAGIC ___
# MAGIC
# MAGIC ## Cluster configuration
# MAGIC
# MAGIC For this Notebook we'll utilise a cluster with a GPU. The GPU is necessary for loading and executing the Llama 2 model.
# MAGIC
# MAGIC - Single node
# MAGIC - Access mode: `Assigned` (can use Shared if not using cluster init script)
# MAGIC - DBR: `13.3+ ML (GPU)`
# MAGIC - Node type: 1 GPU, 256GB memory, 32 cores
# MAGIC   - AWS: `g5.16xlarge`
# MAGIC   - Azure: XXXX
# MAGIC   - GCP: XXXX
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Getting access to Llama-2-13b-chat-hf
# MAGIC
# MAGIC First, we need to gain access to Llama 2 by accepting Meta's terms of services. The steps to do so are outlined below.
# MAGIC
# MAGIC ## Gain access to Llama 2 model
# MAGIC
# MAGIC You will first need to register for access to Meta's Llama 2 model in two places:
# MAGIC
# MAGIC - [Log into **Hugging Face** and accept Meta TOS](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/tree/main)
# MAGIC - Also [register your interest on **Meta's** site](https://ai.meta.com/resources/models-and-libraries/llama-downloads) with the **same** email address you use for your Hugging Face login
# MAGIC
# MAGIC You should get an approval response within a few hours.
# MAGIC
# MAGIC ## Generate Hugging Face token
# MAGIC
# MAGIC Next, [generate a Hugging Face access token](https://huggingface.co/settings/token). This will be required to access the model.
# MAGIC - [Token creation page](https://huggingface.co/settings/token)
# MAGIC
# MAGIC ## Save Hugging Face token to Databricks Secrets
# MAGIC
# MAGIC Throughout the Solution Accelerator you can utilise your token in plain text. However, for completeness of security, we recommend that you store your token in [Databricks Secrets](https://docs.databricks.com/en/security/secrets/index.html). This will ensure your token is obfuscated in your source code.
# MAGIC
# MAGIC You will need to use the [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/index.html) to do this. For this example we'll use:
# MAGIC
# MAGIC - Scope: `tokens`
# MAGIC - Key name: `hf_token`
# MAGIC
# MAGIC The CLI steps are:
# MAGIC
# MAGIC - Create the secrets scope: `databricks secrets create-scope tokens`
# MAGIC - Add our token as a secret: `databricks secrets put-secret tokens hf_token --string-value {paste-hugging-face-token-here}`
# MAGIC
# MAGIC ## Related documentation
# MAGIC
# MAGIC - [`huggingface_hub.notebook_login`](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/login#huggingface_hub.notebook_login)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Install libraries
# MAGIC
# MAGIC We'll first install some required libraries to work with Llama 2 and Hugging Face Transformers. As we'll be doing this quite regularly through our exploration, it would be more efficient to load these as part of an init script for your cluster
# MAGIC
# MAGIC You can find a [sample init script here]($./init_script.sh)

# COMMAND ----------

# DBTITLE 1,Install required libraries
# Uncomment the below if you're not using a cluster init script
# See init_script.sh for a sample init script
# Documentation on using init scripts: https://docs.databricks.com/en/init-scripts/index.html

# %pip install -Uq accelerate==0.23.0 bitsandbytes==0.41.1 mlflow transformers==4.33.2 xformers==0.0.21 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

# # Restart the Python kernel
# dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import libraries
import html
import json
import pandas as pd
import transformers
import torch
import llm_utils

from huggingface_hub import login, notebook_login, snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from config import (
    HF_MODEL_NAME,
    HF_MODEL_REVISION_ID,
    SECRET_SCOPE,
    SECRET_KEY_HF,
    SYSTEM_PROMPT,
    TEST_DATA,
    USER_INSTRUCTION_CATEGORISE_DEPARTMENT,
    USER_INSTRUCTION_FILTER_PII_TYPES,
    USER_INSTRUCTION_PROFILE_PII,
)

username = spark.sql("SELECT CURRENT_USER() as user").collect()[0]["user"]

# Disable progress bars for cleaner output
transformers.utils.logging.disable_progress_bar()


# COMMAND ----------

# MAGIC %md
# MAGIC # Hugging Face Login
# MAGIC
# MAGIC Because we needed to first register to gain access to Llama 2 models, we need to authenticate with Hugging Face to verify we are able to access the model.
# MAGIC
# MAGIC You need to provide the Hugging Face token you generated earlier. There are two ways to login:
# MAGIC
# MAGIC - `notebook_login()`: UI-driven login
# MAGIC - `login()`: programatically login (it is recommended your token is saved in Databricks Secrets)

# COMMAND ----------

# DBTITLE 1,Login to Hugging Face to get access to the model
# notebook_login()
token = dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY_HF)
login(token=token)


# COMMAND ----------

# MAGIC %md
# MAGIC # LLM Pipeline
# MAGIC
# MAGIC We are now going to use the the `transformers` library from Hugging Face to download & load the model.
# MAGIC
# MAGIC You can refer to this [GitHub repository (`databricks-ml-examples`)](https://github.com/databricks/databricks-ml-examples) for examples of how to load common open LLMs into Databricks.
# MAGIC
# MAGIC The below step may take several minutes as close to 15GB of artifacts need to be downloaded from Hugging Face's model repository.
# MAGIC
# MAGIC We use `pipelines` to simplify inference. [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) ease the complexity of utilising chosen models for inference by chaining together all the necessary steps.
# MAGIC
# MAGIC [TextGeneration](https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/pipelines#transformers.TextGenerationPipeline) pipelines simplify working with text generation models. In the pipeline you define
# MAGIC
# MAGIC - Model: utilised for inference
# MAGIC - Tokenizer: used to pre-processing inputs
# MAGIC
# MAGIC The pipeline abstracts away the need for you to implement logic for processes such as:
# MAGIC
# MAGIC - Tokenising prompts
# MAGIC - Tokenising and cleaning up responses
# MAGIC - Removing the original instructions from the returned response
# MAGIC - etc.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download model artificats
# MAGIC
# MAGIC We'll first download a snapshot of the model artifacts. By default, `snapshot_download` downloads the artifacts to a cache on local disk. However, if the cluster shuts down or we wish to run this Notebook on another cluster, the cache is lost and we need to download everything all over again. So we explicitly set the `cache_dir` to point to a Workspace location, so that the cache is persisted across sessions.
# MAGIC
# MAGIC When running this command, if the files are already present in the cache, those files will not be downloaded.

# COMMAND ----------

CACHE_PATH = f"/Users/{username}/hf_cache"

snapshot_location = snapshot_download(
    repo_id=HF_MODEL_NAME,
    revision=HF_MODEL_REVISION_ID,
    ignore_patterns="*.bin",
    cache_dir=CACHE_PATH,
)


displayHTML(
    f"""
  Working with model & revision ID: <b>{HF_MODEL_NAME} : {HF_MODEL_REVISION_ID}</b><br/>
  <a href='{llm_utils.generate_model_revision_url(HF_MODEL_NAME, HF_MODEL_REVISION_ID)}'>View source files on Hugging Face</a><br/>
  Cache folder: {snapshot_location}
"""
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Pipeline

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(
    snapshot_location, padding_side="left"
)

model_config = AutoConfig.from_pretrained(
    snapshot_location,
    trust_remote_code=True,  # this can be needed if we reload from cache
)

model = AutoModelForCausalLM.from_pretrained(
    snapshot_location,
    use_safetensors=True,
    trust_remote_code=True,
    config=model_config,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
)
model.tie_weights()

pipe = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    return_full_text=False,
    torch_dtype=torch.bfloat16,
)

# Required tokenizer setting for batch inference
pipe.tokenizer.pad_token_id = tokenizer.eos_token_id
pipe.tokenizer.eos_token_id = tokenizer.eos_token_id


# COMMAND ----------

# MAGIC %md
# MAGIC # Llama 2 Prompt template
# MAGIC
# MAGIC To get the best out of a Llama 2 model, we utilise a [specific prompt template](https://huggingface.co/blog/llama2#how-to-prompt-llama-2). The tokens (e.g. `<s>[INST]`) used mimic what was provided in the training datasets for the model. Utilising the same format during training will give us the best results.
# MAGIC
# MAGIC If you utilise another model, check the model's documentation for recommended prompt templates, and adjust your prompt templates accordingly.
# MAGIC
# MAGIC For Llama 2, the template consists of two sections:
# MAGIC - **System prompt**: this is the base set of instructions for the model. It will set the tone for the overarching purpose of the model's responses. Think of it as defining the "operating system" of the model
# MAGIC - **User instructions**: this are the instructions (or prompt) provided by the user
# MAGIC
# MAGIC ```plaintext
# MAGIC <s>[INST] <<SYS>>
# MAGIC {system_prompt}
# MAGIC <</SYS>>
# MAGIC
# MAGIC {user_instructions}
# MAGIC [/INST]
# MAGIC ```
# MAGIC
# MAGIC As we move through this solution, we'll take any provided user instructions, and wrap it up in the system prompt (in the `{user_instructions}` placeholder) before submitting the instructions to the model.
# MAGIC
# MAGIC ## System prompt
# MAGIC
# MAGIC After some experimentation, we found this to be an effective system prompt:
# MAGIC
# MAGIC ```plaintext
# MAGIC <s>[INST] <<SYS>>
# MAGIC You are an expert, helpful, respectful and honest privacy officer. Always answer as helpfully as possible, while being safe. 
# MAGIC The primary purpose of your job is to carefully and thoughtfully identify privacy risks in any given documentation. If you 
# MAGIC don't know the answer to a question, please don't share false information.
# MAGIC <</SYS>>
# MAGIC
# MAGIC {instruction}
# MAGIC [/INST]
# MAGIC ```
# MAGIC
# MAGIC There are a few aspects of this prompt that is beneficial for us:
# MAGIC - `You are an expert, helpful, respectful and honest **privacy officer** assistant`: indicates the persona the model shoud mimic
# MAGIC - `...while being **safe**`: ensures it doesn't respond with harmful or controversial content
# MAGIC - `...don't share **false information**`: reduces the risk of hallucinations in the response
# MAGIC
# MAGIC ## User instructions
# MAGIC
# MAGIC With experimentation, we found these to be effective formats for the user instructions we'll use later
# MAGIC

# COMMAND ----------

displayHTML(
    f"""<h1>User Instruction Prompt Templates</h1>
<h2>Prompt: Categorise departments message belongs to</h2><pre>{USER_INSTRUCTION_CATEGORISE_DEPARTMENT}</pre><hr />
<h2>Prompt: Detect PII</h2><pre>{html.escape(USER_INSTRUCTION_PROFILE_PII)}</pre><hr />
<h2>Prompt: Filter PII types for exclusion</h2><pre>{USER_INSTRUCTION_FILTER_PII_TYPES}</pre>
"""
)


# COMMAND ----------

# MAGIC %md
# MAGIC You'll notice a few key elements in these prompts. Keep these in mind as you iterate over these prompts for your specific contexts.
# MAGIC
# MAGIC ## Respond in JSON
# MAGIC
# MAGIC - In most cases we ask the model to respond with a `JSON` representation of its findings. We are also prescriptive of the JSON object's schema. This gives us the benefit of handling the responses as data objects downstream.
# MAGIC - Because we're prescriptive of the JSON schema, we can use the `FROM_JSON()` SQL function to convert the LLM's string responses to structured data types.
# MAGIC - Once we have structured data types, it makes it easier for us to query and manipulate that data in our ETL pipelines.
# MAGIC
# MAGIC ## Wrap messages within tags
# MAGIC
# MAGIC - In some cases, the LLM can get confused between the user's instructions and the raw messages included in the instruction.
# MAGIC - For these scenarios, we delineate the content to be analysed between hard-to-miss tags. This makes it clear to the LLM as to which portion of the prompt is its instructions and which portion is the content to analyse.
# MAGIC - For example, in the PII profiling prompt, we inject the raw text to analyse between the `<start of text to analyse>` and `<end of text to analyse>` tags. 
# MAGIC
# MAGIC ## Provide examples
# MAGIC
# MAGIC - It's helpful in most instances to also provide examples of concepts and expected outputs. This further ensures the LLM responds in a manner that meets your expectations.
# MAGIC
# MAGIC ## Language
# MAGIC
# MAGIC Our instructions are defined in natural language English. LLMs are trained on trillions of tokens based off written material (e.g. web pages, academic research, blog posts, code comments, etc.). Therefore, LLMs best respond to language that mimics what they were trained on.
# MAGIC
# MAGIC Some tips for prompting:
# MAGIC - Write your instructions as if you are speaking to a teenager.
# MAGIC - Be clear in your language and avoid ambiguous instructions.
# MAGIC - Clearly define terms and concepts (e.g. the definition of PII).
# MAGIC - Order your instructions in a logical order of operations. If you are not getting the intended result, try reordering your instructions.
# MAGIC - You may need to repeat instructions. For example, in the PII profiling prompt, we tell it multiple times to only return `JSON` and no other text.
# MAGIC - Avoid spelling and grammatical mistakes. Minor spelling mistakes or the misplacement of a comma can alter the results.
# MAGIC
# MAGIC ## Prompting Resources
# MAGIC
# MAGIC Here are some helpful guides on appropriate Llama 2 system messages:
# MAGIC - [Hugging Face's guidance](https://huggingface.co/blog/llama2#how-to-prompt-llama-2)
# MAGIC

# COMMAND ----------

# DBTITLE 1,Function to test prompts
# Define parameters to generate text
def gen_text(prompts, use_template=False, **kwargs):
    """
    Generates text based on the given prompts using a pre-trained language model.

    Parameters:
    - prompts (list): A list of strings containing the prompts for text generation.
    - use_template (bool): If True, formats each prompt with a pre-defined template before text generation. Default is False.

    Returns:
    - A list of strings containing the generated texts corresponding to the input prompts.
    """

    if use_template:
        full_prompts = [
            SYSTEM_PROMPT.format(instruction=prompt) for prompt in prompts
        ]
    else:
        full_prompts = prompts

    if "batch_size" not in kwargs:
        kwargs["batch_size"] = 1

    # the default max length is pretty small (20), which would cut the generated output in the middle, 
    #   so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    # configure other text generation arguments, see common configurable args here: 
    #   https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    kwargs.update(
        {
            # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    outputs = pipe(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]

    return outputs


# COMMAND ----------

# MAGIC %md
# MAGIC # Testing the pipeline
# MAGIC
# MAGIC Now let's test the pipeline with some generic prompts and for profiling PII data in given text.
# MAGIC
# MAGIC You'll notice that we're passing the pipeline some extra arguments. These arguments help us control the output and performance of the LLM.
# MAGIC
# MAGIC ```
# MAGIC {
# MAGIC   "max_new_tokens": 2048,
# MAGIC   "temperature": 0.01,
# MAGIC   "top_p": 0.65,
# MAGIC   "top_k": 20,
# MAGIC   "repetition_penalty": 1.2,
# MAGIC   "no_repeat_ngram_size": 0,
# MAGIC   "use_cache": False,
# MAGIC   "do_sample": True,
# MAGIC   "eos_token_id": tokenizer.eos_token_id,
# MAGIC   "pad_token_id": tokenizer.eos_token_id,
# MAGIC   "batch_size": 1,
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC There are a few key arguments to be aware of, as you iterate your prompts and pipelines:
# MAGIC - `max_new_tokens`: this controls the maximum number of tokens for the LLM to respond with. Llama 2 supports a maximum of 2048 new tokens. You can set lower values, but keep in mind that the response may get truncated.
# MAGIC - `temperature`: this controls the "creativity" of the responses. Here we set it to a low value (`0.01`) to limit the randomness of the responses. While LLMs are not idempotent, lowering the temperature increases the chances of receiving repeatable responses for the same prompt.
# MAGIC - `do_sample`: enables sampling in decoding strategies, often leading to faster inference times
# MAGIC
# MAGIC You can [find out more about utilising these arguments here](https://huggingface.co/docs/transformers/generation_strategies).
# MAGIC
# MAGIC **However**, we don't need to worry too much about any performance-related arguments as we'll be using GPU Model Serving optimised for LLMs. When you deploy your model for serving, the service will take care of the appropriate performance optimisations for your model.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generic prompts (with and without prompt templates)
# MAGIC
# MAGIC Below we test a simple question with and without the prompt template. We notice a difference in the answer when the system prompt is applied.
# MAGIC
# MAGIC In the second example, we get a more useful answer. This is because:
# MAGIC - We've provided an expected prompt format
# MAGIC - We've defined the model's persona (privacy officer) in the system prompt

# COMMAND ----------

# DBTITLE 1,Inference WITHOUT prompt template
# Inference WITHOUT prompt template
results = gen_text(
    ["What is the definition of personally identifiable information?"],
    use_template=False,
    max_new_tokens=2048,
    temperature=0.01,
)
print(results[0])


# COMMAND ----------

# DBTITLE 1,Inference WITH prompt template
# Inference WITH prompt template
results = gen_text(
    ["What is the definition of personally identifiable information?"],
    use_template=True,
    max_new_tokens=2048,
    temperature=0.01,
)
print(results[0])


# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing PII profiling
# MAGIC
# MAGIC Now let's test our model for dealing with PII. We already have some data in `config.TEST_DATA` to play around with

# COMMAND ----------

df = pd.DataFrame(TEST_DATA, columns=["test_data"])
spark_df = spark.createDataFrame(df).cache()
display(spark_df)


# COMMAND ----------

# DBTITLE 1,Inference using gen_text()
generate_kwargs = {
    "max_new_tokens": 2048,
    "temperature": 0.01,
    "top_p": 0.65,
    "top_k": 20,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 0,
    "use_cache": False,
    "do_sample": True,
}

response = gen_text(
    [USER_INSTRUCTION_PROFILE_PII.format(text_to_analyse=TEST_DATA[0])],
    use_template=True,
    **generate_kwargs,
)

print(response[0].strip())


# COMMAND ----------

# DBTITLE 1,Inference using the pipeline directly
generate_kwargs = {
    "max_new_tokens": 2048,
    "temperature": 0.01,
    "top_p": 0.65,
    "top_k": 20,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 0,
    "use_cache": False,
    "do_sample": True,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.eos_token_id,
    "batch_size": 1,
}

# Take our test data and inject it into the prompt templates
test_prompts = [
    SYSTEM_PROMPT.format(
        instruction=USER_INSTRUCTION_PROFILE_PII.format(text_to_analyse=data_item)
    )
    for data_item in TEST_DATA
]

results = pipe(test_prompts, **generate_kwargs)
print(results)


# COMMAND ----------

# MAGIC %md
# MAGIC ### View results in cleaner format
# MAGIC
# MAGIC Now let's look at the results in a cleaner format. The steps we take here will also be steps we'll need to take when performing processing on our data

# COMMAND ----------

# Extract the 'generated_text' field, strip whitespace, and remove newlines
cleaned_texts = [
    result[0]["generated_text"].strip().replace("\n", "") for result in results
]

# Convert the cleaned strings to dictionaries
parsed_results = [json.loads(text) for text in cleaned_texts]

# Pretty print the results
print(json.dumps(parsed_results, indent=2))


# COMMAND ----------

# MAGIC %md
# MAGIC # Saving prompt templates in config.py
# MAGIC
# MAGIC Now that we're happy with our pipeline and the prompt design, we'll save it to [`config.py`]($./config.py) so that we can easily reference them in subsequent notebooks. 
# MAGIC
# MAGIC In this Notebook we've only played around with the `USER_INSTRUCTION_PROFILE_PII` prompt template. In the ETL Notebook, we'll get to see the other prompts in action.
# MAGIC
# MAGIC We save them as:
# MAGIC - `SYSTEM_PROMPT`: the system prompt that works for us
# MAGIC - `USER_INSTRUCTION_PROFILE_PII`: the user instruction template for profiling PII in text
# MAGIC - `USER_INSTRUCTION_CATEGORISE_DEPARTMENT`: the user instruction template for categorising messages to departments
# MAGIC - `USER_INSTRUCTION_FILTER_PII_TYPES`: the user instruction template for identifying which PII types not to redact based on departmental needs
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Next Step: Registering the model and Deployment to model serving
# MAGIC
# MAGIC Now it's time to [proceed to the next notebook]($./03_Model_Registration_and_Deployment) where we'll register the model in Model Registry

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
