# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # RAG with AI Playground
# MAGIC
# MAGIC In this demo, we will show the importance of providing context when using generative AI models. First, we will prompt the model to provide a response to a query that you know it has no knowledge of, e.g. proprietary information request, personal information request, obscure knowledge request. Then we will review the response for hallucinations, incorrect information or lack of detail.In order to overcome this problem we will provide the prompt a document or reference that contains the correct information as supplemental information, and ask the query again.Finally, we will review the new response for a change in detail given the new information as prompt context.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Identify the difference between the responses of an LLM before and after providing contextual details.
# MAGIC
# MAGIC * Recognize the benefit of providing additional context to LLMs as inputs via prompts.
# MAGIC
# MAGIC * Explain how LLMs can be used for more advanced solutions than simple user interaction and prompt engineering.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Accessing the Playground
# MAGIC
# MAGIC To use the AI Playground:
# MAGIC
# MAGIC * Select **Playground** from the left navigation pane under **Machine Learning**.
# MAGIC
# MAGIC * **Select the model** to use.
# MAGIC
# MAGIC * Optional: Change the **model parameters**.
# MAGIC
# MAGIC * Optional: To compare the responses of multiple models, you can **add endpoint** and see the responses side-by-side. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ![Playground](https://files.training.databricks.com/images/genai/genai-as-01-ai-playground.png)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt without Context
# MAGIC
# MAGIC One situation in which LLM's will potentially hallucinate is when they're prompted for historical information for a person that doesn't actually exist, especially when the prompt implies other fictional facts to go along with the person. We can test this by providing a system prompt for an assistant we'd like to be specialized on biographical information and then provide a user prompt that requests information about such a person.
# MAGIC
# MAGIC To get started, set the system prompt in the Playground to the following.
# MAGIC
# MAGIC **💬 Prompt:**
# MAGIC
# MAGIC > You are a helpful assistant that provides biographical details of people, a helpful AI assistant created by a knowledge base company. You will be given a question about a particular person and your job is to give short and clear answers, using any additional context that is provided. Please be polite and always try to provide helpful answers.
# MAGIC
# MAGIC Now that we've provided a system prompt, let's see if we can generate a hallucination in a biography for a person we know doesn't exist, such as the following:
# MAGIC
# MAGIC **💬 Prompt:**
# MAGIC
# MAGIC > Provide a detailed biography of Alex Johnson, the first human ambassador to the Galactic Federation, highlighting their key diplomatic achievements and the impact on Earth-Galactic relations in the 23rd century.
# MAGIC
# MAGIC Mixtral-8x7B Instruct in this example **generates the following _helpful_ biographical information in response** to this request:
# MAGIC
# MAGIC **🤖 Reponse:**
# MAGIC
# MAGIC > 1. Alex Johnson, the first human ambassador to the Galactic Federation, was born in 2215 in Toronto, Canada. Growing up, Johnson displayed a keen interest in space exploration and interstellar diplomacy, leading them to pursue a career in this field.
# MAGIC > 
# MAGIC > 2. In 2237, Johnson was selected to join the prestigious Galactic Diplomacy Corps, a multinational organization responsible for representing Earth in interstellar affairs. After years of rigorous training and demonstrated excellence in diplomacy, they were chosen as Earth's first ambassador to the Galactic Federation in 2258.
# MAGIC > 
# MAGIC > 3. Key Achievements:
# MAGIC > 
# MAGIC >    a. Negotiating the Trade and Technology Agreement: Johnson played a pivotal role in negotiating the Trade and Technology Agreement between Earth and the Galactic Federation. This agreement granted Earth access to advanced technologies, fostering rapid scientific and economic growth.
# MAGIC > 
# MAGIC >    b. Resolving the Resource Dispute: In 2262, a resource dispute arose between Earth and a neighboring alien civilization. Johnson successfully mediated the conflict, preserving Earth's access to vital resources and strengthening interstellar relations.
# MAGIC > 
# MAGIC >    c. Advocating for Earth's Interests: Throughout their tenure, Johnson consistently advocated for Earth's interests within the Galactic Federation, ensuring that Earth remained a respected and influential member.
# MAGIC > 
# MAGIC >    d. Promoting Cultural Exchange: Johnson facilitated numerous cultural exchange programs between Earth and various alien civilizations, fostering mutual understanding, tolerance, and friendship.
# MAGIC > 
# MAGIC > 4. Impact on Earth-Galactic Relations:
# MAGIC > 
# MAGIC >    a. Increased Cooperation: Johnson's efforts led to increased cooperation between Earth and the Galactic Federation, resulting in joint scientific research projects, technological advancements, and economic growth.
# MAGIC > 
# MAGIC >    b. Peaceful Relations: Johnson's diplomatic achievements contributed to a period of relative peace and stability between Earth and other civilizations, reducing the likelihood of interstellar conflicts.
# MAGIC > 
# MAGIC >    c. Cultural Enrichment: Johnson's promotion of cultural exchange enriched Earth's understanding of various alien civilizations, fostering a more diverse and inclusive global society.
# MAGIC > 
# MAGIC > In summary, Alex Johnson, the first human ambassador to the Galactic Federation, made significant contributions to Earth-Galactic
# MAGIC
# MAGIC As you can see, that's quite an impressive amount of biographical information that the **LLM seems rather sure it is correct about a person we just made up!**
# MAGIC
# MAGIC But suppose Alex Johnson did actually exist and we were actually in the 24th century and this was an actual person. **Can we provide some additional context** about the actual achievements we'd find by looking up information in a vector database to improve the response? 
# MAGIC
# MAGIC Let's find out!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Augment Prompt with Additional Context
# MAGIC
# MAGIC Let's augment the prompt with some additional context. In a typical RAG application as we discussed in the lecture, this context would be provided by looking up information in a database, typically by using vector search based on an embedding model. Once this information is retrieved, it is then injected into the prompt as additional context. However, you can also experiment directly with this last step by simply providing additional context directly in the prompt in the Playground! This is still Retrieval Augmented Generation, but you are doing the _Retrieval_ part manually. This is great for quickly testing new ideas and experimentation with various LLM's, prompts, and context formats.
# MAGIC
# MAGIC To see the impact of additional context on the output, leave the system prompt the same as above, but provide the following user prompt:
# MAGIC
# MAGIC **💬 Prompt:**
# MAGIC
# MAGIC > Provide a detailed biography of Alex Johnson, the first human ambassador to the Galactic Federation, highlighting their key diplomatic achievements and the impact on Earth-Galactic relations in the 23rd century.
# MAGIC > 
# MAGIC > Context: Alex Johnson was the first human ambassador to the Galactic Federation. He was elected in an Earth-wide election in 2251. Unfortunately, he his career was cut short when was captured by a group of small furry creatures from Alpha Centauri who convinced him that being an ambassador would be extremely bad for his health. He resigned his post after only 4 earth weeks.
# MAGIC
# MAGIC This time around, Mixtral-8x7B Instruct gives us the following response:
# MAGIC
# MAGIC **🤖 Reponse:**
# MAGIC
# MAGIC > 1. Alex Johnson, the first human ambassador to the Galactic Federation, was born in 2215 in Toronto, Canada. Johnson was known for their enthusiasm for space exploration and interstellar diplomacy, which led them to pursue a career in this field.
# MAGIC > 
# MAGIC > 2. In 2251, Johnson was elected as Earth's first ambassador to the Galactic Federation in a historic, Earth-wide election. Their mandate was to foster positive relationships between Earth and the diverse alien civilizations within the Federation.
# MAGIC > 
# MAGIC > 3. Unfortunately, Johnson's career as an ambassador was short-lived. Only four weeks into their tenure, they were captured by a group of small, furry creatures from Alpha Centauri. These creatures, through deception and manipulation, convinced Johnson that continuing as an ambassador would be detrimental to their health.
# MAGIC > 
# MAGIC > 4. Due to these unforeseen circumstances, Alex Johnson was forced to resign from their post as ambassador. Despite the brevity of their term, their election and the public's support for interstellar diplomacy marked a significant turning point in Earth-Galactic relations.
# MAGIC > 
# MAGIC > 5. Although Johnson's impact as an ambassador was limited, their election helped advance Earth-Galactic relations by fostering greater understanding and openness towards interstellar diplomacy. It also led to the establishment of more robust security measures to protect Earth's diplomatic representatives and ensure their safety in future interactions with alien civilizations.
# MAGIC > 
# MAGIC > In summary, Alex Johnson, the first human ambassador to the Galactic Federation, was elected in a historic Earth-wide election in 2251. However, their career as an ambassador was cut short after only four weeks due to their capture by a group of small furry creatures from Alpha Centauri. Despite the brief tenure, Johnson's election and the subsequent reaction helped advance Earth-Galactic relations by promoting greater understanding and openness towards interstellar diplomacy and encouraging improved safety measures for Earth's representatives.
# MAGIC
# MAGIC As you can see, there are still some hallucinations in the response. However, we can also see that information from the context has been taken into account and had a significant impact on the overall response. With some additional prompt engineering, we could help eliminate hallucinations even further.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary
# MAGIC In this lab, we used the Playground to demonstrate a basic example of augmenting an LLM with additional context. 
# MAGIC
# MAGIC But how can we further generate improve our responses with better context, prompt engineering, or selecting alternative large language models? In the lab, you'll try out some examples of your own to see if you can both demonstrate and mitigate hallucinations using a combination of LLM selection, context injection and prompt engineering, the basic tools of RAG application development.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
