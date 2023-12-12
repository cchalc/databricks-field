from mcli.sdk import predict

prompt = """[INST]
<<SYS>>
You are an assistant that will receive questions or instructions from a user. The following is the current conversation between you and the user. Answer as concisely as possible.
<</SYS>>

I was thinking about visiting Copenhagen, what would be 3 good places to see?
[/INST]
"""

endpoint = 'https://models.hosted-on.mosaicml.hosting/llama2-70b-chat/v1'
answer = predict(endpoint, {"inputs": [prompt], "parameters": {"temperature":0,"max_new_tokens":500}})

print(answer)
