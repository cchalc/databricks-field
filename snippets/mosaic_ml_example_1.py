from mcli.sdk import predict


user_prompt = "tell me about your day"

static_before = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction: 
As a medieval knight brought up in a fancy castle with a host of servants, 
"""

static_after = """
### Response:
"""

prompt = static_before + user_prompt + static_after

endpoint = 'https://models.hosted-on.mosaicml.hosting/mpt-30b-instruct/v1'
answer = predict(endpoint, {"inputs": [prompt], "parameters": {"temperature": 0}})
print(answer)
