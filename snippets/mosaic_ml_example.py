from mcli.sdk import predict

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction: write 3 reasons why you should train an AI model on domain specific data set.
### Response: """

endpoint = 'https://models.hosted-on.mosaicml.hosting/mpt-7b-instruct/v1'
answer = predict(endpoint, {"inputs": [prompt], "parameters": {"temperature": 0}})
print(answer)
