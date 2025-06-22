from transformers import pipeline
import torch


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# pipe = pipeline("summarization", model="facebook/bart-large-cnn")
model = pipeline("summarization", model="facebook/bart-large-cnn")
# model = pipeline("summarization", model="deepseek-ai/DeepSeek-R1-0528")
response = model("text to summerize")
print(response)


'''
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
'''