from openai import OpenAI
client = OpenAI(
    base_url='https://xiaoai.plus/v1',
    # sk-xxx替换为自己的key
    api_key='sk-jTsJsfqxQRUahM9sdEepWh3JluKmwh4gd4F3QBr4BVqgbnlQ'
    # base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    # api_key='sk-65ded426d36f42b4b0277408baad14c1'
)
completion = client.chat.completions.create(
  # model="qwen-plus-latest",
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，介绍一下你自己"}
  ]
)
print(completion.choices[0].message)