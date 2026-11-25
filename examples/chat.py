from zharf import ChatResponse, chat

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

response: ChatResponse = chat('deepseek-v4-flash', messages=messages)
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)

print(f'\nCost: {response.cost_zharf} ZHARF')
