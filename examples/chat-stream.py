from zharf import chat

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in chat('deepseek-v4-flash', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)

  # the last part carries the totals
  if part.done:
    print(f'\n\nCost: {part.cost_zharf} ZHARF, {part.eval_count} tokens generated')
