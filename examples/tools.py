from zharf import ChatResponse, chat

MODEL = 'deepseek-v4-flash'


def add_two_numbers(a: int, b: int) -> int:
  """
  Add two numbers

  Args:
    a (int): The first number
    b (int): The second number

  Returns:
    int: The sum of the two numbers
  """

  # The cast is necessary because returned tool call arguments don't always follow the schema exactly
  return int(a) + int(b)


def subtract_two_numbers(a: int, b: int) -> int:
  """
  Subtract two numbers
  """

  return int(a) - int(b)


# Tools can also be written out as JSON schema and passed into chat
subtract_two_numbers_tool = {
  'type': 'function',
  'function': {
    'name': 'subtract_two_numbers',
    'description': 'Subtract two numbers',
    'parameters': {
      'type': 'object',
      'required': ['a', 'b'],
      'properties': {
        'a': {'type': 'integer', 'description': 'The first number'},
        'b': {'type': 'integer', 'description': 'The second number'},
      },
    },
  },
}

messages = [{'role': 'user', 'content': 'What is three plus one?'}]
print('Prompt:', messages[0]['content'])

available_functions = {
  'add_two_numbers': add_two_numbers,
  'subtract_two_numbers': subtract_two_numbers,
}

response: ChatResponse = chat(
  MODEL,
  messages=messages,
  tools=[add_two_numbers, subtract_two_numbers_tool],
)

if response.message.tool_calls:
  # There may be several tool calls in the response.
  # Add the assistant message with its tool calls to the conversation
  messages.append(response.message)

  for tool in response.message.tool_calls:
    # Make sure the function exists, then call it
    if function_to_call := available_functions.get(tool.function.name):
      print('Calling function:', tool.function.name)
      print('Arguments:', tool.function.arguments)
      output = function_to_call(**tool.function.arguments)
      print('Function output:', output)
    else:
      print('Function', tool.function.name, 'not found')
      output = 'Function not found'

    # Add each tool result as its own message; tool_name links it to its call
    messages.append({'role': 'tool', 'content': str(output), 'tool_name': tool.function.name})

  # Get the final answer from the model, with all tool results
  final_response = chat(MODEL, messages=messages)
  print('Final response:', final_response.message.content)

else:
  print('No tool calls returned from model')
