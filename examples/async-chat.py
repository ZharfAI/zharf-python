import asyncio

from zharf import AsyncClient


async def main():
  messages = [
    {
      'role': 'user',
      'content': 'Why is the sky blue?',
    },
  ]

  client = AsyncClient()

  response = await client.chat('deepseek-v4-flash', messages=messages)
  print(response['message']['content'])

  # streaming: await the call, then iterate
  async for part in await client.chat('deepseek-v4-flash', messages=messages, stream=True):
    print(part.message.content, end='', flush=True)
  print()

  await client.close()


if __name__ == '__main__':
  asyncio.run(main())
