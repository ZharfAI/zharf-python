"""
The Python library for the Zharf platform (https://platform.zharf.ai).

>>> import zharf
>>> response = zharf.chat('deepseek-v4-flash', [{'role': 'user', 'content': 'Hello'}])  # doctest: +SKIP
>>> print(response.message.content, response.cost_zharf)  # doctest: +SKIP
"""

from zharf._client import AsyncClient, Client, __version__
from zharf._types import (
  ChatResponse,
  EmbedResponse,
  GenerateResponse,
  ImageResponse,
  ListResponse,
  Message,
  Model,
  Options,
  Pricing,
  RequestError,
  ResponseError,
  ShowResponse,
  SpeechResponse,
  SubscriptableBaseModel,
  Tool,
  TranscriptionResponse,
  Usage,
)

__all__ = [
  'AsyncClient',
  'ChatResponse',
  'Client',
  'EmbedResponse',
  'GenerateResponse',
  'ImageResponse',
  'ListResponse',
  'Message',
  'Model',
  'Options',
  'Pricing',
  'RequestError',
  'ResponseError',
  'ShowResponse',
  'SpeechResponse',
  'SubscriptableBaseModel',
  'Tool',
  'TranscriptionResponse',
  'Usage',
  '__version__',
  'audio',
  'chat',
  'embed',
  'generate',
  'images',
  'list',
  'show',
]

_client = Client()

chat = _client.chat
generate = _client.generate
embed = _client.embed
list = _client.list
show = _client.show
images = _client.images
audio = _client.audio
