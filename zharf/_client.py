from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import urllib.parse
from email.utils import parsedate_to_datetime
from importlib import metadata
from typing import (
  Any,
  AsyncIterator,
  Callable,
  Dict,
  Iterator,
  List,
  Literal,
  Mapping,
  Optional,
  Sequence,
  Union,
  overload,
)

import httpx

from zharf._types import (
  ChatResponse,
  EmbedResponse,
  GenerateResponse,
  ImageResponse,
  ListResponse,
  Message,
  Options,
  RequestError,
  ResponseError,
  ShowResponse,
  SpeechResponse,
  SubscriptableBaseModel,
  Tool,
  TranscriptionResponse,
  Usage,
)
from zharf._utils import SSEParser, convert_function_to_tool, file_part, image_url

try:
  __version__ = metadata.version('zharf')
except metadata.PackageNotFoundError:
  __version__ = '0.0.0'

DEFAULT_HOST = 'https://platform.zharf.ai/v1'
DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
CONNECTION_ERROR_MESSAGE = 'Failed to connect to {host}. Check your network, and HTTPS_PROXY if you use a proxy.'

Messages = Optional[Sequence[Union[Mapping[str, Any], Message]]]
Tools = Optional[Sequence[Union[Mapping[str, Any], Tool, Callable[..., Any]]]]
Format = Optional[Union[Literal['', 'json'], Mapping[str, Any]]]
Think = Optional[Union[bool, Literal['low', 'medium', 'high'], str]]
OptionsArg = Optional[Union[Mapping[str, Any], Options]]


class BaseClient(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager):
  def __init__(
    self,
    client: Any,
    host: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    timeout: Any = DEFAULT_TIMEOUT,
    headers: Optional[Mapping[str, str]] = None,
    follow_redirects: bool = True,
    **kwargs: Any,
  ) -> None:
    """
    Creates an httpx client. `host` defaults to $ZHARF_BASE_URL, then
    https://platform.zharf.ai/v1. `api_key` defaults to $ZHARF_API_KEY, read on every
    request. Other keyword arguments (`proxy`, `verify`, `transport`...) go to httpx.
    """
    self._api_key = api_key
    merged = {
      'accept': 'application/json',
      'user-agent': f'zharf-python/{__version__} Python/{platform.python_version()}',
    }
    merged.update({k.lower(): v for k, v in (headers or {}).items() if v is not None})

    self._client = client(
      base_url=_parse_host(host or os.getenv('ZHARF_BASE_URL')),
      timeout=timeout,
      headers=merged,
      follow_redirects=follow_redirects,
      **kwargs,
    )

  def _auth(self) -> Dict[str, str]:
    if 'authorization' in self._client.headers:
      return {}
    key = self._api_key or os.getenv('ZHARF_API_KEY')
    return {'authorization': f'Bearer {key}'} if key else {}

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()


class Client(BaseClient):
  def __init__(self, host: Optional[str] = None, **kwargs: Any) -> None:
    super().__init__(httpx.Client, host, **kwargs)
    self.images = Images(self)
    'Image generation: `client.images.generate(...)`.'
    self.audio = Audio(self)
    'Speech and transcription: `client.audio.speech(...)`, `client.audio.transcribe(...)`.'

  def close(self) -> None:
    self._client.close()

  def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
    headers = {**self._auth(), **kwargs.pop('headers', {})}
    try:
      r = self._client.request(method, path, headers=headers, **kwargs)
    except httpx.ConnectError as e:
      raise ConnectionError(CONNECTION_ERROR_MESSAGE.format(host=self._client.base_url)) from e
    if r.is_error:
      raise _response_error(r)
    return r

  def _stream_chat(self, payload: Dict[str, Any]) -> Iterator[ChatResponse]:
    state = _ChatStream()
    headers = {**self._auth(), 'accept': 'text/event-stream'}
    try:
      with self._client.stream('POST', '/chat/completions', json=payload, headers=headers) as r:
        if r.is_error:
          r.read()
          raise _response_error(r)
        if _is_json(r):  # answered in one piece
          r.read()
          yield _parse_chat(r.json(), r.headers)
          return
        parser = SSEParser()
        for chunk in r.iter_bytes():
          for data in parser.feed(chunk):
            if state.feed(data):
              yield from state.pop()
        for data in parser.close():
          state.feed(data)
        yield from state.pop()
    except httpx.ConnectError as e:
      raise ConnectionError(CONNECTION_ERROR_MESSAGE.format(host=self._client.base_url)) from e
    yield state.finish()

  @overload
  def chat(
    self,
    model: str = '',
    messages: Messages = None,
    *,
    tools: Tools = None,
    stream: Literal[False] = False,
    think: Think = None,
    format: Format = None,
    options: OptionsArg = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> ChatResponse: ...

  @overload
  def chat(
    self,
    model: str = '',
    messages: Messages = None,
    *,
    tools: Tools = None,
    stream: Literal[True] = True,
    think: Think = None,
    format: Format = None,
    options: OptionsArg = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> Iterator[ChatResponse]: ...

  def chat(
    self,
    model: str = '',
    messages: Messages = None,
    *,
    tools: Tools = None,
    stream: bool = False,
    think: Think = None,
    format: Format = None,
    options: OptionsArg = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> Union[ChatResponse, Iterator[ChatResponse]]:
    """
    Create a chat response using the requested model.

    Args:
      tools:
        JSON schema dicts, `Tool` objects or Python functions. A function's type hints
        and Google-style docstring describe it to the model.
      stream: Return a generator of parts. The last part has `done=True` and `cost_zharf`.
      think: True or False to turn reasoning on or off, or an effort such as 'low', 'medium', 'high'.
      format: 'json' for any JSON object, or a JSON schema the answer must follow.
      options: temperature, top_p, top_k, num_predict, stop, seed and the like.
      extra_body: Other request fields, sent as they are (tool_choice, n, ...).

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ChatResponse` if `stream` is `False`, otherwise a `ChatResponse` generator.
    """
    payload = _chat_payload(model, messages, tools, stream, think, format, options, extra_body)
    if stream:
      return self._stream_chat(payload)
    r = self._request('POST', '/chat/completions', json=payload)
    return _parse_chat(r.json(), r.headers)

  @overload
  def generate(
    self,
    model: str = '',
    prompt: str = '',
    system: Optional[str] = None,
    *,
    images: Optional[Sequence[Any]] = None,
    stream: Literal[False] = False,
    think: Think = None,
    format: Format = None,
    options: OptionsArg = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> GenerateResponse: ...

  @overload
  def generate(
    self,
    model: str = '',
    prompt: str = '',
    system: Optional[str] = None,
    *,
    images: Optional[Sequence[Any]] = None,
    stream: Literal[True] = True,
    think: Think = None,
    format: Format = None,
    options: OptionsArg = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> Iterator[GenerateResponse]: ...

  def generate(
    self,
    model: str = '',
    prompt: str = '',
    system: Optional[str] = None,
    *,
    images: Optional[Sequence[Any]] = None,
    stream: bool = False,
    think: Think = None,
    format: Format = None,
    options: OptionsArg = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
    """
    Answer a single prompt, with an optional system prompt. Runs on the chat endpoint.

    Returns `GenerateResponse` if `stream` is `False`, otherwise a `GenerateResponse` generator.
    """
    messages = _generate_messages(prompt, system, images)
    if stream:
      parts = self.chat(model, messages, stream=True, think=think, format=format, options=options, extra_body=extra_body)
      return (_to_generate(part) for part in parts)
    return _to_generate(self.chat(model, messages, think=think, format=format, options=options, extra_body=extra_body))

  def list(self) -> ListResponse:
    """
    Every model on the platform, with prices in ZHARF. Needs no API key.
    """
    return _parse_list(self._request('GET', '/models').json())

  def show(self, model: str) -> ShowResponse:
    """
    One model's details and prices. Needs no API key.
    """
    _require_model(model)
    return ShowResponse(self._request('GET', f'/models/{urllib.parse.quote(model, safe="")}').json())

  def embed(
    self,
    model: str = '',
    input: Union[str, Sequence[str]] = '',
    *,
    dimensions: Optional[int] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> EmbedResponse:
    """
    Embed one text or a list of texts. `embeddings` keeps the input order.
    """
    r = self._request('POST', '/embeddings', json=_embed_payload(model, input, dimensions, extra_body))
    return _parse_embed(r.json(), r.headers)


class Images:
  def __init__(self, client: Client) -> None:
    self._client = client

  def generate(
    self,
    model: str = '',
    prompt: str = '',
    *,
    n: Optional[int] = None,
    size: Optional[str] = None,
    quality: Optional[str] = None,
    response_format: Optional[Literal['url', 'b64_json']] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> ImageResponse:
    """
    Generate images from a prompt. Each item in `data` has a `url` or `b64_json`.
    """
    payload = _image_payload(model, prompt, n, size, quality, response_format, extra_body)
    r = self._client._request('POST', '/images/generations', json=payload)
    return _parse_images(r.json(), r.headers)


class Audio:
  def __init__(self, client: Client) -> None:
    self._client = client

  def speech(
    self,
    model: str = '',
    input: str = '',
    *,
    voice: Optional[str] = None,
    response_format: Optional[str] = None,
    speed: Optional[float] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> SpeechResponse:
    """
    Turn text into speech. `bytes(response)` is the audio; `response.save(path)` writes it.
    """
    payload = _speech_payload(model, input, voice, response_format, speed, extra_body)
    r = self._client._request('POST', '/audio/speech', json=payload, headers={'accept': '*/*'})
    return _parse_speech(r)

  def transcribe(
    self,
    model: str = '',
    file: Any = None,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: Optional[str] = None,
    temperature: Optional[float] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> TranscriptionResponse:
    """
    Turn speech into text. `file` is a path, bytes, an open file or a (filename, bytes) pair.
    """
    data, files = _transcription_form(model, file, language, prompt, response_format, temperature, extra_body)
    r = self._client._request('POST', '/audio/transcriptions', data=data, files=files)
    return _parse_transcription(r)


class AsyncClient(BaseClient):
  def __init__(self, host: Optional[str] = None, **kwargs: Any) -> None:
    super().__init__(httpx.AsyncClient, host, **kwargs)
    self.images = AsyncImages(self)
    'Image generation: `await client.images.generate(...)`.'
    self.audio = AsyncAudio(self)
    'Speech and transcription: `await client.audio.speech(...)`, `await client.audio.transcribe(...)`.'

  async def close(self) -> None:
    await self._client.aclose()

  async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
    headers = {**self._auth(), **kwargs.pop('headers', {})}
    try:
      r = await self._client.request(method, path, headers=headers, **kwargs)
    except httpx.ConnectError as e:
      raise ConnectionError(CONNECTION_ERROR_MESSAGE.format(host=self._client.base_url)) from e
    if r.is_error:
      raise _response_error(r)
    return r

  async def _stream_chat(self, payload: Dict[str, Any]) -> AsyncIterator[ChatResponse]:
    state = _ChatStream()
    headers = {**self._auth(), 'accept': 'text/event-stream'}
    try:
      async with self._client.stream('POST', '/chat/completions', json=payload, headers=headers) as r:
        if r.is_error:
          await r.aread()
          raise _response_error(r)
        if _is_json(r):
          await r.aread()
          yield _parse_chat(r.json(), r.headers)
          return
        parser = SSEParser()
        async for chunk in r.aiter_bytes():
          for data in parser.feed(chunk):
            if state.feed(data):
              for part in state.pop():
                yield part
        for data in parser.close():
          state.feed(data)
        for part in state.pop():
          yield part
    except httpx.ConnectError as e:
      raise ConnectionError(CONNECTION_ERROR_MESSAGE.format(host=self._client.base_url)) from e
    yield state.finish()

  @overload
  async def chat(
    self,
    model: str = '',
    messages: Messages = None,
    *,
    tools: Tools = None,
    stream: Literal[False] = False,
    think: Think = None,
    format: Format = None,
    options: OptionsArg = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> ChatResponse: ...

  @overload
  async def chat(
    self,
    model: str = '',
    messages: Messages = None,
    *,
    tools: Tools = None,
    stream: Literal[True] = True,
    think: Think = None,
    format: Format = None,
    options: OptionsArg = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> AsyncIterator[ChatResponse]: ...

  async def chat(
    self,
    model: str = '',
    messages: Messages = None,
    *,
    tools: Tools = None,
    stream: bool = False,
    think: Think = None,
    format: Format = None,
    options: OptionsArg = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
    """
    Create a chat response using the requested model. See `Client.chat`.

    Returns `ChatResponse` if `stream` is `False`, otherwise an asynchronous `ChatResponse` generator.
    """
    payload = _chat_payload(model, messages, tools, stream, think, format, options, extra_body)
    if stream:
      return self._stream_chat(payload)
    r = await self._request('POST', '/chat/completions', json=payload)
    return _parse_chat(r.json(), r.headers)

  @overload
  async def generate(
    self,
    model: str = '',
    prompt: str = '',
    system: Optional[str] = None,
    *,
    images: Optional[Sequence[Any]] = None,
    stream: Literal[False] = False,
    think: Think = None,
    format: Format = None,
    options: OptionsArg = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> GenerateResponse: ...

  @overload
  async def generate(
    self,
    model: str = '',
    prompt: str = '',
    system: Optional[str] = None,
    *,
    images: Optional[Sequence[Any]] = None,
    stream: Literal[True] = True,
    think: Think = None,
    format: Format = None,
    options: OptionsArg = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> AsyncIterator[GenerateResponse]: ...

  async def generate(
    self,
    model: str = '',
    prompt: str = '',
    system: Optional[str] = None,
    *,
    images: Optional[Sequence[Any]] = None,
    stream: bool = False,
    think: Think = None,
    format: Format = None,
    options: OptionsArg = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
    """
    Answer a single prompt, with an optional system prompt. See `Client.generate`.
    """
    messages = _generate_messages(prompt, system, images)
    if stream:
      parts = await self.chat(model, messages, stream=True, think=think, format=format, options=options, extra_body=extra_body)

      async def inner() -> AsyncIterator[GenerateResponse]:
        async for part in parts:
          yield _to_generate(part)

      return inner()
    return _to_generate(await self.chat(model, messages, think=think, format=format, options=options, extra_body=extra_body))

  async def list(self) -> ListResponse:
    """
    Every model on the platform, with prices in ZHARF. Needs no API key.
    """
    return _parse_list((await self._request('GET', '/models')).json())

  async def show(self, model: str) -> ShowResponse:
    """
    One model's details and prices. Needs no API key.
    """
    _require_model(model)
    r = await self._request('GET', f'/models/{urllib.parse.quote(model, safe="")}')
    return ShowResponse(r.json())

  async def embed(
    self,
    model: str = '',
    input: Union[str, Sequence[str]] = '',
    *,
    dimensions: Optional[int] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> EmbedResponse:
    """
    Embed one text or a list of texts. `embeddings` keeps the input order.
    """
    r = await self._request('POST', '/embeddings', json=_embed_payload(model, input, dimensions, extra_body))
    return _parse_embed(r.json(), r.headers)


class AsyncImages:
  def __init__(self, client: AsyncClient) -> None:
    self._client = client

  async def generate(
    self,
    model: str = '',
    prompt: str = '',
    *,
    n: Optional[int] = None,
    size: Optional[str] = None,
    quality: Optional[str] = None,
    response_format: Optional[Literal['url', 'b64_json']] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> ImageResponse:
    """
    Generate images from a prompt. See `Images.generate`.
    """
    payload = _image_payload(model, prompt, n, size, quality, response_format, extra_body)
    r = await self._client._request('POST', '/images/generations', json=payload)
    return _parse_images(r.json(), r.headers)


class AsyncAudio:
  def __init__(self, client: AsyncClient) -> None:
    self._client = client

  async def speech(
    self,
    model: str = '',
    input: str = '',
    *,
    voice: Optional[str] = None,
    response_format: Optional[str] = None,
    speed: Optional[float] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> SpeechResponse:
    """
    Turn text into speech. See `Audio.speech`.
    """
    payload = _speech_payload(model, input, voice, response_format, speed, extra_body)
    r = await self._client._request('POST', '/audio/speech', json=payload, headers={'accept': '*/*'})
    return _parse_speech(r)

  async def transcribe(
    self,
    model: str = '',
    file: Any = None,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: Optional[str] = None,
    temperature: Optional[float] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
  ) -> TranscriptionResponse:
    """
    Turn speech into text. See `Audio.transcribe`.
    """
    data, files = _transcription_form(model, file, language, prompt, response_format, temperature, extra_body)
    r = await self._client._request('POST', '/audio/transcriptions', data=data, files=files)
    return _parse_transcription(r)


# Requests

# ollama options that only make sense for a model running on your own machine.
_LOCAL_ONLY_OPTIONS = frozenset(
  {
    'numa',
    'num_ctx',
    'num_batch',
    'num_gpu',
    'main_gpu',
    'low_vram',
    'f16_kv',
    'logits_all',
    'vocab_only',
    'use_mmap',
    'use_mlock',
    'embedding_only',
    'num_thread',
    'num_keep',
    'tfs_z',
    'typical_p',
    'repeat_last_n',
    'mirostat',
    'mirostat_tau',
    'mirostat_eta',
    'penalize_newline',
  }
)
_OPTION_NAMES = {'num_predict': 'max_tokens', 'repeat_penalty': 'repetition_penalty'}


def _map_options(options: OptionsArg) -> Dict[str, Any]:
  """
  ollama-style options as request fields.

  >>> _map_options({'temperature': 0.2, 'num_predict': 100, 'num_ctx': 8192, 'stop': ['\\n']})
  {'temperature': 0.2, 'max_tokens': 100, 'stop': ['\\n']}
  """
  if not options:
    return {}
  items = options.to_dict() if isinstance(options, SubscriptableBaseModel) else dict(options)
  out: Dict[str, Any] = {}
  for key, value in items.items():
    if value is None or key in _LOCAL_ONLY_OPTIONS:
      continue
    if key == 'num_predict' and isinstance(value, int) and value < 0:  # ollama's "no limit"
      continue
    out[_OPTION_NAMES.get(key, key)] = list(value) if isinstance(value, tuple) else value
  return out


def _think_fields(think: Think) -> Dict[str, Any]:
  """
  The `think` argument as request fields.

  >>> _think_fields(True)
  {'reasoning': {'enabled': True}}
  >>> _think_fields('high')
  {'reasoning_effort': 'high'}
  >>> _think_fields(None)
  {}
  """
  if think is None:
    return {}
  if isinstance(think, bool):
    return {'reasoning': {'enabled': think}}
  if isinstance(think, str):
    return {'reasoning_effort': think}
  raise RequestError(f"think must be True, False or an effort such as 'low', 'medium', 'high', not {think!r}")


def _format_fields(format: Format) -> Dict[str, Any]:
  """
  The `format` argument as a `response_format` field.

  >>> _format_fields('json')
  {'response_format': {'type': 'json_object'}}
  >>> _format_fields({'title': 'Answer', 'type': 'object', 'properties': {}})['response_format']['json_schema']['name']
  'Answer'
  """
  if format is None or format == '':
    return {}
  if format == 'json':
    return {'response_format': {'type': 'json_object'}}
  if hasattr(format, 'model_json_schema'):  # a pydantic model class
    format = format.model_json_schema()  # type: ignore[union-attr]
  if isinstance(format, Mapping):
    if format.get('type') in ('json_object', 'json_schema', 'text'):
      return {'response_format': dict(format)}
    name = re.sub(r'[^A-Za-z0-9_-]', '_', str(format.get('title') or 'response'))[:64]
    return {'response_format': {'type': 'json_schema', 'json_schema': {'name': name, 'schema': dict(format)}}}
  raise RequestError(f"format must be 'json' or a JSON schema, not {format!r}")


def _copy_tools(tools: Tools) -> Optional[List[Dict[str, Any]]]:
  out = []
  for tool in tools or []:
    if isinstance(tool, SubscriptableBaseModel):
      item = tool.to_dict()
    elif isinstance(tool, Mapping):
      item = _plain_dict(tool)
    elif callable(tool):
      item = convert_function_to_tool(tool).to_dict()
    else:
      raise RequestError(f'tools must be dicts, Tool objects or functions, not {type(tool).__name__}')
    item.setdefault('type', 'function')
    out.append(item)
  return out or None


def _plain_dict(value: Any) -> Dict[str, Any]:
  if isinstance(value, SubscriptableBaseModel):
    return value.to_dict()
  return {k: (v.to_dict() if isinstance(v, SubscriptableBaseModel) else v) for k, v in dict(value).items()}


def _copy_messages(messages: Messages) -> List[Dict[str, Any]]:
  """
  Messages in the API's shape. Accepts what ollama code sends: a response message
  with tool calls appended as it is, then tool results that name the tool.
  """
  out: List[Dict[str, Any]] = []
  pending: List[Dict[str, Any]] = []  # tool calls of the last assistant message without a result yet

  for position, raw in enumerate(messages or []):
    if isinstance(raw, (Mapping, SubscriptableBaseModel)):
      message = {k: v for k, v in _plain_dict(raw).items() if v is not None}
    else:
      raise RequestError(f'messages must be dicts or Message objects, not {type(raw).__name__}')

    message.pop('thinking', None)
    images = message.pop('images', None)
    if images:
      message['content'] = _with_images(message.get('content'), images)

    if message.get('tool_calls'):
      calls = []
      pending = []
      for index, call in enumerate(message['tool_calls']):
        call = _plain_dict(call)
        function = _plain_dict(call.get('function') or {})
        arguments = function.get('arguments', {})
        if not isinstance(arguments, str):
          arguments = json.dumps(arguments, ensure_ascii=False)
        function['arguments'] = arguments
        call_id = call.get('id') or f'call_{position}_{index}'
        calls.append({'id': call_id, 'type': call.get('type') or 'function', 'function': function})
        pending.append({'id': call_id, 'name': function.get('name')})
      message['tool_calls'] = calls

    if message.get('role') == 'tool':
      name = message.pop('tool_name', None)
      if not message.get('tool_call_id') and pending:
        match = next((p for p in pending if name and p['name'] == name), pending[0])
        message['tool_call_id'] = match['id']
        pending.remove(match)
      content = message.get('content')
      is_parts = isinstance(content, list) and all(isinstance(p, Mapping) and 'type' in p for p in content)
      if content is not None and not isinstance(content, str) and not is_parts:
        if isinstance(content, (Mapping, list, tuple)):
          message['content'] = json.dumps(content, ensure_ascii=False)
        else:
          message['content'] = str(content)
    else:
      message.pop('tool_name', None)

    out.append(message)
  return out


def _with_images(content: Any, images: Sequence[Any]) -> List[Dict[str, Any]]:
  parts = list(content) if isinstance(content, list) else [{'type': 'text', 'text': content}] if content else []
  parts.extend({'type': 'image_url', 'image_url': {'url': image_url(image)}} for image in images)
  return parts


def _require_model(model: str) -> None:
  if not model:
    raise RequestError('must provide a model')


def _chat_payload(
  model: str,
  messages: Messages,
  tools: Tools,
  stream: bool,
  think: Think,
  format: Format,
  options: OptionsArg,
  extra_body: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
  _require_model(model)
  payload: Dict[str, Any] = {'model': model, 'messages': _copy_messages(messages)}
  payload.update(_map_options(options))
  payload.update(_think_fields(think))
  payload.update(_format_fields(format))
  copied_tools = _copy_tools(tools)
  if copied_tools:
    payload['tools'] = copied_tools
  if stream:
    payload['stream'] = True
    payload['stream_options'] = {'include_usage': True}
  payload.update(extra_body or {})
  return payload


def _generate_messages(prompt: str, system: Optional[str], images: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
  messages: List[Dict[str, Any]] = []
  if system:
    messages.append({'role': 'system', 'content': system})
  user: Dict[str, Any] = {'role': 'user', 'content': prompt}
  if images:
    user['images'] = list(images)
  messages.append(user)
  return messages


def _embed_payload(model: str, input: Any, dimensions: Optional[int], extra_body: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
  _require_model(model)
  payload: Dict[str, Any] = {'model': model, 'input': input if isinstance(input, str) else [str(i) for i in input]}
  if dimensions is not None:
    payload['dimensions'] = dimensions
  payload.update(extra_body or {})
  return payload


def _image_payload(model, prompt, n, size, quality, response_format, extra_body) -> Dict[str, Any]:
  _require_model(model)
  payload = {'model': model, 'prompt': prompt, 'n': n, 'size': size, 'quality': quality, 'response_format': response_format}
  payload = {k: v for k, v in payload.items() if v is not None}
  payload.update(extra_body or {})
  return payload


def _speech_payload(model, input, voice, response_format, speed, extra_body) -> Dict[str, Any]:
  _require_model(model)
  payload = {'model': model, 'input': input, 'voice': voice, 'response_format': response_format, 'speed': speed}
  payload = {k: v for k, v in payload.items() if v is not None}
  payload.update(extra_body or {})
  return payload


def _transcription_form(model, file, language, prompt, response_format, temperature, extra_body):
  _require_model(model)
  if file is None:
    raise RequestError('must provide a file')
  fields = {'model': model, 'language': language, 'prompt': prompt, 'response_format': response_format, 'temperature': temperature}
  fields.update(extra_body or {})
  data = {k: v if isinstance(v, str) else json.dumps(v) for k, v in fields.items() if v is not None}
  return data, {'file': file_part(file)}


# Responses


def _is_json(r: httpx.Response) -> bool:
  return 'json' in r.headers.get('content-type', '')


def _float(value: Any) -> Optional[float]:
  try:
    return float(value) if value is not None and value != '' else None
  except (TypeError, ValueError):
    return None


def _cost(usage: Optional[Mapping[str, Any]], headers: Optional[httpx.Headers]) -> Optional[float]:
  cost = _float((usage or {}).get('cost_zharf'))
  if cost is None and headers is not None:
    cost = _float(headers.get('x-zharf-cost'))
  return cost


def _text(content: Any) -> str:
  if content is None:
    return ''
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    return ''.join(p.get('text', '') for p in content if isinstance(p, Mapping) and isinstance(p.get('text'), str))
  return str(content)


def _thinking(message: Mapping[str, Any]) -> Optional[str]:
  for key in ('reasoning_content', 'reasoning', 'thinking'):
    value = message.get(key)
    if isinstance(value, str) and value:
      return value
  return None


def _usage_fields(usage: Optional[Mapping[str, Any]], headers: Optional[httpx.Headers] = None) -> Dict[str, Any]:
  usage = usage or {}
  return {
    'usage': Usage(usage) if usage else None,
    'cost_zharf': _cost(usage, headers),
    'prompt_eval_count': usage.get('prompt_tokens'),
    'eval_count': usage.get('completion_tokens'),
  }


def _parse_chat(data: Mapping[str, Any], headers: Optional[httpx.Headers] = None) -> ChatResponse:
  choice = (data.get('choices') or [{}])[0]
  raw = choice.get('message') or {}
  message = Message(
    role=raw.get('role') or 'assistant',
    content=_text(raw.get('content')),
    thinking=_thinking(raw),
    tool_calls=raw.get('tool_calls') or None,
  )
  return ChatResponse(
    id=data.get('id'),
    model=data.get('model'),
    created=data.get('created'),
    message=message,
    done=True,
    done_reason=choice.get('finish_reason'),
    **_usage_fields(data.get('usage'), headers),
  )


class _ChatStream:
  """Turns stream chunks into ollama-style parts; tool call pieces are joined for the final part."""

  def __init__(self) -> None:
    self.id: Optional[str] = None
    self.model: Optional[str] = None
    self.created: Optional[int] = None
    self.finish_reason: Optional[str] = None
    self.usage: Optional[Mapping[str, Any]] = None
    self.tool_calls: Dict[int, Dict[str, Any]] = {}
    self.parts: List[ChatResponse] = []
    self.ended = False

  def feed(self, data: str) -> bool:
    """Take one event's data; True when parts are waiting in `pop()`."""
    if self.ended:
      return False
    if data.strip() == '[DONE]':
      self.ended = True
      return False
    try:
      chunk = json.loads(data)
    except ValueError:
      raise ResponseError(f'Unreadable stream data: {data[:200]}') from None
    if not isinstance(chunk, Mapping):
      return False
    if chunk.get('error'):
      raise _payload_error(chunk['error'], -1)

    self.id = chunk.get('id') or self.id
    self.model = chunk.get('model') or self.model
    self.created = chunk.get('created') or self.created
    if chunk.get('usage'):
      self.usage = chunk['usage']

    for choice in chunk.get('choices') or []:
      if choice.get('index', 0) != 0:
        continue
      if choice.get('finish_reason'):
        self.finish_reason = choice['finish_reason']
      delta = choice.get('delta') or choice.get('message') or {}
      for position, call in enumerate(delta.get('tool_calls') or []):
        self._add_tool_call(position, call)
      content = _text(delta.get('content'))
      thinking = _thinking(delta)
      if content or thinking:
        self.parts.append(
          ChatResponse(
            id=self.id,
            model=self.model,
            created=self.created,
            message=Message(role='assistant', content=content, thinking=thinking),
            done=False,
          )
        )
    return bool(self.parts)

  def _add_tool_call(self, position: int, call: Mapping[str, Any]) -> None:
    index = call.get('index', position)
    entry = self.tool_calls.setdefault(index, {'id': None, 'type': 'function', 'name': '', 'arguments': ''})
    if call.get('id'):
      entry['id'] = call['id']
    function = call.get('function') or {}
    if function.get('name'):
      entry['name'] += function['name']
    arguments = function.get('arguments')
    if isinstance(arguments, str):
      entry['arguments'] += arguments
    elif arguments is not None:
      entry['arguments'] = json.dumps(arguments, ensure_ascii=False)

  def pop(self) -> List[ChatResponse]:
    parts, self.parts = self.parts, []
    return parts

  def finish(self) -> ChatResponse:
    tool_calls = [
      {'id': e['id'], 'type': e['type'], 'function': {'name': e['name'], 'arguments': e['arguments']}}
      for _, e in sorted(self.tool_calls.items())
    ]
    return ChatResponse(
      id=self.id,
      model=self.model,
      created=self.created,
      message=Message(role='assistant', content='', tool_calls=tool_calls or None),
      done=True,
      done_reason=self.finish_reason,
      **_usage_fields(self.usage),
    )


def _to_generate(part: ChatResponse) -> GenerateResponse:
  fields = {k: v for k, v in part.to_dict(exclude_none=False).items() if k != 'message'}
  fields['usage'] = part.usage
  return GenerateResponse(fields, response=part.message.content or '', thinking=part.message.thinking)


def _parse_list(data: Mapping[str, Any]) -> ListResponse:
  fields = {k: v for k, v in data.items() if k not in ('data', 'object')}
  return ListResponse(fields, models=data.get('data') or [])


def _parse_embed(data: Mapping[str, Any], headers: httpx.Headers) -> EmbedResponse:
  items = sorted(data.get('data') or [], key=lambda item: item.get('index', 0))
  fields = _usage_fields(data.get('usage'), headers)
  fields.pop('eval_count')
  return EmbedResponse(model=data.get('model'), embeddings=[item.get('embedding') for item in items], **fields)


def _parse_images(data: Mapping[str, Any], headers: httpx.Headers) -> ImageResponse:
  fields = {k: v for k, v in data.items() if k != 'usage'}
  return ImageResponse(fields, usage=data.get('usage'), cost_zharf=_cost(data.get('usage'), headers))


def _parse_speech(r: httpx.Response) -> SpeechResponse:
  return SpeechResponse(content=r.content, content_type=r.headers.get('content-type'), cost_zharf=_cost(None, r.headers))


def _parse_transcription(r: httpx.Response) -> TranscriptionResponse:
  if _is_json(r):
    data = r.json()
    return TranscriptionResponse(data, cost_zharf=_cost(data.get('usage'), r.headers))
  return TranscriptionResponse(text=r.text, cost_zharf=_cost(None, r.headers))


# Errors


def _retry_after(value: Optional[str]) -> Optional[float]:
  """
  >>> _retry_after('7')
  7.0
  >>> _retry_after(None) is None
  True
  """
  if not value:
    return None
  seconds = _float(value)
  if seconds is not None:
    return max(seconds, 0.0)
  try:
    import datetime

    when = parsedate_to_datetime(value)
    return max((when - datetime.datetime.now(when.tzinfo)).total_seconds(), 0.0)
  except (TypeError, ValueError, IndexError):
    return None


def _payload_error(error: Any, status_code: int, retry_after: Optional[float] = None) -> ResponseError:
  if isinstance(error, Mapping):
    return ResponseError(
      str(error.get('message') or error.get('code') or 'Unknown error'),
      status_code,
      message_fa=error.get('message_fa'),
      code=error.get('code'),
      type=error.get('type'),
      retry_after=retry_after,
    )
  return ResponseError(str(error), status_code, retry_after=retry_after)


def _response_error(r: httpx.Response) -> ResponseError:
  retry_after = _retry_after(r.headers.get('retry-after'))
  try:
    body = r.json()
  except ValueError:
    body = None
  if isinstance(body, Mapping) and body.get('error'):
    return _payload_error(body['error'], r.status_code, retry_after)
  return ResponseError(r.text.strip() or r.reason_phrase or 'Unknown error', r.status_code, retry_after=retry_after)


# Host


def _parse_host(host: Optional[str]) -> str:
  """
  >>> _parse_host(None)
  'https://platform.zharf.ai/v1'
  >>> _parse_host('https://platform.zharf.ai')
  'https://platform.zharf.ai/v1'
  >>> _parse_host('platform.zharf.ai/v1/')
  'https://platform.zharf.ai/v1'
  >>> _parse_host('http://localhost:8080')
  'http://localhost:8080/v1'
  >>> _parse_host('https://example.com/api/v1')
  'https://example.com/api/v1'
  """
  host = (host or '').strip() or DEFAULT_HOST
  if '://' not in host:
    host = f'https://{host}'
  split = urllib.parse.urlsplit(host)
  path = split.path.rstrip('/') or '/v1'
  return urllib.parse.urlunsplit((split.scheme, split.netloc, path, '', ''))
