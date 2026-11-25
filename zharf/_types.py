"""Response and request types.

Every object reads both ways, like the ollama library: ``response.message.content``
and ``response['message']['content']`` return the same value. Fields the platform
adds later stay reachable the same way, so nothing it sends is lost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, KeysView, List, Mapping, Optional, Sequence, Tuple, Union


class SubscriptableBaseModel:
  """
  A small dict-backed object with attribute and item access.

  >>> msg = Message(role='user', content='سلام')
  >>> msg.content
  'سلام'
  >>> msg['role']
  'user'
  >>> msg.thinking is None
  True
  >>> 'thinking' in msg
  False
  >>> msg.get('thinking', 'none yet')
  'none yet'
  >>> msg['nonexistent']
  Traceback (most recent call last):
  KeyError: 'nonexistent'
  """

  _fields: ClassVar[Tuple[str, ...]] = ()
  'Known field names. Reading one that the response did not include returns None.'

  _nested: ClassVar[Dict[str, Any]] = {}
  'Field name -> type (or [type] for a list) used to wrap nested dicts.'

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    names: List[str] = []
    nested: Dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
      for name in vars(klass).get('__annotations__', {}):
        if not name.startswith('_') and name not in names:
          names.append(name)
      nested.update(vars(klass).get('_nested', {}))
    cls._fields = tuple(names)
    cls._nested = nested

  def __init__(self, *args: Union[Mapping[str, Any], SubscriptableBaseModel], **kwargs: Any) -> None:
    if len(args) > 1:
      raise TypeError(f'{type(self).__name__} takes at most one positional argument')
    object.__setattr__(self, '_data', {})
    values: Dict[str, Any] = {}
    if args and args[0] is not None:
      source = args[0]
      values.update(source._data if isinstance(source, SubscriptableBaseModel) else source)
    values.update(kwargs)
    for key, value in values.items():
      self[key] = value

  # attribute access

  def __getattr__(self, name: str) -> Any:
    if name.startswith('__') or name == '_data':
      raise AttributeError(name)
    data = self.__dict__.get('_data', {})
    if name in data:
      return data[name]
    if name in type(self)._fields:
      return None
    raise AttributeError(f'{type(self).__name__!r} object has no attribute {name!r}')

  def __setattr__(self, name: str, value: Any) -> None:
    if name.startswith('_'):
      object.__setattr__(self, name, value)
    else:
      self[name] = value

  # item access

  def __getitem__(self, key: str) -> Any:
    if key in self._data:
      return self._data[key]
    if key in self._fields:
      return None
    if isinstance(getattr(type(self), key, None), property):
      return getattr(self, key)
    raise KeyError(key)

  def __setitem__(self, key: str, value: Any) -> None:
    self._data[key] = self._coerce(key, value)

  def __delitem__(self, key: str) -> None:
    del self._data[key]

  def __contains__(self, key: object) -> bool:
    return self._data.get(key) is not None  # type: ignore[call-overload]

  def __iter__(self) -> Iterator[str]:
    return iter(self._data)

  def keys(self) -> KeysView[str]:
    return self._data.keys()

  def get(self, key: str, default: Any = None) -> Any:
    value = self._data.get(key)
    return default if value is None else value

  # conversion

  def to_dict(self, *, exclude_none: bool = True) -> Dict[str, Any]:
    """The object as plain dicts and lists, ready for `json.dumps`."""
    return {k: _plain(v) for k, v in self._data.items() if not (exclude_none and v is None)}

  model_dump = to_dict
  'Alias of `to_dict`, for code written against pydantic models.'

  def _coerce(self, key: str, value: Any) -> Any:
    spec = self._nested.get(key)
    if spec is None or value is None:
      return value
    if isinstance(spec, list):
      return [spec[0]._wrap(item) for item in value]
    return spec._wrap(value)

  @classmethod
  def _wrap(cls, value: Any) -> Any:
    if isinstance(value, cls):
      return value
    if isinstance(value, (Mapping, SubscriptableBaseModel)):
      return cls(value)
    return value

  def __eq__(self, other: object) -> bool:
    if isinstance(other, SubscriptableBaseModel):
      return type(self) is type(other) and self.to_dict() == other.to_dict()
    if isinstance(other, Mapping):
      return self.to_dict() == _plain(other)
    return NotImplemented

  __hash__ = None  # type: ignore[assignment]

  def __repr__(self) -> str:
    fields = ', '.join(f'{k}={v!r}' for k, v in self._data.items() if v is not None)
    return f'{type(self).__name__}({fields})'


def _plain(value: Any) -> Any:
  if isinstance(value, SubscriptableBaseModel):
    return value.to_dict()
  if isinstance(value, (list, tuple)):
    return [_plain(v) for v in value]
  if isinstance(value, Mapping):
    return {k: _plain(v) for k, v in value.items()}
  return value


class Options(SubscriptableBaseModel):
  """
  Sampling options, named as in ollama. `num_predict` is sent as `max_tokens` and
  `repeat_penalty` as `repetition_penalty`; any other key is sent as it is.
  """

  temperature: Optional[float]
  top_p: Optional[float]
  top_k: Optional[int]
  min_p: Optional[float]
  num_predict: Optional[int]
  stop: Optional[Union[str, Sequence[str]]]
  seed: Optional[int]
  presence_penalty: Optional[float]
  frequency_penalty: Optional[float]
  repeat_penalty: Optional[float]


class Message(SubscriptableBaseModel):
  """
  Chat message.
  """

  class ToolCall(SubscriptableBaseModel):
    """
    A tool call the model asked for.
    """

    class Function(SubscriptableBaseModel):
      """
      Tool call function. `arguments` is a dict (the API sends a JSON string; it is parsed for you).

      >>> Message.ToolCall.Function(name='add', arguments='{"a": 1, "b": 2}').arguments
      {'a': 1, 'b': 2}
      """

      name: str
      'Name of the function.'

      arguments: Mapping[str, Any]
      'Arguments of the function.'

      def _coerce(self, key: str, value: Any) -> Any:
        if key == 'arguments' and isinstance(value, str):
          if not value.strip():
            return {}
          try:
            return json.loads(value)
          except ValueError:
            return value
        return super()._coerce(key, value)

    id: Optional[str]
    'Id of the call. Tool results answer it with `tool_call_id`.'

    type: Optional[str]
    'Always "function".'

    function: Function
    'Function to be called.'

    _nested = {'function': Function}

  role: str
  "Role of the message: 'system', 'user', 'assistant' or 'tool'."

  content: Optional[str]
  'Text of the message. When streaming, each part carries the next piece.'

  thinking: Optional[str]
  'Reasoning text, when the model returns it.'

  images: Optional[Sequence[Any]]
  'Images for vision models: file paths, bytes, base64 strings or URLs.'

  tool_calls: Optional[Sequence[ToolCall]]
  'Tools calls the model asked for.'

  tool_name: Optional[str]
  'Name of the executed tool, on a tool result message.'

  tool_call_id: Optional[str]
  'Id of the call a tool result answers. Filled in for you from `tool_name` when missing.'

  _nested = {'tool_calls': [ToolCall]}


class Tool(SubscriptableBaseModel):
  """
  A tool the model may call, in the OpenAI function format.
  """

  class Function(SubscriptableBaseModel):
    name: Optional[str]
    description: Optional[str]
    parameters: Optional[Mapping[str, Any]]

  type: Optional[str]
  function: Optional[Function]

  _nested = {'function': Function}


class Usage(SubscriptableBaseModel):
  """
  Token counts and cost of a request.
  """

  prompt_tokens: Optional[int]
  completion_tokens: Optional[int]
  total_tokens: Optional[int]
  cost_zharf: Optional[float]
  'What the request cost, in ZHARF.'


class BaseResponse(SubscriptableBaseModel):
  id: Optional[str]
  model: Optional[str]
  'Model that answered.'

  created: Optional[int]
  'Unix time the response was created.'

  usage: Optional[Usage]
  'Token counts, as the platform reports them.'

  cost_zharf: Optional[float]
  'What the request cost, in ZHARF. On a stream, set on the final part.'

  _nested = {'usage': Usage}


class BaseGenerateResponse(BaseResponse):
  done: Optional[bool]
  'True on a complete response, and on the final part of a stream.'

  done_reason: Optional[str]
  "Why the model stopped: 'stop', 'length', 'tool_calls' and so on."

  prompt_eval_count: Optional[int]
  'Prompt tokens (same as usage.prompt_tokens).'

  eval_count: Optional[int]
  'Generated tokens (same as usage.completion_tokens).'


class ChatResponse(BaseGenerateResponse):
  """
  Response returned by chat requests.
  """

  message: Message
  'Response message.'

  _nested = {'message': Message}


class GenerateResponse(BaseGenerateResponse):
  """
  Response returned by generate requests.
  """

  response: str
  'Response text. When streaming, each part carries the next piece.'

  thinking: Optional[str]
  'Reasoning text, when the model returns it.'


class EmbedResponse(BaseResponse):
  """
  Response returned by embed requests.
  """

  embeddings: List[List[float]]
  'One embedding per input, in input order.'

  prompt_eval_count: Optional[int]


class Pricing(SubscriptableBaseModel):
  """
  A model's prices, in ZHARF. Media models are priced per `media_unit`.
  """

  currency: Optional[str]
  prompt_per_1m_tokens: Optional[float]
  completion_per_1m_tokens: Optional[float]
  input_cache_read_per_1m_tokens: Optional[float]
  input_cache_write_per_1m_tokens: Optional[float]
  per_unit: Optional[float]
  media_unit: Optional[str]
  priced_on_use: Optional[bool]


class Model(SubscriptableBaseModel):
  """
  A model on the platform, as listed by `GET /v1/models`.

  >>> Model(id='example-model').model
  'example-model'
  """

  Pricing = Pricing

  id: str
  'Model id to pass as `model`.'

  name: Optional[str]
  description: Optional[str]
  owned_by: Optional[str]
  created: Optional[int]
  added_at: Optional[int]
  'Unix time the model was added to the platform.'

  context_length: Optional[int]
  max_completion_tokens: Optional[int]
  modality: Optional[str]
  input_modalities: Optional[List[str]]
  output_modalities: Optional[List[str]]
  features: Optional[List[str]]
  supported_parameters: Optional[List[str]]
  operations: Optional[List[str]]
  pricing: Optional[Pricing]

  _nested = {'pricing': Pricing}

  @property
  def model(self) -> Optional[str]:
    """The model id, under the name ollama's `ListResponse.Model` uses."""
    return self._data.get('id')


class ListResponse(SubscriptableBaseModel):
  """
  Response returned by `list()`.
  """

  Model = Model

  models: List[Model]
  'Every model on the platform, with prices.'

  prices_valid_until: Optional[str]
  'Prices hold until this time (ISO 8601, UTC).'

  _nested = {'models': [Model]}


class ShowResponse(Model):
  """
  Response returned by `show()`: one model's details and prices.
  """


class ImageResponse(BaseResponse):
  """
  Response returned by `images.generate()`.
  """

  class Image(SubscriptableBaseModel):
    url: Optional[str]
    b64_json: Optional[str]
    revised_prompt: Optional[str]

  data: List[Image]

  _nested = {'data': [Image]}


class SpeechResponse(BaseResponse):
  """
  Response returned by `audio.speech()`. `bytes(response)` is the audio.
  """

  content: bytes
  content_type: Optional[str]

  def __bytes__(self) -> bytes:
    return self._data.get('content') or b''

  def save(self, path: Union[str, Path]) -> Path:
    """Write the audio to `path` and return it."""
    target = Path(path)
    target.write_bytes(bytes(self))
    return target

  def __repr__(self) -> str:
    return f'SpeechResponse(content=<{len(bytes(self))} bytes>, content_type={self.content_type!r}, cost_zharf={self.cost_zharf!r})'


class TranscriptionResponse(BaseResponse):
  """
  Response returned by `audio.transcribe()`.
  """

  text: str
  language: Optional[str]
  duration: Optional[float]
  segments: Optional[List[Any]]


class RequestError(Exception):
  """
  Raised before a request is sent, when its arguments are wrong.
  """

  def __init__(self, error: str):
    super().__init__(error)
    self.error = error
    'Reason for the error.'


class ResponseError(Exception):
  """
  Raised when the platform answers with an error.

  >>> e = ResponseError('Missing API key.', 401, message_fa='کلید API فرستاده نشده است.', code='no_key')
  >>> str(e)
  'Missing API key. (status code: 401)'
  >>> e.code
  'no_key'
  """

  def __init__(
    self,
    error: str,
    status_code: int = -1,
    *,
    message_fa: Optional[str] = None,
    code: Optional[str] = None,
    type: Optional[str] = None,
    retry_after: Optional[float] = None,
  ):
    super().__init__(error)
    self.error = error
    'Reason for the error, in English.'

    self.status_code = status_code
    'HTTP status code of the response, or -1 for an error inside a stream.'

    self.message_fa = message_fa
    'Reason for the error, in Persian.'

    self.code = code
    "Machine-readable code, such as 'no_key', 'insufficient_balance' or 'rate_limit'."

    self.type = type
    "Error type, such as 'authentication_error'."

    self.retry_after = retry_after
    'Seconds to wait before retrying, from the Retry-After header (429 and 503).'

  def __str__(self) -> str:
    return f'{self.error} (status code: {self.status_code})'
