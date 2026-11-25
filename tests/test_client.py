import asyncio
import json
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx
import pytest

from zharf import AsyncClient, ChatResponse, Client, GenerateResponse, Options, RequestError, ResponseError

HOST = 'https://platform.test/v1'
PERSIAN = 'سلام! ژرف پاسخ را تکه‌تکه می‌فرستد؛ نیم‌فاصله، عدد ۱۲۳ و 🌙 هم دارد.'


# helpers


class Recorder:
  """A MockTransport handler that records requests and answers with a fixed response."""

  def __init__(self, respond: Callable[[httpx.Request], httpx.Response]):
    self.respond = respond
    self.requests: List[httpx.Request] = []

  def __call__(self, request: httpx.Request) -> httpx.Response:
    request.read()
    self.requests.append(request)
    return self.respond(request)

  @property
  def body(self) -> Dict[str, Any]:
    return json.loads(self.requests[-1].content)


def make_client(respond, **kwargs) -> 'tuple[Client, Recorder]':
  recorder = Recorder(respond)
  kwargs.setdefault('api_key', 'zk-test')
  return Client(host=HOST, transport=httpx.MockTransport(recorder), **kwargs), recorder


def make_async_client(respond, **kwargs) -> 'tuple[AsyncClient, Recorder]':
  recorder = Recorder(respond)
  kwargs.setdefault('api_key', 'zk-test')
  return AsyncClient(host=HOST, transport=httpx.MockTransport(recorder), **kwargs), recorder


def completion(message: Optional[Dict[str, Any]] = None, usage: Optional[Dict[str, Any]] = None, finish_reason: str = 'stop') -> Dict[str, Any]:
  return {
    'id': 'chatcmpl-1',
    'object': 'chat.completion',
    'created': 1790000000,
    'model': 'test-model',
    'choices': [{'index': 0, 'message': {'role': 'assistant', **(message or {})}, 'finish_reason': finish_reason}],
    'usage': usage if usage is not None else {'prompt_tokens': 12, 'completion_tokens': 5, 'total_tokens': 17, 'cost_zharf': 0.0042},
  }


def chunk(delta: Optional[Dict[str, Any]] = None, finish_reason: Optional[str] = None) -> Dict[str, Any]:
  return {
    'id': 'chatcmpl-2',
    'object': 'chat.completion.chunk',
    'created': 1790000000,
    'model': 'test-model',
    'choices': [{'index': 0, 'delta': delta or {}, 'finish_reason': finish_reason}],
  }


USAGE_CHUNK = {
  'id': 'chatcmpl-2',
  'object': 'chat.completion.chunk',
  'model': 'test-model',
  'choices': [],
  'usage': {'prompt_tokens': 9, 'completion_tokens': 30, 'total_tokens': 39, 'cost_zharf': 0.0157},
}


def sse(*events: Dict[str, Any], done: bool = True) -> bytes:
  body = ''.join(f'data: {json.dumps(e, ensure_ascii=False)}\n\n' for e in events)
  return (body + ('data: [DONE]\n\n' if done else '')).encode()


class Chunks(httpx.SyncByteStream):
  def __init__(self, data: bytes, size: int):
    self.data, self.size = data, size

  def __iter__(self) -> Iterator[bytes]:
    for i in range(0, len(self.data), self.size):
      yield self.data[i : i + self.size]


class AsyncChunks(httpx.AsyncByteStream):
  def __init__(self, data: bytes, size: int):
    self.data, self.size = data, size

  async def __aiter__(self):
    for i in range(0, len(self.data), self.size):
      await asyncio.sleep(0)
      yield self.data[i : i + self.size]


def sse_response(body: bytes, size: Optional[int] = None, *, use_async: bool = False) -> httpx.Response:
  headers = {'content-type': 'text/event-stream'}
  if size is None:
    return httpx.Response(200, headers=headers, content=body)
  stream = AsyncChunks(body, size) if use_async else Chunks(body, size)
  return httpx.Response(200, headers=headers, stream=stream)


def persian_stream() -> bytes:
  pieces = [PERSIAN[:7], PERSIAN[7:23], PERSIAN[23:41], PERSIAN[41:]]
  return sse(
    chunk({'role': 'assistant', 'content': ''}),
    *[chunk({'content': piece}) for piece in pieces],
    chunk({}, finish_reason='stop'),
    USAGE_CHUNK,
  )


# chat


def test_chat():
  client, recorder = make_client(lambda r: httpx.Response(200, json=completion({'content': 'Hi there', 'reasoning_content': 'The user greets me.'})))
  response = client.chat('test-model', messages=[{'role': 'user', 'content': 'Hello'}])

  request = recorder.requests[-1]
  assert request.method == 'POST'
  assert str(request.url) == f'{HOST}/chat/completions'
  assert request.headers['authorization'] == 'Bearer zk-test'
  assert recorder.body == {'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}]}

  assert isinstance(response, ChatResponse)
  assert response.message.content == 'Hi there'
  assert response['message']['content'] == 'Hi there'
  assert response.message.role == 'assistant'
  assert response.message.thinking == 'The user greets me.'
  assert response.message.tool_calls is None
  assert response.done is True
  assert response.done_reason == 'stop'
  assert response.model == 'test-model'
  assert response.cost_zharf == pytest.approx(0.0042)
  assert response.usage.total_tokens == 17
  assert response['usage']['cost_zharf'] == pytest.approx(0.0042)
  assert (response.prompt_eval_count, response.eval_count) == (12, 5)


def test_chat_cost_from_header():
  usage = {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}
  body = completion({'content': 'ok'}, usage=usage)
  client, _ = make_client(lambda r: httpx.Response(200, json=body, headers={'x-zharf-cost': '0.0021'}))
  assert client.chat('test-model', [{'role': 'user', 'content': 'hi'}]).cost_zharf == pytest.approx(0.0021)


def test_chat_reasoning_field():
  client, _ = make_client(lambda r: httpx.Response(200, json=completion({'content': None, 'reasoning': 'thinking...'})))
  response = client.chat('test-model', [{'role': 'user', 'content': 'hi'}])
  assert response.message.thinking == 'thinking...'
  assert response.message.content == ''


def test_chat_requires_model():
  client, recorder = make_client(lambda r: httpx.Response(200, json=completion()))
  with pytest.raises(RequestError):
    client.chat('', [{'role': 'user', 'content': 'hi'}])
  assert recorder.requests == []


@pytest.mark.parametrize(
  ('options', 'expected'),
  [
    ({'temperature': 0.3}, {'temperature': 0.3}),
    ({'top_p': 0.9, 'top_k': 40, 'min_p': 0.05}, {'top_p': 0.9, 'top_k': 40, 'min_p': 0.05}),
    ({'num_predict': 256}, {'max_tokens': 256}),
    ({'num_predict': -1}, {}),
    ({'stop': ['\n\n', 'END']}, {'stop': ['\n\n', 'END']}),
    ({'stop': ('a', 'b')}, {'stop': ['a', 'b']}),
    ({'seed': 42}, {'seed': 42}),
    (
      {'repeat_penalty': 1.1, 'presence_penalty': 0.5, 'frequency_penalty': 0.2},
      {'repetition_penalty': 1.1, 'presence_penalty': 0.5, 'frequency_penalty': 0.2},
    ),
    ({'num_ctx': 8192, 'num_gpu': 1, 'mirostat': 2, 'temperature': 0}, {'temperature': 0}),
    ({'max_tokens': 10, 'logit_bias': {'50256': -100}}, {'max_tokens': 10, 'logit_bias': {'50256': -100}}),
    (Options(temperature=0.7, num_predict=64, seed=1), {'temperature': 0.7, 'max_tokens': 64, 'seed': 1}),
  ],
)
def test_chat_options(options, expected):
  client, recorder = make_client(lambda r: httpx.Response(200, json=completion({'content': 'ok'})))
  client.chat('test-model', [{'role': 'user', 'content': 'hi'}], options=options)
  body = recorder.body
  assert {k: v for k, v in body.items() if k not in ('model', 'messages')} == expected


@pytest.mark.parametrize(
  ('think', 'expected'),
  [
    (True, {'reasoning': {'enabled': True}}),
    (False, {'reasoning': {'enabled': False}}),
    ('low', {'reasoning_effort': 'low'}),
    ('high', {'reasoning_effort': 'high'}),
    (None, {}),
  ],
)
def test_chat_think(think, expected):
  client, recorder = make_client(lambda r: httpx.Response(200, json=completion({'content': 'ok'})))
  client.chat('test-model', [{'role': 'user', 'content': 'hi'}], think=think)
  assert {k: v for k, v in recorder.body.items() if k not in ('model', 'messages')} == expected


def test_chat_format():
  client, recorder = make_client(lambda r: httpx.Response(200, json=completion({'content': '{"city": "Tehran"}'})))

  client.chat('test-model', [{'role': 'user', 'content': 'hi'}], format='json')
  assert recorder.body['response_format'] == {'type': 'json_object'}

  schema = {'title': 'City Answer', 'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}
  client.chat('test-model', [{'role': 'user', 'content': 'hi'}], format=schema)
  assert recorder.body['response_format'] == {'type': 'json_schema', 'json_schema': {'name': 'City_Answer', 'schema': schema}}

  ready = {'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': {'type': 'object'}}}
  client.chat('test-model', [{'role': 'user', 'content': 'hi'}], format=ready)
  assert recorder.body['response_format'] == ready


def test_chat_extra_body_and_images(tmp_path):
  image = tmp_path / 'dot.png'
  image.write_bytes(b'\x89PNG\r\n\x1a\nfake')
  client, recorder = make_client(lambda r: httpx.Response(200, json=completion({'content': 'a dot'})))
  client.chat(
    'test-model',
    [{'role': 'user', 'content': 'What is this?', 'images': [str(image), 'https://example.com/a.jpg']}],
    extra_body={'tool_choice': 'none', 'temperature': 0.1},
  )
  body = recorder.body
  assert body['tool_choice'] == 'none'
  assert body['temperature'] == 0.1
  content = body['messages'][0]['content']
  assert content[0] == {'type': 'text', 'text': 'What is this?'}
  assert content[1]['image_url']['url'].startswith('data:image/png;base64,')
  assert content[2] == {'type': 'image_url', 'image_url': {'url': 'https://example.com/a.jpg'}}


# streaming


def test_chat_stream():
  client, recorder = make_client(lambda r: sse_response(persian_stream()))
  parts = list(client.chat('test-model', [{'role': 'user', 'content': 'سلام'}], stream=True))

  body = recorder.body
  assert body['stream'] is True
  assert body['stream_options'] == {'include_usage': True}
  assert recorder.requests[-1].headers['accept'] == 'text/event-stream'

  assert ''.join(p.message.content for p in parts) == PERSIAN
  assert ''.join(p['message']['content'] for p in parts) == PERSIAN
  assert [p.done for p in parts] == [False] * (len(parts) - 1) + [True]
  assert all(p.cost_zharf is None for p in parts[:-1])

  final = parts[-1]
  assert final.done_reason == 'stop'
  assert final.cost_zharf == pytest.approx(0.0157)
  assert final.usage.completion_tokens == 30
  assert final.eval_count == 30
  assert final.message.content == ''


@pytest.mark.parametrize('size', [1, 2, 3, 5, 7, 13, 64])
def test_chat_stream_multibyte_split(size):
  body = persian_stream()
  # make sure the chunking really cuts characters in half
  assert any(not _decodes(body[i : i + size]) for i in range(0, len(body), size))
  client, _ = make_client(lambda r: sse_response(body, size))
  parts = list(client.chat('test-model', [{'role': 'user', 'content': 'سلام'}], stream=True))
  assert ''.join(p.message.content for p in parts) == PERSIAN
  assert '�' not in ''.join(p.message.content for p in parts)
  assert parts[-1].done is True
  assert parts[-1].cost_zharf == pytest.approx(0.0157)


def _decodes(data: bytes) -> bool:
  try:
    data.decode('utf-8')
    return True
  except UnicodeDecodeError:
    return False


def test_chat_stream_crlf_comments_and_no_done_marker():
  events = [chunk({'content': 'یک '}), chunk({'content': 'دو'}, finish_reason='stop'), USAGE_CHUNK]
  body = b': keep-alive\r\n\r\n' + sse(*events, done=False).replace(b'\n', b'\r\n')
  client, _ = make_client(lambda r: sse_response(body, 4))
  parts = list(client.chat('test-model', [{'role': 'user', 'content': 'hi'}], stream=True))
  assert ''.join(p.message.content for p in parts) == 'یک دو'
  assert parts[-1].done is True
  assert parts[-1].cost_zharf == pytest.approx(0.0157)


def test_chat_stream_thinking():
  body = sse(
    chunk({'reasoning_content': 'Let me '}),
    chunk({'reasoning': 'think. '}),
    chunk({'content': 'Answer.'}),
    chunk({}, finish_reason='stop'),
    USAGE_CHUNK,
  )
  client, _ = make_client(lambda r: sse_response(body))
  parts = list(client.chat('test-model', [{'role': 'user', 'content': 'hi'}], stream=True, think=True))
  assert ''.join(p.message.thinking or '' for p in parts) == 'Let me think. '
  assert ''.join(p.message.content for p in parts) == 'Answer.'


def test_chat_stream_tool_calls():
  body = sse(
    chunk({'role': 'assistant', 'tool_calls': [{'index': 0, 'id': 'call_a', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': ''}}]}),
    chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '{"city": "تهر'}}]}),
    chunk({'tool_calls': [{'index': 0, 'function': {'arguments': 'ان"}'}}]}),
    chunk({'tool_calls': [{'index': 1, 'id': 'call_b', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 1, "b": 2}'}}]}),
    chunk({}, finish_reason='tool_calls'),
    USAGE_CHUNK,
  )
  client, _ = make_client(lambda r: sse_response(body, 3))
  parts = list(client.chat('test-model', [{'role': 'user', 'content': 'weather?'}], stream=True))
  final = parts[-1]
  assert final.done is True
  assert final.done_reason == 'tool_calls'
  calls = final.message.tool_calls
  assert [c.id for c in calls] == ['call_a', 'call_b']
  assert calls[0].function.name == 'get_weather'
  assert calls[0].function.arguments == {'city': 'تهران'}
  assert calls[1]['function']['arguments'] == {'a': 1, 'b': 2}


def test_chat_stream_json_answer():
  client, _ = make_client(lambda r: httpx.Response(200, json=completion({'content': 'whole'})))
  parts = list(client.chat('test-model', [{'role': 'user', 'content': 'hi'}], stream=True))
  assert len(parts) == 1
  assert parts[0].message.content == 'whole'
  assert parts[0].done is True


# tools


def get_weather(city: str, unit: Optional[str] = None) -> str:
  """
  Get the current weather in a city.

  Args:
    city: The city name
    unit (str): 'c' or 'f'
  """
  return f'{city}: 18 degrees'


def test_tools_round_trip():
  first = completion(
    {
      'content': None,
      'tool_calls': [
        {'id': 'call_1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "اصفهان"}'}},
        {'id': 'call_2', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 2, "b": 3}'}},
      ],
    },
    finish_reason='tool_calls',
  )
  answers = [first, completion({'content': 'In Isfahan it is 18 degrees, and 2 + 3 = 5.'})]
  client, recorder = make_client(lambda r: httpx.Response(200, json=answers[len(recorder.requests) - 1]))

  add_tool = {
    'type': 'function',
    'function': {'name': 'add', 'description': 'Add two numbers', 'parameters': {'type': 'object', 'properties': {}}},
  }
  messages: List[Any] = [{'role': 'user', 'content': 'Weather in Isfahan, and 2 + 3?'}]
  response = client.chat('test-model', messages, tools=[get_weather, add_tool])

  tools = recorder.body['tools']
  assert tools[0]['type'] == 'function'
  assert tools[0]['function']['name'] == 'get_weather'
  assert tools[0]['function']['description'] == 'Get the current weather in a city.'
  assert tools[0]['function']['parameters']['required'] == ['city']
  assert tools[0]['function']['parameters']['properties']['city'] == {'type': 'string', 'description': 'The city name'}
  assert tools[1] == add_tool

  assert response.message.content == ''
  assert response.done_reason == 'tool_calls'
  calls = response.message.tool_calls
  assert calls[0].function.arguments == {'city': 'اصفهان'}

  # the ollama way: append the message, then one result per tool, named by tool_name
  messages.append(response.message)
  for call in reversed(calls):  # answered out of order on purpose
    output = get_weather(**call.function.arguments) if call.function.name == 'get_weather' else 5
    messages.append({'role': 'tool', 'content': output, 'tool_name': call.function.name})

  final = client.chat('test-model', messages, tools=[get_weather, add_tool])
  assert final.message.content.startswith('In Isfahan')

  sent = recorder.body['messages']
  assert sent[1] == {
    'role': 'assistant',
    'content': '',
    'tool_calls': [
      {'id': 'call_1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "اصفهان"}'}},
      {'id': 'call_2', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 2, "b": 3}'}},
    ],
  }
  assert sent[2] == {'role': 'tool', 'content': '5', 'tool_call_id': 'call_2'}
  assert sent[3] == {'role': 'tool', 'content': 'اصفهان: 18 degrees', 'tool_call_id': 'call_1'}


def test_tool_results_without_names_or_ids():
  client, recorder = make_client(lambda r: httpx.Response(200, json=completion({'content': 'ok'})))
  messages = [
    {'role': 'user', 'content': 'hi'},
    {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'a', 'arguments': {}}}, {'function': {'name': 'b', 'arguments': {'x': 1}}}]},
    {'role': 'tool', 'content': {'result': 1}},
    {'role': 'tool', 'content': 'two'},
  ]
  client.chat('test-model', messages)
  sent = recorder.body['messages']
  ids = [c['id'] for c in sent[1]['tool_calls']]
  assert len(set(ids)) == 2
  assert sent[1]['tool_calls'][1]['function']['arguments'] == '{"x": 1}'
  assert [sent[2]['tool_call_id'], sent[3]['tool_call_id']] == ids
  assert sent[2]['content'] == '{"result": 1}'


# errors


def test_error_json():
  error = {'error': {'message': 'Missing API key.', 'message_fa': 'کلید API فرستاده نشده است.', 'type': 'authentication_error', 'code': 'no_key'}}
  client, _ = make_client(lambda r: httpx.Response(401, json=error))
  with pytest.raises(ResponseError) as info:
    client.chat('test-model', [{'role': 'user', 'content': 'hi'}])
  e = info.value
  assert e.status_code == 401
  assert e.error == 'Missing API key.'
  assert e.message_fa == 'کلید API فرستاده نشده است.'
  assert e.code == 'no_key'
  assert e.type == 'authentication_error'
  assert e.retry_after is None
  assert str(e) == 'Missing API key. (status code: 401)'


@pytest.mark.parametrize(('status', 'code'), [(402, 'insufficient_balance'), (403, 'model_not_offered'), (404, 'model_not_found')])
def test_error_codes(status, code):
  client, _ = make_client(lambda r: httpx.Response(status, json={'error': {'message': 'nope', 'message_fa': 'نه', 'type': 'x', 'code': code}}))
  with pytest.raises(ResponseError) as info:
    client.embed('test-model', 'hi')
  assert (info.value.status_code, info.value.code) == (status, code)


def test_error_rate_limit_retry_after():
  error = {'error': {'message': 'Too many requests.', 'message_fa': 'درخواست‌ها زیاد است.', 'type': 'rate_limit_error', 'code': 'rate_limit'}}
  client, _ = make_client(lambda r: httpx.Response(429, json=error, headers={'retry-after': '12'}))
  with pytest.raises(ResponseError) as info:
    client.chat('test-model', [{'role': 'user', 'content': 'hi'}])
  assert info.value.status_code == 429
  assert info.value.code == 'rate_limit'
  assert info.value.retry_after == 12.0


def test_error_plain_text():
  client, _ = make_client(lambda r: httpx.Response(503, text='Service Unavailable', headers={'retry-after': '30'}))
  with pytest.raises(ResponseError) as info:
    client.list()
  assert info.value.status_code == 503
  assert info.value.error == 'Service Unavailable'
  assert info.value.code is None
  assert info.value.retry_after == 30.0


def test_error_on_stream_request():
  error = {'error': {'message': 'Not enough credit.', 'message_fa': 'اعتبار کافی نیست.', 'type': 'billing_error', 'code': 'insufficient_balance'}}
  client, _ = make_client(lambda r: httpx.Response(402, json=error))
  stream = client.chat('test-model', [{'role': 'user', 'content': 'hi'}], stream=True)
  with pytest.raises(ResponseError) as info:
    next(iter(stream))
  assert info.value.status_code == 402
  assert info.value.message_fa == 'اعتبار کافی نیست.'


def test_error_inside_stream():
  body = sse(chunk({'content': 'partial '}), {'error': {'message': 'The model stopped.', 'message_fa': 'مدل متوقف شد.', 'code': 'stream_error'}})
  client, _ = make_client(lambda r: sse_response(body))
  parts = []
  with pytest.raises(ResponseError) as info:
    for part in client.chat('test-model', [{'role': 'user', 'content': 'hi'}], stream=True):
      parts.append(part)
  assert [p.message.content for p in parts] == ['partial ']
  assert info.value.status_code == -1
  assert info.value.code == 'stream_error'


def test_connection_error():
  def refuse(request):
    raise httpx.ConnectError('connection refused', request=request)

  client, _ = make_client(refuse)
  with pytest.raises(ConnectionError):
    client.list()
  with pytest.raises(ConnectionError):
    list(client.chat('test-model', [{'role': 'user', 'content': 'hi'}], stream=True))


# async


def test_async_chat():
  async def main():
    client, recorder = make_async_client(lambda r: httpx.Response(200, json=completion({'content': 'سلام'})))
    response = await client.chat('test-model', [{'role': 'user', 'content': 'hi'}])
    assert response.message.content == 'سلام'
    assert response.cost_zharf == pytest.approx(0.0042)
    assert recorder.requests[-1].headers['authorization'] == 'Bearer zk-test'
    await client.close()

  asyncio.run(main())


@pytest.mark.parametrize('size', [1, 3, 7])
def test_async_chat_stream_multibyte_split(size):
  async def main():
    client, recorder = make_async_client(lambda r: sse_response(persian_stream(), size, use_async=True))
    parts = [part async for part in await client.chat('test-model', [{'role': 'user', 'content': 'hi'}], stream=True)]
    assert ''.join(p.message.content for p in parts) == PERSIAN
    assert parts[-1].done is True
    assert parts[-1].cost_zharf == pytest.approx(0.0157)
    assert recorder.body['stream_options'] == {'include_usage': True}

  asyncio.run(main())


def test_async_generate_stream_and_errors():
  async def main():
    client, _ = make_async_client(lambda r: sse_response(persian_stream(), 5, use_async=True))
    parts = [p async for p in await client.generate('test-model', 'hi', stream=True)]
    assert ''.join(p.response for p in parts) == PERSIAN
    assert isinstance(parts[-1], GenerateResponse)
    assert parts[-1].done is True

    busy = {'error': {'message': 'slow down', 'code': 'too_many_inflight'}}
    failing, _ = make_async_client(lambda r: httpx.Response(429, json=busy, headers={'retry-after': '2'}))
    with pytest.raises(ResponseError) as info:
      await failing.list()
    assert (info.value.code, info.value.retry_after) == ('too_many_inflight', 2.0)

    stream = await failing.chat('test-model', [{'role': 'user', 'content': 'hi'}], stream=True)
    with pytest.raises(ResponseError):
      async for _ in stream:
        pass

  asyncio.run(main())


def test_async_list_embed_and_extras():
  def respond(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith('/models'):
      return httpx.Response(200, json=MODELS)
    if request.url.path.endswith('/embeddings'):
      return httpx.Response(200, json=EMBEDDINGS)
    if request.url.path.endswith('/audio/speech'):
      return httpx.Response(200, content=b'ID3audio', headers={'content-type': 'audio/mpeg', 'x-zharf-cost': '0.5'})
    return httpx.Response(200, json={'text': 'سلام دنیا'}, headers={'x-zharf-cost': '0.25'})

  async def main():
    async with AsyncClient(host=HOST, api_key='zk-test', transport=httpx.MockTransport(respond)) as client:
      assert len((await client.list()).models) == 2
      assert (await client.embed('embed-model', ['a', 'b'])).embeddings == [[0.1, 0.2], [0.3, 0.4]]
      assert bytes(await client.audio.speech('tts-model', 'hi', voice='v1')) == b'ID3audio'
      assert (await client.audio.transcribe('stt-model', b'RIFF0000WAVEdata')).text == 'سلام دنیا'

  asyncio.run(main())


# generate


def test_generate():
  client, recorder = make_client(lambda r: httpx.Response(200, json=completion({'content': 'Blue light scatters more.'})))
  response = client.generate('test-model', 'Why is the sky blue?', system='Answer in one sentence.')
  assert recorder.body['messages'] == [
    {'role': 'system', 'content': 'Answer in one sentence.'},
    {'role': 'user', 'content': 'Why is the sky blue?'},
  ]
  assert isinstance(response, GenerateResponse)
  assert response.response == 'Blue light scatters more.'
  assert response['response'] == 'Blue light scatters more.'
  assert response.done is True
  assert response.cost_zharf == pytest.approx(0.0042)
  assert response.usage.total_tokens == 17


def test_generate_stream():
  client, _ = make_client(lambda r: sse_response(persian_stream(), 2))
  parts = list(client.generate('test-model', 'hi', stream=True))
  assert ''.join(p.response for p in parts) == PERSIAN
  assert parts[-1].cost_zharf == pytest.approx(0.0157)


# models

MODELS = {
  'object': 'list',
  'prices_valid_until': '2026-11-25T20:30:00.000Z',
  'data': [
    {
      'id': 'chat-model',
      'object': 'model',
      'name': 'Chat Model',
      'owned_by': 'Example Lab',
      'created': 1735689600,
      'added_at': 1790000000,
      'context_length': 128000,
      'max_completion_tokens': 8192,
      'modality': 'text+image->text',
      'input_modalities': ['text', 'image'],
      'output_modalities': ['text'],
      'features': ['streaming', 'tools', 'reasoning'],
      'supported_parameters': ['tools', 'temperature'],
      'operations': [],
      'reasoning': {'mandatory': False},
      'pricing': {
        'currency': 'ZHARF',
        'prompt_per_1m_tokens': 40.5,
        'completion_per_1m_tokens': 120.25,
        'input_cache_read_per_1m_tokens': 4.05,
      },
    },
    {
      'id': 'voice-model',
      'name': 'Voice Model',
      'owned_by': 'Example Lab',
      'modality': 'text->audio',
      'operations': ['tts'],
      'voices': ['a', 'b'],
      'pricing': {'currency': 'ZHARF', 'prompt_per_1m_tokens': 0, 'completion_per_1m_tokens': 0, 'per_unit': 3.5, 'media_unit': '1K characters'},
    },
  ],
}


def test_list():
  client, recorder = make_client(lambda r: httpx.Response(200, json=MODELS), api_key=None)
  response = client.list()
  request = recorder.requests[-1]
  assert (request.method, str(request.url)) == ('GET', f'{HOST}/models')
  assert response.prices_valid_until == '2026-11-25T20:30:00.000Z'
  assert len(response.models) == 2

  chat_model, voice = response.models
  assert chat_model.id == chat_model.model == chat_model['model'] == 'chat-model'
  assert chat_model.pricing.currency == 'ZHARF'
  assert chat_model.pricing.prompt_per_1m_tokens == 40.5
  assert chat_model['pricing']['completion_per_1m_tokens'] == 120.25
  assert chat_model.pricing.per_unit is None
  assert chat_model.reasoning == {'mandatory': False}  # a field this version does not name
  assert 'tools' in chat_model.features
  assert voice.voices == ['a', 'b']
  assert voice.pricing.media_unit == '1K characters'
  assert voice.context_length is None


def test_show():
  client, recorder = make_client(lambda r: httpx.Response(200, json=MODELS['data'][0]))
  model = client.show('chat-model')
  assert str(recorder.requests[-1].url) == f'{HOST}/models/chat-model'
  assert model.name == 'Chat Model'
  assert model.pricing.input_cache_read_per_1m_tokens == 4.05


def test_embed():
  client, recorder = make_client(lambda r: httpx.Response(200, json=EMBEDDINGS))
  response = client.embed('embed-model', ['first', 'second'], dimensions=2)
  assert recorder.body == {'model': 'embed-model', 'input': ['first', 'second'], 'dimensions': 2}
  assert response.embeddings == [[0.1, 0.2], [0.3, 0.4]]
  assert response['embeddings'][1] == [0.3, 0.4]
  assert response.cost_zharf == pytest.approx(0.0001)
  assert response.prompt_eval_count == 4


EMBEDDINGS = {
  'object': 'list',
  'model': 'embed-model',
  'data': [
    {'object': 'embedding', 'index': 1, 'embedding': [0.3, 0.4]},
    {'object': 'embedding', 'index': 0, 'embedding': [0.1, 0.2]},
  ],
  'usage': {'prompt_tokens': 4, 'total_tokens': 4, 'cost_zharf': 0.0001},
}


# images and audio


def test_images_generate():
  body = {'created': 1790000000, 'data': [{'b64_json': 'aGVsbG8=', 'revised_prompt': 'a cat'}], 'usage': {'cost_zharf': 1.5}}
  client, recorder = make_client(lambda r: httpx.Response(200, json=body))
  response = client.images.generate('image-model', 'a cat', size='1024x1024', response_format='b64_json')
  assert recorder.body == {'model': 'image-model', 'prompt': 'a cat', 'size': '1024x1024', 'response_format': 'b64_json'}
  assert response.data[0].b64_json == 'aGVsbG8='
  assert response['data'][0]['revised_prompt'] == 'a cat'
  assert response.cost_zharf == 1.5


def test_audio_speech(tmp_path):
  client, recorder = make_client(lambda r: httpx.Response(200, content=b'ID3fake-mp3', headers={'content-type': 'audio/mpeg', 'x-zharf-cost': '0.031'}))
  response = client.audio.speech('tts-model', 'سلام', voice='v1', response_format='mp3')
  assert recorder.body == {'model': 'tts-model', 'input': 'سلام', 'voice': 'v1', 'response_format': 'mp3'}
  assert bytes(response) == b'ID3fake-mp3'
  assert response.content_type == 'audio/mpeg'
  assert response.cost_zharf == pytest.approx(0.031)
  path = response.save(tmp_path / 'hello.mp3')
  assert path.read_bytes() == b'ID3fake-mp3'


def test_audio_transcribe(tmp_path):
  audio = tmp_path / 'voice.ogg'
  audio.write_bytes(b'OggSfake-audio')
  client, recorder = make_client(lambda r: httpx.Response(200, json={'text': 'سلام دنیا', 'language': 'fa'}, headers={'x-zharf-cost': '0.02'}))
  response = client.audio.transcribe('stt-model', audio, language='fa')

  request = recorder.requests[-1]
  assert request.headers['content-type'].startswith('multipart/form-data; boundary=')
  assert b'name="model"\r\n\r\nstt-model' in request.content
  assert b'name="language"\r\n\r\nfa' in request.content
  assert b'filename="voice.ogg"' in request.content
  assert b'OggSfake-audio' in request.content
  assert response.text == 'سلام دنیا'
  assert response.language == 'fa'
  assert response.cost_zharf == pytest.approx(0.02)

  plain, _ = make_client(lambda r: httpx.Response(200, text='plain words', headers={'content-type': 'text/plain'}))
  assert plain.audio.transcribe('stt-model', b'RIFF0000WAVE', response_format='text').text == 'plain words'


# configuration


def test_api_key_from_env_at_request_time(monkeypatch):
  monkeypatch.delenv('ZHARF_API_KEY', raising=False)
  client, recorder = make_client(lambda r: httpx.Response(200, json=MODELS), api_key=None)
  client.list()
  assert 'authorization' not in recorder.requests[-1].headers

  monkeypatch.setenv('ZHARF_API_KEY', 'zk-from-env')
  client.list()
  assert recorder.requests[-1].headers['authorization'] == 'Bearer zk-from-env'


def test_explicit_key_and_headers(monkeypatch):
  monkeypatch.setenv('ZHARF_API_KEY', 'zk-from-env')
  client, recorder = make_client(lambda r: httpx.Response(200, json=MODELS), api_key='zk-explicit', headers={'X-Trace': 'abc'})
  client.list()
  headers = recorder.requests[-1].headers
  assert headers['authorization'] == 'Bearer zk-explicit'
  assert headers['x-trace'] == 'abc'
  assert headers['user-agent'].startswith('zharf-python/')

  custom, recorder = make_client(lambda r: httpx.Response(200, json=MODELS), headers={'Authorization': 'Bearer zk-header'})
  custom.list()
  assert recorder.requests[-1].headers.get_list('authorization') == ['Bearer zk-header']


def test_base_url_from_env(monkeypatch):
  monkeypatch.setenv('ZHARF_BASE_URL', 'https://proxy.example.com')
  recorder = Recorder(lambda r: httpx.Response(200, json=MODELS))
  client = Client(transport=httpx.MockTransport(recorder))
  client.list()
  assert str(recorder.requests[-1].url) == 'https://proxy.example.com/v1/models'


def test_default_host(monkeypatch):
  monkeypatch.delenv('ZHARF_BASE_URL', raising=False)
  recorder = Recorder(lambda r: httpx.Response(200, json=MODELS))
  with Client(transport=httpx.MockTransport(recorder)) as client:
    client.list()
  assert str(recorder.requests[-1].url) == 'https://platform.zharf.ai/v1/models'


def test_module_functions():
  import zharf

  for name in ('chat', 'generate', 'list', 'show', 'embed'):
    assert callable(getattr(zharf, name))
  assert callable(zharf.images.generate)
  assert callable(zharf.audio.speech)
  assert callable(zharf.audio.transcribe)
