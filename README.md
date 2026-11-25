# Zharf Python Library

The Zharf Python library is the easiest way to use [platform.zharf.ai](https://platform.zharf.ai) from Python 3.9+. Its calls and response objects follow the [Ollama Python library](https://github.com/ollama/ollama-python), so much code written for Ollama runs after changing the import and the model name. Underneath, it speaks the platform's OpenAI-compatible API. Its only dependency is [httpx](https://www.python-httpx.org).

[راهنمای فارسی](#راهنمای-فارسی)

## Prerequisites

- An API key from [platform.zharf.ai](https://platform.zharf.ai). Keys start with `zk-`. Listing models and prices needs no key.
- Put the key in the environment:

```sh
export ZHARF_API_KEY=zk-...
```

## Install

```sh
pip install zharf
```

## Usage

```python
from zharf import chat
from zharf import ChatResponse

response: ChatResponse = chat(
  model='deepseek-v4-flash',
  messages=[
    {
      'role': 'user',
      'content': 'Why is the sky blue?',
    },
  ],
)
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)
# what this request cost, in ZHARF
print(response.cost_zharf)
```

Any chat model id from [`list()`](#list) works as `model`. See [_types.py](zharf/_types.py) for the response types.

## Streaming responses

Response streaming can be enabled by setting `stream=True`.

```python
from zharf import chat

stream = chat(
  model='deepseek-v4-flash',
  messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
  stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
  if chunk.done:
    print('\nCost:', chunk.cost_zharf, 'ZHARF')
```

The last part has `done=True`, the finish reason in `done_reason`, token counts in `usage`, and `cost_zharf` for the whole request. Persian and other multi-byte text arrives intact even when a character's bytes are split across network packets.

## Thinking

```python
response = chat('deepseek-v4-flash', messages=[{'role': 'user', 'content': 'Is 1001 prime?'}], think=True)
print('Thinking:', response.message.thinking)
print('Answer:', response.message.content)
```

`think=True` or `False` turns reasoning on or off, and `'low'`, `'medium'` or `'high'` sets its effort, on models that support it (see `features` and `supported_parameters` in [`show()`](#show)). When streaming, the reasoning arrives in `part.message.thinking`.

## Tools

Pass Python functions, or tools written as JSON schema. A function is described to the model from its type hints and Google-style docstring.

```python
from zharf import chat


def get_temperature(city: str) -> str:
  """
  Get the current temperature in a city.

  Args:
    city: The city name, such as Tehran
  """
  return '18 degrees'


messages = [{'role': 'user', 'content': 'How warm is it in Tehran?'}]
response = chat('deepseek-v4-flash', messages=messages, tools=[get_temperature])

if response.message.tool_calls:
  messages.append(response.message)
  for call in response.message.tool_calls:
    result = get_temperature(**call.function.arguments)
    messages.append({'role': 'tool', 'content': result, 'tool_name': call.function.name})
  print(chat('deepseek-v4-flash', messages=messages).message.content)
```

`call.function.arguments` is a dict. Tool results are matched to their calls by `tool_name` (or set `tool_call_id` yourself). When streaming, the complete tool calls are on the final part. See [examples/tools.py](examples/tools.py).

## Structured outputs

```python
response = chat('deepseek-v4-flash', messages=[...], format='json')  # any JSON object

schema = {
  'title': 'City',
  'type': 'object',
  'properties': {'name': {'type': 'string'}, 'population': {'type': 'integer'}},
  'required': ['name', 'population'],
}
response = chat('deepseek-v4-flash', messages=[...], format=schema)  # this shape
```

A pydantic model works too: `format=City.model_json_schema()`.

## Options

`options` takes Ollama's names:

| Option | Sent as |
|---|---|
| `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `presence_penalty`, `frequency_penalty` | the same name |
| `num_predict` | `max_tokens` |
| `repeat_penalty` | `repetition_penalty` |
| `num_ctx`, `num_gpu`, `num_thread` and other options for local models | not sent |

```python
chat('deepseek-v4-flash', messages=[...], options={'temperature': 0.2, 'num_predict': 500, 'seed': 7})
```

Any other key in `options` is sent as it is. `extra_body` adds any other request field, for example `extra_body={'tool_choice': 'required'}`. Each model's `supported_parameters` lists what it accepts.

## Custom client

A custom client can be created by instantiating `Client` or `AsyncClient` from `zharf`.

All extra keyword arguments are passed into the [`httpx.Client`](https://www.python-httpx.org/api/#client).

```python
from zharf import Client

client = Client(
  host='https://platform.zharf.ai',  # /v1 is added when missing
  api_key='zk-...',                   # default: $ZHARF_API_KEY
  timeout=120,                        # default: 600 s per read, 10 s to connect
  headers={'x-some-header': 'some-value'},
)
response = client.chat(model='deepseek-v4-flash', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
```

| Environment variable | Use |
|---|---|
| `ZHARF_API_KEY` | API key, read on every request |
| `ZHARF_BASE_URL` | API address (default `https://platform.zharf.ai/v1`) |
| `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY` | proxy settings, honoured by httpx; for a SOCKS proxy also `pip install 'httpx[socks]'` |

## Async client

The `AsyncClient` class is used to make asynchronous requests. It can be configured with the same fields as the `Client` class.

```python
import asyncio
from zharf import AsyncClient


async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  response = await AsyncClient().chat(model='deepseek-v4-flash', messages=[message])


asyncio.run(chat())
```

Setting `stream=True` modifies functions to return a Python asynchronous generator:

```python
import asyncio
from zharf import AsyncClient


async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  async for part in await AsyncClient().chat(model='deepseek-v4-flash', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)


asyncio.run(chat())
```

## API

### Chat

```python
zharf.chat(model='deepseek-v4-flash', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
```

Messages may carry `images`: file paths, bytes, base64 strings or URLs, for models with `vision` in `features`.

### Generate

```python
zharf.generate(model='deepseek-v4-flash', prompt='Why is the sky blue?', system='Answer in one sentence.')
```

Returns `.response` instead of `.message`. It runs on the chat endpoint.

### List

```python
zharf.list()
```

Every model with its prices; needs no key. Prices are in ZHARF per million tokens, or per `media_unit` for image, audio and video models, and hold until `prices_valid_until`.

```python
for model in zharf.list().models:
  print(model.id, model.pricing.prompt_per_1m_tokens, model.pricing.completion_per_1m_tokens, model.pricing.currency)
```

### Show

```python
zharf.show('deepseek-v4-flash')
```

One model's details: `context_length`, `max_completion_tokens`, `modality`, `features`, `supported_parameters`, `pricing` and any model-specific fields such as `voices`.

### Embed

```python
zharf.embed(model=embedding_model, input='The sky is blue because of Rayleigh scattering')
```

### Embed (batch)

```python
zharf.embed(model=embedding_model, input=['The sky is blue because of Rayleigh scattering', 'Grass is green because of chlorophyll'])
```

`embedding_model` is the id of a model whose `operations` include `'embedding'`; [examples/embed.py](examples/embed.py) finds one from the list.

### Images

```python
models = zharf.list().models
image_model = next(m.id for m in models if m.modality == 'text->image')

response = zharf.images.generate(image_model, 'A teapot on a Persian rug', size='1024x1024')
print(response.data[0].url or response.data[0].b64_json[:40])
```

### Speech

```python
tts = next(m for m in models if 'tts' in m.operations and 'mp3' in m.get('response_formats', []))

speech = zharf.audio.speech(tts.id, 'Hello from Zharf', voice=tts.voices[0], response_format='mp3')
speech.save('hello.mp3')  # bytes(speech) is the audio itself
```

### Transcription

```python
stt_model = next(m.id for m in models if 'stt' in m.operations)
print(zharf.audio.transcribe(stt_model, 'voice-note.ogg', language='fa').text)
```

`file` may be a path, bytes, an open file or a `(filename, bytes)` pair.

## Cost

Every response carries `cost_zharf`, what the request cost in ZHARF, and `usage` with the token counts. On a stream, both are on the final part. The platform's usage page shows the same figures.

## Errors

Errors are raised if requests return an error status or if an error is detected while streaming.

```python
import zharf

try:
  zharf.chat('does-not-exist', messages=[{'role': 'user', 'content': 'Hi'}])
except zharf.ResponseError as e:
  print('Error:', e.error)        # in English
  print('خطا:', e.message_fa)      # in Persian, ready to show to people
  print(e.status_code, e.code)    # 404 model_not_found
```

| Status | `code` | Meaning |
|---|---|---|
| 401 | `no_key`, `bad_key` | The key is missing, wrong or revoked |
| 402 | `insufficient_balance` | The account is out of credit; top it up from the wallet |
| 402 | `key_spend_limit` | This key reached its spending limit; raise it on the keys page |
| 403 | `model_not_offered` | The model is not available on the platform; the message says why |
| 404 | `model_not_found` | The model id is wrong; copy it from `list()` |
| 429 | `rate_limit`, `too_many_inflight` | Too many requests per minute, or at once, for this key |
| 503 | | The models are unavailable right now; try again shortly |

For 429 and 503, `e.retry_after` holds the seconds from the `Retry-After` header, when the platform sends one:

```python
import time

for attempt in range(3):
  try:
    response = zharf.chat('deepseek-v4-flash', messages=[{'role': 'user', 'content': 'Hi'}])
    break
  except zharf.ResponseError as e:
    if e.status_code not in (429, 503) or attempt == 2:
      raise
    time.sleep(e.retry_after or 2)
```

A request with bad arguments (no model, for example) raises `zharf.RequestError` before anything is sent. If the platform cannot be reached at all, you get `ConnectionError`; timeouts and dropped connections raise httpx's own exceptions (`httpx.TimeoutException`, `httpx.TransportError`).

## Privacy

The library sends nothing but your own requests, and only to the host you configure. Requests carry a `User-Agent` naming the library and Python versions. There is no telemetry.

## License

[MIT](LICENSE). Security issues: see [SECURITY.md](SECURITY.md).

## راهنمای فارسی

کتابخانه `zharf` ساده‌ترین راه برای کار با [پلتفرم ژرف](https://platform.zharf.ai) در پایتون است. متن‌باز است، روی پایتون 3.9 و بالاتر کار می‌کند و جز `httpx` به پکیج دیگری نیاز ندارد. نام تابع‌ها و ساختار پاسخ‌ها همان است که در کتابخانه پایتون Ollama دیده‌اید؛ اگر با آن کد نوشته‌اید، بیشترش با عوض کردن `import` و نام مدل اینجا هم کار می‌کند.

### نصب

```sh
pip install zharf
```

### کلید API

کلید را در صفحه کلیدهای پلتفرم بسازید. کلیدها با `zk-` شروع می‌شوند. اگر کلید را در متغیر محیطی `ZHARF_API_KEY` بگذارید، کتابخانه خودش آن را می‌خواند:

```sh
export ZHARF_API_KEY=zk-...
```

کلید را مستقیم هم می‌شود داد: `Client(api_key='zk-...')`. برای دیدن فهرست مدل‌ها و قیمت‌ها کلید لازم نیست.

### گفت‌وگو با مدل

```python
from zharf import chat

response = chat('deepseek-v4-flash', messages=[
  {'role': 'user', 'content': 'چرا آسمان آبی است؟'},
])
print(response.message.content)
print(response.cost_zharf)  # هزینه همین درخواست، به واحد ژرف
```

فیلدهای پاسخ را هم با نقطه می‌شود خواند و هم مثل دیکشنری؛ `response.message.content` و `response['message']['content']` همان مقدار را برمی‌گردانند. برای هر مدل گفت‌وگوی فهرست کافی است شناسه‌اش را در `model` بگذارید.

### استریم

با `stream=True` پاسخ تکه‌تکه می‌رسد و می‌توانید همان لحظه نشانش دهید. آخرین تکه `done=True` دارد و `cost_zharf` همین تکه آخر، هزینه کل درخواست است.

```python
for part in chat('deepseek-v4-flash', messages=[{'role': 'user', 'content': 'یک شعر کوتاه بگو'}], stream=True):
  print(part.message.content, end='', flush=True)
  if part.done:
    print('\nهزینه:', part.cost_zharf)
```

حروف فارسی سالم می‌رسند، حتی اگر بایت‌های یک حرف در دو تکه جدا برسند.

### تفکر، ابزار و خروجی JSON

- `think=True` حالت تفکر مدل را روشن می‌کند و آنچه مدل در این حالت می‌نویسد در `response.message.thinking` می‌آید. در مدل‌هایی که سطح تفکر دارند، `'low'` و `'medium'` و `'high'` را هم می‌شود داد.
- در `tools` می‌شود تابع‌های پایتون را مستقیم داد. کتابخانه از type hint و docstring هر تابع، توضیح ابزار را برای مدل می‌سازد. نمونه کامل در [examples/tools.py](examples/tools.py) است.
- با `format='json'` پاسخ به شکل JSON می‌آید، و اگر JSON Schema بدهید، پاسخ همان ساختار را دارد.

### مدل‌ها و قیمت‌ها

```python
import zharf

for model in zharf.list().models:
  print(model.id, model.pricing.prompt_per_1m_tokens, model.pricing.completion_per_1m_tokens)

print(zharf.show('deepseek-v4-flash').context_length)
```

قیمت مدل‌های متنی به ژرف و برای هر یک میلیون توکن است. مدل‌های تصویر و صدا و ویدیو برای هر واحد قیمت دارند (`per_unit` و `media_unit`). قیمت‌ها تا زمانی که در `prices_valid_until` آمده ثابت می‌مانند.

### خطاها

هر خطایی که پلتفرم برگرداند، به شکل `ResponseError` می‌رسد. پیام انگلیسی در `error` است و پیام فارسی در `message_fa`؛ پیام فارسی را می‌شود همان‌طور به کاربر نشان داد.

```python
import zharf

try:
  zharf.chat('deepseek-v4-flash', messages=[{'role': 'user', 'content': 'سلام'}])
except zharf.ResponseError as e:
  print(e.status_code, e.code)  # مثلاً 402 insufficient_balance
  print(e.message_fa)
```

| کد HTTP | `code` | یعنی |
|---|---|---|
| 401 | `no_key`، `bad_key` | کلید فرستاده نشده، اشتباه است یا باطل شده است |
| 402 | `insufficient_balance` | اعتبار حساب تمام شده است؛ حساب را از کیف پول شارژ کنید |
| 402 | `key_spend_limit` | این کلید به سقف هزینه‌اش رسیده است؛ سقف را در صفحه کلیدها بالا ببرید |
| 403 | `model_not_offered` | این مدل روی پلتفرم در دسترس نیست؛ دلیلش در پیام خطا آمده است |
| 404 | `model_not_found` | شناسه مدل درست نیست؛ شناسه درست را از فهرست مدل‌ها کپی کنید |
| 429 | `rate_limit`، `too_many_inflight` | تعداد درخواست‌ها از سقف کلید بیشتر شده است؛ چند ثانیه بعد دوباره بفرستید |
| 503 | | مدل‌ها فعلاً در دسترس نیستند؛ کمی بعد دوباره امتحان کنید |

در خطای 429 و 503، اگر پلتفرم زمان انتظار را فرستاده باشد، `e.retry_after` می‌گوید چند ثانیه صبر کنید.

### پروکسی و تنظیمات

- `Client(host=..., api_key=..., timeout=...)` کلاینت جداگانه‌ای می‌سازد. پارامترهای دیگر، مثل `proxy` یا `verify`، مستقیم به `httpx` داده می‌شوند.
- کتابخانه متغیرهای `HTTPS_PROXY` و `ALL_PROXY` را می‌خواند؛ اگر پشت پروکسی هستید، کافی است همین‌ها را تنظیم کنید. برای پروکسی SOCKS پکیج `httpx[socks]` را هم نصب کنید.
- `ZHARF_BASE_URL` آدرس API را عوض می‌کند. پیش‌فرض آن `https://platform.zharf.ai/v1` است.
- `AsyncClient` همین تابع‌ها را دارد، فقط باید آن‌ها را با `await` صدا بزنید. در حالت استریم، اول `await` کنید و بعد روی نتیجه `async for` بزنید.

### حریم خصوصی و امنیت

کتابخانه هیچ آماری از استفاده شما جمع نمی‌کند و جز درخواست‌های خودتان چیزی نمی‌فرستد، آن هم فقط به آدرسی که تعیین کرده‌اید.

اگر مشکل امنیتی پیدا کردید، لطفاً آن را در issue عمومی ننویسید و به security@zharf.ai ایمیل بزنید. جزئیات در [SECURITY.md](SECURITY.md) است.
