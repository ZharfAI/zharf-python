import copy
import enum
import json
import pickle
from typing import Dict, List, Literal, Optional, Union

import pytest

from zharf import ChatResponse, ListResponse, Message, Options, ResponseError, Tool
from zharf._utils import _parse_docstring, convert_function_to_tool, image_url


def test_attribute_and_item_access():
  response = ChatResponse(
    {
      'model': 'm',
      'message': {'role': 'assistant', 'content': 'hi', 'tool_calls': [{'id': 'c1', 'function': {'name': 'f', 'arguments': '{"x": 1}'}}]},
      'done': True,
      'brand_new_field': {'a': 1},
    }
  )
  assert isinstance(response.message, Message)
  assert response.message.content == response['message']['content'] == 'hi'
  assert response.message.tool_calls[0].function.arguments == {'x': 1}
  assert response['message']['tool_calls'][0]['function']['name'] == 'f'
  assert response.brand_new_field == {'a': 1}
  assert response['brand_new_field'] == {'a': 1}

  # known but absent fields read as None; unknown names raise
  assert response.cost_zharf is None
  assert response['cost_zharf'] is None
  assert 'cost_zharf' not in response
  assert 'message' in response
  with pytest.raises(AttributeError):
    _ = response.not_a_field
  with pytest.raises(KeyError):
    _ = response['not_a_field']


def test_setting_and_converting():
  message = Message(role='user')
  message['content'] = 'سلام'
  message.images = ['https://example.com/a.png']
  message['tool_calls'] = [{'function': {'name': 'f', 'arguments': {}}}]
  assert isinstance(message.tool_calls[0], Message.ToolCall)
  assert message.to_dict() == {
    'role': 'user',
    'content': 'سلام',
    'images': ['https://example.com/a.png'],
    'tool_calls': [{'function': {'name': 'f', 'arguments': {}}}],
  }
  assert message.model_dump() == message.to_dict()
  assert dict(message)['content'] == 'سلام'
  assert json.loads(json.dumps(message.to_dict(), ensure_ascii=False))['content'] == 'سلام'
  assert message == {'role': 'user', 'content': 'سلام', 'images': ['https://example.com/a.png'], 'tool_calls': [{'function': {'name': 'f', 'arguments': {}}}]}
  assert message.get('thinking', 'none') == 'none'
  assert 'Message(' in repr(message)


def test_copy_and_pickle():
  response = ListResponse(models=[{'id': 'a', 'pricing': {'currency': 'ZHARF', 'per_unit': 2.0}}])
  for clone in (copy.deepcopy(response), pickle.loads(pickle.dumps(response))):
    assert clone == response
    assert clone.models[0].pricing.per_unit == 2.0


def test_tool_call_arguments_that_are_not_json():
  call = Message.ToolCall(function={'name': 'f', 'arguments': 'not json'})
  assert call.function.arguments == 'not json'
  assert Message.ToolCall(function={'name': 'f', 'arguments': ''}).function.arguments == {}


def test_options_object():
  options = Options(temperature=0.5, num_predict=10)
  assert options.temperature == 0.5
  assert options['num_predict'] == 10
  assert options.seed is None


def test_response_error():
  e = ResponseError('Too many requests.', 429, message_fa='درخواست‌ها زیاد است.', code='rate_limit', type='rate_limit_error', retry_after=3.0)
  assert str(e) == 'Too many requests. (status code: 429)'
  assert (e.status_code, e.code, e.type, e.retry_after) == (429, 'rate_limit', 'rate_limit_error', 3.0)
  assert ResponseError('boom').status_code == -1


# tools from functions


class Unit(enum.Enum):
  C = 'c'
  F = 'f'


def forecast(
  city: str,
  days: int,
  unit: Unit,
  hourly: bool = False,
  fields: Optional[List[str]] = None,
  extra: Dict[str, str] = None,  # type: ignore[assignment]  # noqa: RUF013 (implicit Optional on purpose)
  mode: Literal['fast', 'full'] = 'fast',
  threshold: Union[int, float] = 0,
  note=None,
  *args,
  **kwargs,
) -> str:
  """
  Weather forecast for a city.

  Uses the nearest station.

  Args:
    city: City name, such as Shiraz
    days (int): How many days,
      from 1 to 7
    unit: Temperature unit
    hourly: Hour by hour
    fields: Which fields to return

  Returns:
    str: The forecast
  """
  return ''


def test_convert_function_to_tool():
  tool = convert_function_to_tool(forecast)
  assert isinstance(tool, Tool)
  assert tool.type == 'function'
  assert tool.function.name == 'forecast'
  assert tool.function.description == 'Weather forecast for a city.\n\nUses the nearest station.'
  parameters = tool.function.parameters
  assert parameters['type'] == 'object'
  assert parameters['required'] == ['city', 'days', 'unit']
  properties = parameters['properties']
  assert list(properties) == ['city', 'days', 'unit', 'hourly', 'fields', 'extra', 'mode', 'threshold', 'note']
  assert properties['city'] == {'type': 'string', 'description': 'City name, such as Shiraz'}
  assert properties['days'] == {'type': 'integer', 'description': 'How many days, from 1 to 7'}
  assert properties['unit'] == {'type': 'string', 'enum': ['c', 'f'], 'description': 'Temperature unit'}
  assert properties['hourly'] == {'type': 'boolean', 'description': 'Hour by hour'}
  assert properties['fields'] == {'type': 'array', 'items': {'type': 'string'}, 'description': 'Which fields to return'}
  assert properties['extra'] == {'type': 'object'}
  assert properties['mode'] == {'type': 'string', 'enum': ['fast', 'full']}
  assert properties['threshold'] == {'anyOf': [{'type': 'integer'}, {'type': 'number'}]}
  assert properties['note'] == {'type': 'string'}
  json.dumps(tool.to_dict())


def test_convert_function_without_docstring():
  def ping(host):
    return host

  tool = convert_function_to_tool(ping)
  assert tool.function.description == ''
  assert tool.function.parameters == {'type': 'object', 'properties': {'host': {'type': 'string'}}, 'required': ['host']}


def test_parse_docstring_sections():
  description, args = _parse_docstring(
    """
    Do a thing.

    Args:
      a: first
      b (str): second
        continues: with a colon

    Raises:
      ValueError: never
    """
  )
  assert description == 'Do a thing.'
  assert args == {'a': 'first', 'b': 'second continues: with a colon'}


# images


def test_image_url(tmp_path):
  jpeg = tmp_path / 'photo.jpg'
  jpeg.write_bytes(b'\xff\xd8\xff\xe0fake')
  assert image_url(str(jpeg)).startswith('data:image/jpeg;base64,')
  assert image_url(jpeg).startswith('data:image/jpeg;base64,')
  assert image_url(b'RIFF0000WEBPVP8 ').startswith('data:image/webp;base64,')
  assert image_url('data:image/png;base64,AAAA') == 'data:image/png;base64,AAAA'
  assert image_url('iVBORw0KGgo=') == 'data:image/png;base64,iVBORw0KGgo='
  with pytest.raises(ValueError):
    image_url(str(tmp_path / 'missing.png'))
  with pytest.raises(ValueError):
    image_url('not base64 at all!')
