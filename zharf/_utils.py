"""Helpers: Python functions as tools, images as data URLs, and server-sent events."""

from __future__ import annotations

import collections.abc
import enum
import inspect
import os
import re
import typing
from base64 import b64decode, b64encode
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from zharf._types import Tool

try:  # Python 3.10+: `int | None`
  from types import UnionType  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
  UnionType = None  # type: ignore[assignment,misc]


# Python functions as tools

_SIMPLE_TYPES = {str: 'string', int: 'integer', float: 'number', bool: 'boolean'}
_NAMED_TYPES = {'str': 'string', 'int': 'integer', 'float': 'number', 'bool': 'boolean', 'list': 'array', 'dict': 'object'}
_ARRAY_ORIGINS = (list, tuple, set, frozenset, collections.abc.Sequence, collections.abc.Iterable, collections.abc.Set)
_OBJECT_ORIGINS = (dict, collections.abc.Mapping, collections.abc.MutableMapping)
_SECTION_HEADERS = ('returns', 'return', 'yields', 'raises', 'examples', 'example', 'note', 'notes', 'see also', 'attributes')
_ARG_LINE = re.compile(r'^\*{0,2}(\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$')


def _parse_docstring(doc: Optional[str]) -> Tuple[str, Dict[str, str]]:
  """
  Split a Google-style docstring into its description and argument descriptions.

  >>> _parse_docstring('''Add two numbers.
  ...
  ... Args:
  ...   a (int): The first number
  ...   b: The second
  ...     number
  ...
  ... Returns:
  ...   int: The sum
  ... ''')
  ('Add two numbers.', {'a': 'The first number', 'b': 'The second number'})
  """
  if not doc:
    return '', {}

  summary: List[str] = []
  args: Dict[str, str] = {}
  section = 'summary'
  current: Optional[str] = None
  arg_indent: Optional[int] = None

  for line in inspect.cleandoc(doc).splitlines():
    stripped = line.strip()
    lowered = stripped.lower()
    if lowered in ('args:', 'arguments:', 'parameters:', 'params:'):
      section, current, arg_indent = 'args', None, None
      continue
    if stripped.endswith(':') and lowered[:-1] in _SECTION_HEADERS:
      section = 'other'
      continue
    if section == 'summary':
      summary.append(stripped)
    elif section == 'args' and stripped:
      indent = len(line) - len(line.lstrip())
      match = _ARG_LINE.match(stripped)
      if match and (arg_indent is None or indent <= arg_indent):
        arg_indent = indent
        current = match.group(1)
        args[current] = match.group(3).strip()
      elif current:
        args[current] = f'{args[current]} {stripped}'.strip()

  return '\n'.join(summary).strip(), args


def _schema_for(annotation: Any) -> Tuple[Dict[str, Any], bool]:
  """
  JSON schema for a type annotation, and whether the annotation allows None.

  >>> _schema_for(int)
  ({'type': 'integer'}, False)
  >>> _schema_for(Optional[List[str]])
  ({'type': 'array', 'items': {'type': 'string'}}, True)
  >>> _schema_for(Literal['c', 'f'])
  ({'type': 'string', 'enum': ['c', 'f']}, False)
  """
  if annotation is inspect.Parameter.empty or annotation is Any:
    return {'type': 'string'}, False
  if isinstance(annotation, str):
    return {'type': _NAMED_TYPES.get(annotation.strip(), 'string')}, False
  if annotation is type(None):
    return {'type': 'null'}, True

  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)

  if origin is Union or (UnionType is not None and origin is UnionType):
    members = [a for a in args if a is not type(None)]
    optional = len(members) < len(args)
    if len(members) == 1:
      return _schema_for(members[0])[0], optional
    return {'anyOf': [_schema_for(a)[0] for a in members]}, optional

  if origin is Literal:
    values = list(args)
    first = type(values[0]) if values else str
    return {'type': _SIMPLE_TYPES.get(first, 'string'), 'enum': values}, False

  if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
    values = [member.value for member in annotation]
    first = type(values[0]) if values else str
    return {'type': _SIMPLE_TYPES.get(first, 'string'), 'enum': values}, False

  if annotation in _SIMPLE_TYPES:
    return {'type': _SIMPLE_TYPES[annotation]}, False

  if annotation in (list, tuple, set, frozenset) or origin in _ARRAY_ORIGINS:
    schema: Dict[str, Any] = {'type': 'array'}
    if args and args[0] is not Ellipsis:
      schema['items'] = _schema_for(args[0])[0]
    return schema, False

  if annotation is dict or origin in _OBJECT_ORIGINS:
    return {'type': 'object'}, False

  return {'type': 'string'}, False


def convert_function_to_tool(func: Callable[..., Any]) -> Tool:
  """
  Describe a Python function as a tool. Type hints give the parameter types and a
  Google-style docstring gives the descriptions.

  >>> def get_weather(city: str, unit: Optional[str] = None) -> str:
  ...   '''
  ...   Current weather in a city.
  ...
  ...   Args:
  ...     city: City name, such as Tehran
  ...     unit: 'c' or 'f'
  ...   '''
  ...   return ''
  >>> tool = convert_function_to_tool(get_weather)
  >>> tool.function.name, tool.function.description
  ('get_weather', 'Current weather in a city.')
  >>> tool.function.parameters['required']
  ['city']
  >>> tool.function.parameters['properties']['city']
  {'type': 'string', 'description': 'City name, such as Tehran'}
  """
  description, arg_docs = _parse_docstring(inspect.getdoc(func))
  try:
    hints = typing.get_type_hints(func)
  except Exception:
    hints = {}

  properties: Dict[str, Any] = {}
  required: List[str] = []
  for name, param in inspect.signature(func).parameters.items():
    if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
      continue
    schema, optional = _schema_for(hints.get(name, param.annotation))
    if arg_docs.get(name):
      schema = {**schema, 'description': arg_docs[name]}
    properties[name] = schema
    if param.default is inspect.Parameter.empty and not optional:
      required.append(name)

  return Tool(
    type='function',
    function=Tool.Function(
      name=func.__name__,
      description=description,
      parameters={'type': 'object', 'properties': properties, 'required': required},
    ),
  )


# Images

_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp', '.gif')


def _sniff_image(data: bytes) -> str:
  if data.startswith(b'\x89PNG'):
    return 'image/png'
  if data.startswith(b'\xff\xd8\xff'):
    return 'image/jpeg'
  if data.startswith(b'GIF8'):
    return 'image/gif'
  if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
    return 'image/webp'
  return 'image/png'


def image_url(value: Union[str, bytes, os.PathLike]) -> str:
  """
  An image as a URL the API accepts: http(s) and data URLs pass through; file paths,
  bytes and base64 strings become data URLs.

  >>> image_url('https://example.com/cat.png')
  'https://example.com/cat.png'
  >>> image_url(b'\\x89PNG\\r\\n')
  'data:image/png;base64,iVBORw0K'
  """
  if isinstance(value, (bytes, bytearray, memoryview)):
    data = bytes(value)
  elif isinstance(value, os.PathLike):
    data = Path(value).read_bytes()
  elif isinstance(value, str):
    if value.startswith(('http://', 'https://', 'data:')):
      return value
    try:
      path = Path(value)
      is_file = path.is_file()
    except (OSError, ValueError):  # a long base64 string is not a usable path
      is_file = False
    if is_file:
      data = path.read_bytes()
    elif value.lower().endswith(_IMAGE_SUFFIXES):
      raise ValueError(f'File {value} does not exist')
    else:
      encoded = re.sub(r'\s+', '', value)
      try:
        head = b64decode(encoded[:64] + '=' * (-len(encoded[:64]) % 4), validate=True)
      except ValueError:
        raise ValueError('Invalid image data, expected a file path, bytes, a URL or a base64 string') from None
      return f'data:{_sniff_image(head)};base64,{encoded}'
  else:
    raise TypeError(f'Unsupported image type: {type(value).__name__}')

  return f'data:{_sniff_image(data)};base64,{b64encode(data).decode()}'


# Audio files

def _sniff_audio(data: bytes) -> str:
  if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
    return 'wav'
  if data.startswith(b'OggS'):
    return 'ogg'
  if data.startswith(b'fLaC'):
    return 'flac'
  if data.startswith(b'\x1aE\xdf\xa3'):
    return 'webm'
  if data[4:8] == b'ftyp':
    return 'm4a'
  return 'mp3'


def file_part(file: Any) -> Tuple[str, bytes]:
  """A file argument as a (filename, bytes) pair for a multipart upload."""
  if isinstance(file, tuple):
    name, content = file[0], file[1]
    return str(name), content.read() if hasattr(content, 'read') else bytes(content)
  if isinstance(file, (str, os.PathLike)):
    path = Path(file)
    return path.name, path.read_bytes()
  if isinstance(file, (bytes, bytearray, memoryview)):
    data = bytes(file)
    return f'audio.{_sniff_audio(data)}', data
  if hasattr(file, 'read'):
    data = file.read()
    name = Path(str(getattr(file, 'name', '') or '')).name
    return name or f'audio.{_sniff_audio(data)}', data
  raise TypeError(f'Unsupported file type: {type(file).__name__}')


# Server-sent events

class SSEParser:
  """
  Turns the bytes of a server-sent event stream into the `data` of each event.

  Lines are split on bytes, before any decoding, so a character whose UTF-8 bytes
  arrive in two network chunks is never cut in half.

  >>> parser = SSEParser()
  >>> parser.feed('data: {"a": "سل'.encode()[:-1])
  []
  >>> parser.feed('data: {"a": "سلام"}\\n\\n'.encode()[len('data: {"a": "سل'.encode()) - 1:])
  ['{"a": "سلام"}']
  """

  def __init__(self) -> None:
    self._buffer = bytearray()
    self._data: List[str] = []

  def feed(self, chunk: bytes) -> List[str]:
    self._buffer.extend(chunk)
    events: List[str] = []
    while True:
      end = self._buffer.find(b'\n')
      if end < 0:
        break
      line = bytes(self._buffer[:end]).rstrip(b'\r').decode('utf-8', errors='replace')
      del self._buffer[: end + 1]
      self._line(line, events)
    return events

  def close(self) -> List[str]:
    events: List[str] = []
    if self._buffer:
      line = bytes(self._buffer).rstrip(b'\r').decode('utf-8', errors='replace')
      self._buffer.clear()
      self._line(line, events)
    self._line('', events)
    return events

  def _line(self, line: str, events: List[str]) -> None:
    if not line:
      if self._data:
        events.append('\n'.join(self._data))
        self._data = []
      return
    if line.startswith(':'):
      return
    field, _, value = line.partition(':')
    if field == 'data':
      self._data.append(value[1:] if value.startswith(' ') else value)
