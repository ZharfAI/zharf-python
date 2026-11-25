# Examples

Run any example from the repository root after setting your key:

```sh
export ZHARF_API_KEY=zk-...
python3 examples/chat.py
```

- [chat.py](chat.py): a single chat request, and what it cost
- [chat-stream.py](chat-stream.py): stream the answer as it is written
- [async-chat.py](async-chat.py): the same with `AsyncClient`
- [tools.py](tools.py): let the model call Python functions
- [embed.py](embed.py): embeddings for a list of texts
- [list.py](list.py): every model with its prices (needs no key)

The examples use `deepseek-v4-flash`; any chat model id from `list.py` works in its place.
