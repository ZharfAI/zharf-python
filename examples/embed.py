import zharf

# Pick a model that makes embeddings from the model list
model = next(m.id for m in zharf.list().models if 'embedding' in (m.operations or []))

response = zharf.embed(
  model=model,
  input=[
    'The sky is blue because of Rayleigh scattering',
    'آسمان آبی است، چون نور آبی بیشتر از رنگ‌های دیگر پخش می‌شود',
  ],
)

print(len(response.embeddings), 'embeddings of', len(response.embeddings[0]), 'numbers each')
print(f'Cost: {response.cost_zharf} ZHARF')
