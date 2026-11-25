from zharf import ListResponse, list

# Listing models needs no API key
response: ListResponse = list()

print(f'{len(response.models)} models; prices hold until {response.prices_valid_until}\n')

for model in response.models:
  pricing = model.pricing
  if pricing.per_unit:
    price = f'{pricing.per_unit} {pricing.currency} per {pricing.media_unit}'
  else:
    price = f'{pricing.prompt_per_1m_tokens} in, {pricing.completion_per_1m_tokens} out ({pricing.currency} per 1M tokens)'
  print(f'{model.id:48} {model.modality or "":28} {price}')
