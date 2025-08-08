import ollama

response = ollama.chat(model='mistral', messages=[
  {'role': 'user', 'content': 'Explain LCA in 3 sentences.'}
])
print(response['message']['content'])