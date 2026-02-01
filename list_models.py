from google import genai

client = genai.Client(api_key="AIzaSyCIqZOyW5eiuREHRD5eibiMjMLVTSPcRkc")

for model in client.models.list():
    print(model.name)