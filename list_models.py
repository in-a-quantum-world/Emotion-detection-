from google import genai

client = genai.Client(api_key="API KEY")

for model in client.models.list():

    print(model.name)
