import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API")

try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test prompt"}],
        temperature=0.7,
    )
    print("Response:", response.choices[0].message["content"].strip())
except Exception as e:
    print("OpenAI API call failed:", e)
