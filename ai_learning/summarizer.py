from google import genai
from dotenv import load_dotenv
import os

# Load keys from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Connect to Gemini
client = genai.Client(api_key=API_KEY)

# Your text to summarize
text = """
Artificial intelligence is transforming 
the way we work and learn. 
Python is one of the most popular 
programming languages for AI development.
"""

# Ask AI to summarize
prompt = f"Summarize this in 3 bullet points:\n\n{text}"

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)

print("=== Gemini Summary ===")
print(response.text)