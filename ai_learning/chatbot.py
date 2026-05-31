from google import genai
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Connect to Gemini
client = genai.Client(api_key=API_KEY)

# Memory list
chat_history = []

print("=== AI Chatbot ===")
print("Type 'quit' to exit")
print("Type 'clear' to forget everything")
print("=" * 30)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    if user_input.lower() == "clear":
        chat_history = []
        print("Memory cleared!")
        continue

    if user_input.strip() == "":
        continue

    # Add to history
    chat_history.append(f"User: {user_input}")

    # Keep only last 6 messages
    if len(chat_history) > 6:
        chat_history = chat_history[-6:]

    # Build conversation
    conversation = "\n".join(chat_history)

    prompt = f"""You are a helpful assistant. 
Be concise and friendly.

Conversation so far:
{conversation}

Reply to the last User message."""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    ai_reply = response.text.strip()
    chat_history.append(f"AI: {ai_reply}")

    print(f"\nAI: {ai_reply}")