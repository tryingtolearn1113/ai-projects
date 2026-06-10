import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)


def fetch_webpage(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url,
                                headers=headers,
                                timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for tag in soup(['script', 'style']):
            tag.decompose()
            
        text = soup.get_text(separator='\n')
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return '\n'.join(lines)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def generate_qa_pairs(text, num_pairs=5):
    prompt = f"""Read this EPICS documentation and generate {num_pairs} question and answer pairs.
    
Return ONLY a JSON array like this:
[
  {{"input": "question here", "output": "answer here"}},
  {{"input": "question here", "output": "answer here"}}
]

No explanation. Only JSON.

Documentation:
{text[:2000]}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    try:
        text_response = response.text.strip()
        if "```json" in text_response:
            text_response = text_response.split("```json")[1].split("```")[0]
        elif "```" in text_response:
            text_response = text_response.split("```")[1].split("```")[0]
        pairs = json.loads(text_response)
        return pairs
    except:
        print("Failed to parse response")
        return []
    
def main():
    urls = [
        "https://docs.epics-controls.org/en/latest/guides/EPICS_Intro.html",
        "https://pyepics.github.io/pyepics/overview.html",
        "https://pyepics.github.io/pyepics/pv.html",
        "https://pyepics.github.io/pyepics/ca.html",
        "https://pyepics.github.io/pyepics/epics_device.html",
    ]
    
    all_pairs = []
    
    for url in urls:
        print(f"Fetching: {url}")
        text = fetch_webpage(url)
        if not text:
            continue
        
        print(f"Generating Q&A pairs...")
        pairs = generate_qa_pairs(text, num_pairs=10)
        all_pairs.extend(pairs)
        print(f"Got {len(pairs)} pairs")
    
    print(f"\nTotal pairs: {len(all_pairs)}")
    
    with open("training_data.json", "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    
    print("Saved to training_data.json")

if __name__ == "__main__":
    main()
