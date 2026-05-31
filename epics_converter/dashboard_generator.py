"""
Dashboard Generator
Reads pv_analysis.txt and generates
a complete PyQt6 dashboard automatically!
"""
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

def read_pv_analysis(filepath="pv_analysis.txt"):
    """Read the AI analysis results"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    with open(filepath, 'r') as f:
        return f.read()

def generate_dashboard(pv_analysis):
    """Ask Gemini to generate dashboard code"""

    prompt = f"""You are an expert Python developer 
specializing in EPICS control system displays.

I have analyzed an EPICS EDL file and found these PVs:

{pv_analysis}

Please generate a complete Python dashboard using tkinter that:

1. Shows KEY METRICS at the top
   (beam current, energy, lifetime etc.)

2. Shows CELL STATUS as a grid
   (12 cells × 5 status types = colored indicators)
   Green = OK (value=1), Red = alarm (value=0)

3. Shows INTERLOCKS as colored indicators
   Green = OK, Red = fault

4. Shows HEARTBEATS as animated bars

5. Shows BPM data as small bar charts

6. Has a SIMULATOR built in
   (no real EPICS needed, uses random data)

7. Uses this dark color scheme:
   Background: #07090f
   Panel: #0d1120
   Green: #16A34A
   Red: #DC2626
   Gold: #F5C518
   Cyan: #22D3EE

8. Updates every 500ms

Important:
-  Use PyQt6 for the UI
- Include simulator with random data (no epics needed)
- Include complete simulator with random data
- Make it runnable immediately
- Add comments explaining each section

Generate COMPLETE runnable Python code only.
No explanations outside the code.
"""

    print("Asking Gemini to generate dashboard...")
    print("This might take a moment...\n")

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text

def save_dashboard(code, output="generated_dashboard.py"):
    """Save generated code to file"""
    # Clean up markdown if present
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    with open(output, 'w',
              encoding='utf-8') as f:
        f.write(code)

    print(f"✅ Dashboard saved to {output}")
    return output

def main():
    print("=== Dashboard Generator ===\n")

    # Read PV analysis
    print("Reading PV analysis...")
    pv_analysis = read_pv_analysis()
    if not pv_analysis:
        print("Run edl_ai_converter.py first!")
        return

    print("PV analysis loaded! ✅\n")

    # Generate dashboard
    code = generate_dashboard(pv_analysis)

    # Save to file
    output = save_dashboard(code)

    print(f"\n=== Done! ===")
    print(f"Run your dashboard:")
    print(f"py {output}")

if __name__ == "__main__":
    main()