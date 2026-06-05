"""
EDL to Python Converter - Generic Version
Converts ANY .edl file(s) to a PyQt6 monitoring dashboard
Usage:
    py edl_to_python.py                        # auto-finds all .edl files
    py edl_to_python.py myfile.edl             # single file
    py edl_to_python.py file1.edl file2.edl    # multiple files
"""

import os
import sys
import re
import glob
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# All known EDL widget PV attributes
PV_PATTERNS = [
    r'controlPv\s+"([^"]+)"',
    r'visPv\s+"([^"]+)"',
    r'indicatorPv\s+"([^"]+)"',
    r'readPv\s+"([^"]+)"',
    r'writePv\s+"([^"]+)"',
    r'alarmPv\s+"([^"]+)"',
    r'pvName\s+"([^"]+)"',
    r'pv\s+"([^"]+)"',
    r'xPv\s+"([^"]+)"',
    r'yPv\s+"([^"]+)"',
    r'triggerPv\s+"([^"]+)"',
    r'resetPv\s+"([^"]+)"',
    r'colorPv\s+"([^"]+)"',
    r'nullPv\s+"([^"]+)"',
]


def find_edl_files(paths):
    """
    Accept a list of paths from command line.
    If empty, auto-find all .edl files in current directory.
    """
    if not paths:
        found = glob.glob("*.edl")
        if not found:
            print("No .edl files found in current directory.")
            sys.exit(1)
        print(f"Auto-detected {len(found)} .edl file(s): {found}")
        return found
    
    valid = []
    for p in paths:
        if not os.path.exists(p):
            print(f"Warning: file not found, skipping: {p}")
        else:
            valid.append(p)
    
    if not valid:
        print("No valid .edl files provided.")
        sys.exit(1)
    
    return valid


def extract_pvs_from_edl(filepath):
    """Extract all PV names from a single EDL file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    pvs = set()
    for pattern in PV_PATTERNS:
        for pv in re.findall(pattern, content):
            pv = pv.strip()
            # Filter out empty, too short, or macro-only values like $(P)
            if pv and len(pv) > 2 and not pv.startswith("#"):
                pvs.add(pv)

    return sorted(pvs)


def extract_all_pvs(edl_files):
    """Extract PVs from all files, return combined sorted list."""
    all_pvs = set()
    for filepath in edl_files:
        pvs = extract_pvs_from_edl(filepath)
        print(f"  {filepath}: {len(pvs)} PVs found")
        all_pvs.update(pvs)
    return sorted(all_pvs)


def generate_python_code(pvs, source_label):
    """Send PVs to Gemini and get a PyQt6 dashboard back."""

    # Send up to 150 PVs — enough for good analysis without hitting token limits
    pv_sample = pvs[:150]
    pv_list_str = "\n".join(pv_sample)
    total = len(pvs)
    shown = len(pv_sample)

    prompt = f"""You are an EPICS control system expert and Python developer.

I have EDL file(s) from: {source_label}
Total unique PVs found: {total} (showing first {shown} below)

PV LIST:
{pv_list_str}

TASK:
1. Analyze the PV names and determine:
   - What type of facility or system this is (accelerator, vacuum, RF, etc.)
   - Logical groupings (e.g. beam, vacuum, magnet, RF, temperature)
   - Which PVs are most critical for operators

2. Generate a complete, runnable PyQt6 dashboard that:
   - Has a dark professional theme (dark background, bright indicators)
   - Shows a title bar with facility name and live clock
   - Groups PVs into labeled sections/tabs based on their category
   - Uses colored LED indicators: green=OK, red=ALARM, yellow=WARNING
   - Displays numeric values with units where obvious from PV name
   - Has a SIMULATOR MODE that generates realistic random data (for testing without EPICS)
   - Updates every 500ms
   - Works on any EPICS facility, not just PAL

IMPORTANT RULES:
- Output ONLY Python code, no explanation, no markdown
- The code must run with: py generated_monitor.py
- Use PyQt6 only (not PyQt5)
- Start with: import sys
- Simulator mode should be ON by default so it runs without real EPICS
"""

    print(f"Sending {shown} PVs to Gemini AI...")
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text


def save_code(code, output_file):
    """Clean and save the generated code."""
    code = code.strip()

    # Remove markdown code fences if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    code = code.strip()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(code)

    print(f"Saved to: {output_file}")


def main():
    # Get EDL file paths from command line, or auto-detect
    input_paths = sys.argv[1:]
    edl_files = find_edl_files(input_paths)

    output_file = "generated_monitor.py"
    source_label = ", ".join(edl_files)

    print(f"\n=== EDL to Python Converter ===")
    print(f"Input files: {source_label}\n")

    print("Extracting PVs...")
    pvs = extract_all_pvs(edl_files)
    print(f"Total unique PVs: {len(pvs)}\n")

    if len(pvs) == 0:
        print("No PVs found. Check that your .edl files have PV attributes.")
        sys.exit(1)

    code = generate_python_code(pvs, source_label)
    save_code(code, output_file)

    print(f"\nDone! Run your dashboard:")
    print(f"  py {output_file}")


if __name__ == "__main__":
    main()