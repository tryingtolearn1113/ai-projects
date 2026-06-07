"""
EDL to Python Converter
Converts .edl files to a Python monitoring dashboard.
AI learns from your existing reference code style.

Usage:
    py edl_to_python.py                            # auto-finds .edl files
    py edl_to_python.py file.edl                   # single file
    py edl_to_python.py file1.edl file2.edl        # multiple files
    py edl_to_python.py --ref mycode.py file.edl   # learn from your code
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

# All known EDL PV attribute patterns
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


def parse_args(argv):
    """
    Parse command line arguments.
    Separates --ref flag from EDL file paths.
    Returns (ref_file, edl_paths)
    """
    ref_file = None
    edl_paths = []

    i = 0
    while i < len(argv):
        if argv[i] == "--ref":
            # Next argument is the reference file path
            if i + 1 < len(argv):
                ref_file = argv[i + 1]
                i += 2
            else:
                print("Error: --ref needs a filename after it.")
                sys.exit(1)
        else:
            edl_paths.append(argv[i])
            i += 1

    return ref_file, edl_paths


def load_ref_file(ref_path):
    """
    Read the reference Python file.
    This is the code AI will learn style from.
    """
    if not os.path.exists(ref_path):
        print(f"Warning: reference file not found: {ref_path}")
        return None

    with open(ref_path, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()

    print(f"Reference file loaded: {ref_path} ({len(code)} characters)")
    return code


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

def parse_block(widget_type, block_lines):
    """Extract useful info from one widget block."""
    info = {"type": widget_type}
    
    for line in block_lines:
        # Extract PV names
        for pattern in PV_PATTERNS:
            match = re.search(pattern, line)
            if match:
                info["pv"] = match.group(1)
        
        # Extract title
        if line.startswith("graphTitle") or line.startswith("title"):
            info["title"] = line.split('"')[1] if '"' in line else ""
        
        # Extract min/max
        if line.startswith("min "):
            info["min"] = line.split()[1]
        if line.startswith("max "):
            info["max"] = line.split()[1]
    
    # Only return blocks that have a PV
    if "pv" in info:
        return info
    return None

def format_widget_summary(widgets):
    """Convert widget list into clean text for Gemini prompt."""
    lines = []
    for w in widgets:
        # Clean up type name
        wtype = w["type"].replace("object ", "").strip()
        
        parts = [wtype, w["pv"]]
        
        if "title" in w:
            parts.append(f"title: {w['title']}")
        if "min" in w:
            parts.append(f"min: {w['min']}")
        if "max" in w:
            parts.append(f"max: {w['max']}")
            
        lines.append(" | ".join(parts))
    
    return "\n".join(lines)

def extract_widget_info(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    widgets = []
    current_block = None
    widget_type = ""
    
    for i, line in enumerate(lines):
        if "beginObjectProperties" in line:
            widget_type = lines[i-1].strip()
            current_block = []
        elif "endObjectProperties" in line:
            if current_block is not None:
                result = parse_block(widget_type, current_block)
                if result:
                    widgets.append(result)
                current_block = None
        elif current_block is not None:
            current_block.append(line.strip())
    
    return widgets


def generate_python_code(widgets, source_label, framework="tkinter"):
    """Send widget info to Gemini and get a Python dashboard back."""
    pv_list_str = format_widget_summary(widgets)
    total = len(widgets)

    # Build the reference section of the prompt
    if ref_code:
        # Send first 300 lines of reference - enough for style without hitting limits
        ref_lines = ref_code.splitlines()[:150]
        ref_section = f"""
REFERENCE CODE (learn from this style):
The user has already built a working dashboard. 
Study its structure, color scheme, layout, and coding patterns.
Follow the same style when generating new code.

```python
{chr(10).join(ref_lines)}
```
"""
    else:
        ref_section = ""

    prompt = f"""You are an EPICS control system expert and Python developer.

I have EDL file(s) from: {source_label}
Total widgets found: {total}

WIDGET LIST (ALL {total} widgets with context):
format: widget_type | pv_name | title | min | max
{pv_list_str}
{ref_section}
TASK:
1. Analyze the PV names and determine:
   - What type of facility or system this is
   - Logical groupings (beam, vacuum, RF, magnet, BPM, etc.)
   - Which PVs are most critical for operators

2. Generate a complete, runnable Python dashboard that:
   - Follows the same style as the reference code if provided
   - Has a dark professional theme
   - Uses tk.PanedWindow for resizable, draggable panels between sections
   - Shows a title bar with facility name and live clock
   - Groups PVs into labeled sections based on category
   - Uses colored indicators: green=OK, red=ALARM, yellow=WARNING
   - Has SIMULATOR MODE on by default (runs without real EPICS)
   - Updates every 500ms

IMPORTANT RULES:
- Output ONLY Python code, no explanation, no markdown backticks
- The code must be complete and runnable
- Start with: import sys
- - Use {framework} for the UI framework..
- Simulator must be ON by default
"""

    print(f"Sending {total} widgets to Gemini AI...")
    if ref_code:
        print("(Using your reference code as style guide)")

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
    # Parse arguments - separate --ref from EDL files
    ref_file, input_paths = parse_args(sys.argv[1:])

    # Load reference code if provided
    ref_code = None
    if ref_file:
        ref_code = load_ref_file(ref_file)

    # Find EDL files
    edl_files = find_edl_files(input_paths)
    output_file = "generated_monitor.py"
    source_label = ", ".join(edl_files)

    print(f"\n=== EDL to Python Converter ===")
    print(f"Input files: {source_label}")
    if ref_file:
        print(f"Reference:   {ref_file}")
    print()

    
    
    print("Extracting widget info...")
    widgets = []
    for edl_file in edl_files:
        file_widgets = extract_widget_info(edl_file)
        widgets.extend(file_widgets)
        print(f"  {edl_file}: {len(file_widgets)} widgets found")

    print(f"Total widgets: {len(widgets)}\n")

    if len(widgets) == 0:
        print("No widgets found. Check that your .edl files have PV attributes.")
        sys.exit(1)
        
        # Ask user to choose framework
    print("Choose output framework:")
    print("  1. tkinter (default, no install needed)")
    print("  2. PyQt6 (modern look, requires pip install PyQt6)")
    choice = input("Enter 1 or 2 (default=1): ").strip()
    framework = "PyQt6" if choice == "2" else "tkinter"
    print(f"Using: {framework}\n")


    code = generate_python_code(widgets, source_label, ref_code)

    save_code(code, output_file)

    print(f"\nDone! Run your dashboard:")
    print(f"  py {output_file}")
    if not ref_file:
        print(f"\nTip: next time use --ref to teach AI your coding style:")
        print(f"  py edl_to_python.py --ref pls_monitor_end2.py SurvLeft.edl")




if __name__ == "__main__":
    main()