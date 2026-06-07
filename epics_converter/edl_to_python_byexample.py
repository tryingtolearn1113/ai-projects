"""
EDL to Python Converter — By Example
Converts any .edl file to a Python monitoring dashboard.
Style is learned from a real working dashboard (pls_monitor_end2.py).
No reference file needed — style is hardcoded from the example.

Usage:
    py edl_to_python_byexample.py                          # auto-finds .edl files
    py edl_to_python_byexample.py file.edl                 # single file
    py edl_to_python_byexample.py file1.edl file2.edl      # multiple files
    py edl_to_python_byexample.py --ref extra.py file.edl  # add extra reference
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

# ═══════════════════════════════════════════════════════════════
# STYLE GUIDE — extracted from pls_monitor_end2.py
# Applied to every conversion automatically, no file needed
# ═══════════════════════════════════════════════════════════════
STYLE_GUIDE = """
STYLE GUIDE (always follow exactly — extracted from a real working EPICS dashboard):

== COLORS (use these exact hex values) ==
BG='#07090f', PANEL='#0d1120', PANEL2='#0f1525', BORDER='#1a2540'
HDR='#0a1530', GOLD='#F5C518', CYAN='#22D3EE', LIME='#4ADE80'
WHITE='#E2E8F0', GRAY='#4B5563', DIM='#1F2937'
GREEN='#16A34A', RED='#DC2626', YELLOW='#CA8A04'
BLUE='#2563EB', AMBER='#D97706', PINK='#DB2777'
TEAL='#2DD4BF', LBLUE='#60A5FA'

== ARCHITECTURE — use these exact classes ==

1. PVManager class:
   - self.values = {}
   - self.history = defaultdict(lambda: deque(maxlen=1200))
   - self._dirty = set()  — only update widgets when values change
   - self._lock = threading.Lock()  — thread safety for EPICS callbacks
   - Methods: get(pv), get_history(pv, n), take_dirty(), connect_all(pvs), _cb(callback)
   - Simulator: separate daemon thread, updates every 0.5s
     use math.sin/cos for slow drift + random.uniform for noise
     binary PVs mostly 1 (OK), rarely flip to 0
     heartbeat PVs increment by 1, wrap at 10000

2. SectionFrame class (panel with scrollable title bar):
   - PANEL background, BORDER highlight 1px
   - Title bar: HDR background, 3px accent color strip on left, Consolas 7 bold
   - Optional scrollable body: canvas + scrollbar + mousewheel binding
   - Usage: SectionFrame(parent, "TITLE", accent=CYAN, scrollable=True)

3. MiniPlot class (auto-resizing trend chart):
   - Canvas-based, fills parent, bind Configure for resize
   - Background '#050810', filled polygon + smooth line on top
   - GRAY grid lines (dashed), GOLD last value top-right corner
   - Y-axis labels in GRAY Consolas 5

4. DockablePanel class (floating, draggable, resizable):
   - Drag by title bar (cursor='fleur'), binds ButtonPress-1 and B1-Motion
   - Resize grip bottom-right corner triangle (cursor='sizing')
   - Auto-scales all child fonts on resize using font registry
   - Saves layout to ~/.dashboard_layout.json on window close
   - Ctrl+R resets layout to DEFAULT_LAYOUT

5. HeartbeatWidget (inline, fits in subheader bar):
   - LED canvas oval: GREEN when alive, fades to YELLOW when stalled
   - EKG mini-canvas shows value history as line
   - tick() method called every 500ms

6. BPMStation widget:
   - Compact: label + X bar canvas + Y bar canvas + numeric labels
   - Bar shows deflection from center, RED if > threshold
   - TEAL for X, LBLUE for Y

== LAYOUT ==
- Header bar: HDR background, GOLD bold title left, CYAN clock Consolas 11 bold right
  Lightning bolt emoji before title, CYAN 1px underline strip below header
- Subheader bar: PANEL background, sections divided by BORDER lines
  Contains: Tune & Beam | Injection beam params | Heartbeats — all in one row
- Main workspace: tk.Frame with all DockablePanel instances placed with .place(x=,y=)
- DEFAULT_LAYOUT dict: defines x, y, w, h for each panel name at 1920x1080

== UPDATE PATTERN (critical — always use this pattern) ==
- 500ms tick: root.after(500, self._tick)
- self._tick calls self._tick_inner() inside try/except to prevent crashes
- dirty = pvm.take_dirty() — atomically get changed PVs
- Only update widgets whose PV is in dirty set — never update everything every tick
- For numeric labels: compare new text to last text, skip .config() if same
- Heartbeats tick every cycle regardless of dirty set

== WIDGET PATTERNS ==
- Key metric: GRAY Consolas 6 label → LIME Consolas 16 bold value → DIM unit label
  all in a vertical Frame, separated by BORDER 1px dividers
- Status/interlock indicator: tk.Label, bg=GREEN if value==1 else RED, bg=DIM if None
  width=3-7, no border, relief='flat'
- Shutter: tk.Canvas rectangle 26x9, GREEN='OPN' or RED='CLS' text in BG color Consolas 5
- RF LED: small Canvas oval 7x7, GREEN if abs(value)<threshold else YELLOW
- Scrollable list (ID gaps, shutters, RF): Canvas + Scrollbar + mousewheel binding
  inner Frame holds rows, each row is label + value label packed LEFT/RIGHT

== FONTS (always Consolas, always monospace) ==
- 5: axis tick labels
- 6: small labels, units, secondary info
- 7: section titles, subheader labels
- 8-9: normal values, BPM labels
- 11-12: header clock and title
- 16: key metrics values
- Always bold for values, titles, and status text

== CODE STRUCTURE (follow this order exactly) ==
1. imports (os, time, json, math, threading, logging, collections, tkinter, epics)
2. logging setup
3. Color constants
4. PV definition lists and dicts (KEY_METRICS, INTERLOCKS, etc.)
5. PVManager class
6. Widget classes: SectionFrame, MiniPlot, DockablePanel, HeartbeatWidget, BPMStation
7. DEFAULT_LAYOUT dict
8. Dashboard class with _build(), _tick(), panel content methods
9. if __name__ == "__main__": block with simulator/online mode selection

== SIMULATOR MODE ==
- Default: simulate=True
- Command line: python script.py --online switches to real EPICS
- Print clear INFO message showing which mode is active
- try: import epics / except ImportError: EPICS_AVAILABLE = False
"""

# ═══════════════════════════════════════════════════════════════
# EDL PV ATTRIBUTE PATTERNS
# ═══════════════════════════════════════════════════════════════
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
    """Parse --ref flag and EDL file paths from command line arguments."""
    ref_file = None
    edl_paths = []
    i = 0
    while i < len(argv):
        if argv[i] == "--ref":
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
    """Load an optional extra reference file."""
    if not os.path.exists(ref_path):
        print(f"Warning: reference file not found: {ref_path}")
        return None
    with open(ref_path, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()
    print(f"Extra reference loaded: {ref_path} ({len(code)} characters)")
    return code


def find_edl_files(paths):
    """Find EDL files from paths or auto-detect in current directory."""
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
    """Extract useful info from one EDL widget block."""
    info = {"type": widget_type}
    for line in block_lines:
        for pattern in PV_PATTERNS:
            match = re.search(pattern, line)
            if match:
                info["pv"] = match.group(1)
        if line.startswith("graphTitle") or line.startswith("title "):
            info["title"] = line.split('"')[1] if '"' in line else ""
        if line.startswith("min "):
            try: info["min"] = line.split()[1]
            except: pass
        if line.startswith("max "):
            try: info["max"] = line.split()[1]
            except: pass
        if line.startswith("label "):
            info["label"] = line.split('"')[1] if '"' in line else ""
    if "pv" in info:
        return info
    return None


def extract_widget_info(filepath):
    """Parse EDL file block by block and extract widget context."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    widgets = []
    current_block = None
    widget_type = ""
    for i, line in enumerate(lines):
        if "beginObjectProperties" in line:
            widget_type = lines[i-1].strip() if i > 0 else ""
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


def compress_repeated_pvs(widgets):
    """
    Detect repeated PV patterns and compress them.
    Example: SR:MISC01CPST, SR:MISC02CPST ... SR:MISC12CPST
    becomes: SR:MISC01~12CPST (count: 12)
    """
    # Group widgets by PV pattern (replace numbers with placeholder)
    import re as _re
    groups = {}
    for w in widgets:
        pv = w["pv"]
        # Replace all numbers with {N}
        pattern = _re.sub(r'\d+', '{N}', pv)
        if pattern not in groups:
            groups[pattern] = []
        groups[pattern].append(w)

    compressed = []
    for pattern, group in groups.items():
        if len(group) == 1:
            compressed.append(group[0])
        else:
            # Extract all numbers from PVs
            nums = []
            for w in group:
                found = _re.findall(r'\d+', w["pv"])
                if found:
                    nums.append(int(found[0]))
            nums.sort()
            # Build compressed summary
            base = group[0].copy()
            if nums:
                base["pv"] = f"{pattern.replace('{N}', str(nums[0]).zfill(len(str(nums[0]))))}~{nums[-1]} (repeated {len(group)}x)"
            base["compressed"] = len(group)
            compressed.append(base)

    return compressed


def format_widget_summary(widgets):
    """Convert widget list into structured text for the Gemini prompt."""
    lines = []
    for w in widgets:
        wtype = w["type"].replace("object ", "").strip()
        parts = [wtype, w["pv"]]
        if "title" in w and w["title"]:
            parts.append(f"title:{w['title']}")
        if "label" in w and w["label"]:
            parts.append(f"label:{w['label']}")
        if "min" in w:
            parts.append(f"min:{w['min']}")
        if "max" in w:
            parts.append(f"max:{w['max']}")
        if "compressed" in w:
            parts.append(f"[{w['compressed']} similar PVs grouped]")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def ask_framework():
    """Ask user to choose UI framework."""
    print("Choose output framework:")
    print("  1. tkinter (default — built into Python, no install needed)")
    print("  2. PyQt6   (modern look, requires: pip install PyQt6)")
    choice = input("Enter 1 or 2 (default=1): ").strip()
    if choice not in ("1", "2", ""):
        print(f"Unknown choice '{choice}', defaulting to tkinter.")
    framework = "PyQt6" if choice == "2" else "tkinter"
    print(f"Using: {framework}\n")
    return framework


def generate_python_code(widgets, source_label, framework, ref_code=None):
    """Send widget context + style guide to Gemini and get a Python dashboard."""
    compressed = compress_repeated_pvs(widgets)
    pv_list_str = format_widget_summary(compressed)
    total_original = len(widgets)
    total_compressed = len(compressed)

    # Optional extra reference code
    extra_ref = ""
    if ref_code:
        ref_lines = ref_code.splitlines()[:100]
        extra_ref = f"""
ADDITIONAL REFERENCE CODE (also follow this style):
```python
{chr(10).join(ref_lines)}
```
"""

    prompt = f"""You are an EPICS control system expert and Python developer.

I have EDL file(s) from: {source_label}
Original widget count: {total_original} (compressed to {total_compressed} after grouping repeated PVs)
UI Framework: {framework}

WIDGET LIST (format: widget_type | pv_name | title | label | min | max | [N similar PVs grouped]):
{pv_list_str}

{STYLE_GUIDE}
{extra_ref}

TASK:
1. Analyze the PV names and widget types to determine:
   - What type of facility or system this is
   - Logical groupings (beam, vacuum, RF, magnet, BPM, temperature, etc.)
   - Which PVs are most critical for operators
   - Which PVs are binary status/interlock vs numeric measurements

2. Generate a complete, runnable Python dashboard that:
   - Follows the STYLE GUIDE above exactly
   - Uses {framework} as the UI framework
   - Implements PVManager, DockablePanel, SectionFrame, MiniPlot classes
   - Groups PVs into logical DockablePanel sections
   - Uses dirty-set pattern for efficient updates
   - Has SIMULATOR MODE on by default with realistic random values
   - Supports --online flag to connect to real EPICS

IMPORTANT RULES:
- Output ONLY Python code — no explanation, no markdown backticks
- Code must be complete and immediately runnable
- Start with: import sys
- Simulator must be ON by default
- CRITICAL: Only use PV names from the WIDGET LIST above. 
  Never invent or guess PV names. If a chart needs a PV, 
  use one from the list. No exceptions.
- Use the exact colors from the style guide
- Follow the exact code structure order from the style guide
"""

    print(f"Sending {total_compressed} widget summaries to Gemini AI...")
    print(f"(Compressed from {total_original} original widgets)")
    if ref_code:
        print("(Extra reference file included)")

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text


def save_code(code, output_file):
    """Clean, validate, and save the generated code."""
    code = code.strip()

    # Remove markdown fences if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    code = code.strip()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(code)

    # Syntax check
    try:
        compile(code, output_file, 'exec')
        print(f"Saved to: {output_file} ✓ syntax OK")
    except SyntaxError as e:
        print(f"Saved to: {output_file}")
        print(f"⚠ Warning: syntax error on line {e.lineno}: {e.msg}")
        print(f"  Fix manually before running.")


def main():
    ref_file, input_paths = parse_args(sys.argv[1:])

    ref_code = None
    if ref_file:
        ref_code = load_ref_file(ref_file)

    edl_files = find_edl_files(input_paths)
    output_file = "generated_monitor.py"
    source_label = ", ".join(edl_files)

    print(f"\n=== EDL to Python Converter — By Example ===")
    print(f"Input:  {source_label}")
    if ref_file:
        print(f"Extra ref: {ref_file}")
    print(f"Style:  pls_monitor_end2.py (hardcoded)")
    print()

    # Ask framework
    framework = ask_framework()

    # Extract widget info
    print("Extracting widget info...")
    widgets = []
    for edl_file in edl_files:
        file_widgets = extract_widget_info(edl_file)
        widgets.extend(file_widgets)
        print(f"  {edl_file}: {len(file_widgets)} widgets found")

    print(f"Total widgets: {len(widgets)}")

    if len(widgets) == 0:
        print("No widgets found. Check that your .edl files have PV attributes.")
        sys.exit(1)

    # Generate
    code = generate_python_code(widgets, source_label, framework, ref_code)
    save_code(code, output_file)

    print(f"\nDone! Run your dashboard:")
    print(f"  py {output_file}           # simulator mode")
    print(f"  py {output_file} --online  # real EPICS")


if __name__ == "__main__":
    main()