"""
EDL to Python Converter v2
Converts EPICS EDL display files into Python monitoring dashboards.
Uses the Gemini API (google.genai) to generate clean, styled code.

Features:
  - Robust state-machine block parser handling nested groups
  - Rich widget context extraction (x, y, w, h, colors, precision, visibility)
  - Smart PV pattern compression and spatial grouping
  - Syntax check validation with automatic retry correction
  - Scriptable CLI via argparse (no interactive inputs)
"""

import os
import sys
import re
import argparse
import glob
from collections import defaultdict
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# ===============================================================
# STYLE GUIDE - extracted from pls_monitor_end2.py
# ===============================================================
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
     CRITICAL: Bind mousewheel ONLY to the local scrollable canvas, NOT globally, to avoid scroll conflicts.
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

# PV attribute regex patterns
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


class EDLParser:
    """Robust EDL file parser handling nesting structure (Groups) and properties."""
    
    def __init__(self):
        self.widgets = []
        self.screen_info = {"title": "EPICS Monitor", "w": 800, "h": 600}

    def parse_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"Error: file not found: {filepath}")
            return [], self.screen_info

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        widgets = []
        screen_info = self.screen_info.copy()
        
        in_screen = False
        stack = []  # Stack of {"type": str, "class": str, "lines": []}
        
        last_comment = ""
        last_class = ""
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line == "beginScreenProperties":
                in_screen = True
                i += 1
                continue
            elif line == "endScreenProperties":
                in_screen = False
                i += 1
                continue
                
            if in_screen:
                if line.startswith("title "):
                    m = re.search(r'title\s+"([^"]+)"', line)
                    if m: screen_info["title"] = m.group(1)
                elif line.startswith("w "):
                    m = re.search(r'w\s+(\d+)', line)
                    if m: screen_info["w"] = int(m.group(1))
                elif line.startswith("h "):
                    m = re.search(r'h\s+(\d+)', line)
                    if m: screen_info["h"] = int(m.group(1))
                i += 1
                continue
                
            # Track comments and classes outside/inside blocks
            if line.startswith("# (") and line.endswith(")"):
                last_comment = line[3:-1]
            elif line.startswith("object "):
                parts = line.split(None, 1)
                if len(parts) > 1:
                    last_class = parts[1]
            elif line == "beginObjectProperties":
                stack.append({
                    "type": last_comment or "Unknown",
                    "class": last_class or "UnknownClass",
                    "lines": []
                })
                last_comment = ""
                last_class = ""
            elif line == "endObjectProperties":
                if stack:
                    obj = stack.pop()
                    parsed = self._parse_block(obj["type"], obj["class"], obj["lines"])
                    if parsed:
                        widgets.append(parsed)
            else:
                if stack:
                    stack[-1]["lines"].append(lines[i])
            i += 1
            
        self.widgets = widgets
        self.screen_info = screen_info
        return widgets, screen_info

    def _parse_block(self, wtype, wclass, block_lines):
        block_content = "\n".join(block_lines)
        info = {
            "type": wtype,
            "class": wclass,
        }
        
        # Dimensions
        for prop in ['x', 'y', 'w', 'h']:
            m = re.search(rf'^{prop}\s+(\d+)', block_content, re.MULTILINE)
            if m:
                info[prop] = int(m.group(1))
                
        # Extract all potential PVs
        pvs = {}
        for pattern in PV_PATTERNS:
            match = re.search(pattern, block_content)
            if match:
                pv_name = match.group(1)
                role = pattern.split('\\s+')[0]
                pvs[role] = pv_name
                if "pv" not in info:
                    info["pv"] = pv_name
        
        if pvs:
            info["pvs"] = pvs
            
        # Label/Value
        m_label = re.search(r'label\s+"([^"]+)"', block_content)
        if m_label:
            info["label"] = m_label.group(1)
        else:
            m_val = re.search(r'value\s+\{[^}]*"([^"]+)"', block_content, re.DOTALL)
            if m_val:
                info["label"] = m_val.group(1)
                
        # Colors
        for color_key in ['fgColor', 'bgColor', 'fillColor', 'lineColor']:
            m = re.search(rf'{color_key}\s+index\s+(\d+)', block_content)
            if m:
                info[color_key] = int(m.group(1))
                
        # Precision
        m_prec = re.search(r'precision\s+(\d+)', block_content)
        if m_prec:
            info["precision"] = int(m_prec.group(1))
            
        # Visibility
        m_vis = re.search(r'visPv\s+"([^"]+)"', block_content)
        if m_vis:
            info["visPv"] = m_vis.group(1)
            for vis_prop in ['visMin', 'visMax']:
                m_v = re.search(rf'{vis_prop}\s+"?([0-9\.-]+)"?', block_content)
                if m_v:
                    info[vis_prop] = float(m_v.group(1))
            if 'visInvert' in block_content:
                info['visInvert'] = True
                
        # Related displays
        m_disp = re.search(r'displayFileName\s+\{\s*\d*\s*"([^"]+)"', block_content)
        if m_disp:
            info["displayFileName"] = m_disp.group(1)
        else:
            m_disp_simple = re.search(r'displayFileName\s+"([^"]+)"', block_content)
            if m_disp_simple:
                info["displayFileName"] = m_disp_simple.group(1)
                
        # We only keep widgets if they have a PV or a label/related display
        if "pv" in info or "label" in info or "displayFileName" in info:
            return info
        return None


def compress_repeated_pvs(widgets):
    """Compress repeated sequentially-numbered PVs into single pattern summaries."""
    groups = defaultdict(list)
    for w in widgets:
        pv = w.get("pv", "")
        if not pv:
            continue
        # Replace all digits with {N} to identify patterns
        pattern = re.sub(r'\d+', '{N}', pv)
        groups[pattern].append(w)
        
    compressed = []
    for pattern, group in groups.items():
        if len(group) <= 3:
            # Not enough repetition to compress
            compressed.extend(group)
        else:
            nums = []
            for w in group:
                found = re.findall(r'\d+', w["pv"])
                if found:
                    nums.append(int(found[0]))
            nums.sort()
            
            base = group[0].copy()
            if nums:
                padding = len(re.findall(r'\d+', group[0]["pv"])[0])
                start_str = str(nums[0]).zfill(padding)
                base["pv"] = f"{pattern.replace('{N}', start_str)}~{nums[-1]} (repeated {len(group)}x)"
            base["compressed_count"] = len(group)
            compressed.append(base)
            
    # Add widgets without PVs (like buttons or text only)
    for w in widgets:
        if "pv" not in w:
            compressed.append(w)
            
    return compressed


def build_prompt(widgets, screen_info, framework, style_guide, ref_code=None):
    """Construct a detailed prompt for Gemini."""
    # Filter down widget info to keep prompt neat but highly descriptive
    widget_summaries = []
    for w in widgets:
        parts = [f"type:{w['type']}", f"class:{w['class']}"]
        if "pv" in w:
            parts.append(f"pv:{w['pv']}")
        if "label" in w:
            parts.append(f"label:{w['label']}")
        if "x" in w and "y" in w:
            parts.append(f"pos:({w['x']},{w['y']})")
        if "w" in w and "h" in w:
            parts.append(f"size:{w['w']}x{w['h']}")
        if "precision" in w:
            parts.append(f"precision:{w['precision']}")
        if "visPv" in w:
            parts.append(f"visPv:{w['visPv']}")
        if "displayFileName" in w:
            parts.append(f"opens:{w['displayFileName']}")
        if "compressed_count" in w:
            parts.append(f"[Compressed Group of {w['compressed_count']} widgets]")
        widget_summaries.append(" | ".join(parts))

    widget_list_str = "\n".join(widget_summaries)
    
    extra_ref = ""
    if ref_code:
        # Include first 150 lines of reference code to avoid excessive token count
        ref_lines = ref_code.splitlines()[:150]
        extra_ref = f"\nADDITIONAL STYLE REFERENCE CODE:\n```python\n" + "\n".join(ref_lines) + "\n```\n"

    prompt = f"""You are an EPICS control system expert and Python developer.

I have parsed an EPICS EDL file with the following screen properties:
Screen Title: "{screen_info['title']}"
Original Canvas Size: {screen_info['w']}x{screen_info['h']}

WIDGET CONTEXT LIST:
{widget_list_str}

{style_guide}
{extra_ref}

TASK:
1. Analyze the PV names and widget types to determine:
   - Logical groupings (beam metrics, status indicators, interlocks, BPMs, RF, etc.)
   - Which PVs are binary status/interlocks vs numeric measurements vs graph coordinates
2. Generate a complete, premium, immediately runnable Python dashboard that:
   - Uses the {framework} UI framework
   - Implements the exact PVManager, DockablePanel, SectionFrame, MiniPlot classes (or PyQt6 equivalents)
   - Groups PVs into logical DockablePanel sections based on the spatial positions and names
   - Uses the dirty-set pattern for highly efficient updates (only update GUI when PV changes)
   - Has SIMULATOR MODE enabled by default, generating realistic, smooth simulator values
   - Supports --online command line flag to switch to real EPICS (pyepics)
   
IMPORTANT RULES:
- Output ONLY the raw Python code — no markdown block formatting (e.g. do not wrap in ```python), no explanations.
- Code must compile and run out-of-the-box.
- Do not invent any new PV names. Use only PVs present in the WIDGET CONTEXT LIST.
- Follow the exact colors, monospace fonts (Consolas), and layout rules from the style guide.
- Use list comprehensions or loops for generating large numbers of similar PVs (e.g. cells or interlock rows) to keep the code compact.
"""
    return prompt


def clean_code(code_text):
    """Strips Markdown wrappers or code block markers if the model returns them."""
    code_text = code_text.strip()
    if "```python" in code_text:
        code_text = code_text.split("```python")[1].split("```")[0]
    elif "```" in code_text:
        code_text = code_text.split("```")[1].split("```")[0]
    return code_text.strip()


def validate_and_generate(prompt, model_name, output_file, retries=2):
    """Generates code with Gemini API and validates syntax, retrying on errors."""
    if not API_KEY:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)
    
    current_prompt = prompt
    for attempt in range(retries + 1):
        print(f"Sending request to Gemini AI (model: {model_name})... [Attempt {attempt + 1}/{retries + 1}]")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=current_prompt
            )
            code = clean_code(response.text)
            
            # Syntax Check
            try:
                compile(code, output_file, 'exec')
                print("Code compilation: SUCCESS [OK] (No syntax errors found!)")
                return code
            except SyntaxError as e:
                print(f"Syntax validation failed: Line {e.lineno}: {e.msg}")
                if attempt < retries:
                    print("Prompting Gemini to correct the syntax error...")
                    current_prompt = f"""Here is the code you generated, which contains a SyntaxError:
---
{code}
---
The syntax check failed with the error:
Line {e.lineno}: {e.msg}

Please correct the syntax error and return the full updated Python code. Do not include explanations, output only raw Python code.
"""
                else:
                    print("Failed to resolve syntax errors within retry limits. Saving code with errors.")
                    return code
        except Exception as e:
            print(f"Gemini API request failed: {e}")
            if attempt >= retries:
                sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="EPICS EDL to Python Dashboard Converter v2")
    parser.add_argument("edl_files", nargs="*", help="EDL files to convert (auto-detects *.edl if omitted)")
    parser.add_argument("--ref", help="Reference Python file for style learning")
    parser.add_argument("--framework", choices=["tkinter", "pyqt6"], default="tkinter", help="UI framework (default: tkinter)")
    parser.add_argument("--output", default="generated_monitor.py", help="Output filename (default: generated_monitor.py)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name (default: gemini-2.5-flash)")
    parser.add_argument("--retries", type=int, default=2, help="Number of retries on syntax error (default: 2)")
    
    args = parser.parse_args()
    
    # 1. Locate EDL files
    edl_files = args.edl_files
    if not edl_files:
        edl_files = glob.glob("*.edl")
        if not edl_files:
            print("No .edl files specified and none found in current directory.")
            sys.exit(1)
        print(f"Auto-detected {len(edl_files)} .edl file(s): {edl_files}")
        
    # 2. Load extra reference
    ref_code = None
    if args.ref:
        if os.path.exists(args.ref):
            with open(args.ref, 'r', encoding='utf-8', errors='ignore') as f:
                ref_code = f.read()
            print(f"Reference file loaded: {args.ref} ({len(ref_code)} bytes)")
        else:
            print(f"Warning: reference file not found: {args.ref}")
            
    # 3. Parse EDL Widgets
    parser_inst = EDLParser()
    all_widgets = []
    combined_screen_info = {"title": "EPICS Monitor", "w": 0, "h": 0}
    
    print("\nParsing EDL files...")
    for edl in edl_files:
        widgets, s_info = parser_inst.parse_file(edl)
        all_widgets.extend(widgets)
        print(f"  {edl}: {len(widgets)} widgets found")
        # Combine screen bounds
        combined_screen_info["w"] = max(combined_screen_info["w"], s_info["w"])
        combined_screen_info["h"] = max(combined_screen_info["h"], s_info["h"])
        if combined_screen_info["title"] == "EPICS Monitor" and s_info["title"] != "EPICS Monitor":
            combined_screen_info["title"] = s_info["title"]
            
    if not all_widgets:
        print("Error: No widgets with PVs or labels found. Cannot generate dashboard.")
        sys.exit(1)
        
    print(f"Total widget elements parsed: {len(all_widgets)}")
    
    # 4. Compress PVs
    compressed_widgets = compress_repeated_pvs(all_widgets)
    print(f"Compressed elements for LLM context: {len(compressed_widgets)}")
    
    # 5. Build prompt & generate
    prompt = build_prompt(compressed_widgets, combined_screen_info, args.framework, STYLE_GUIDE, ref_code)
    
    generated_code = validate_and_generate(prompt, args.model, args.output, args.retries)
    
    # 6. Save final file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(generated_code)
        
    print(f"\nDone! Dashboard code saved to: {args.output}")
    print("Run the dashboard in simulation mode:")
    print(f"  python {args.output}")
    print("Run the dashboard connecting to real EPICS PVs:")
    print(f"  python {args.output} --online")


if __name__ == "__main__":
    main()
