import sys
from google import genai

def build_prompt(widgets, screen_info, framework, style_guide, ref_code=None):
    """Construct a detailed prompt for Gemini."""
    widget_summaries = []
    for w in widgets:
        parts = [f"type:{w['type']}", f"class:{w.get('class', 'Unknown')}"]
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
        ref_lines = ref_code.splitlines()[:150]
        extra_ref = f"\nADDITIONAL STYLE REFERENCE CODE:\n```python\n" + "\n".join(ref_lines) + "\n```\n"

    prompt = f"""You are an EPICS control system expert and Python developer.

I have parsed an EPICS display file with the following screen properties:
Screen Title: "{screen_info.get('title', 'Monitor')}"
Original Canvas Size: {screen_info.get('w', 800)}x{screen_info.get('h', 600)}

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
    """Strips Markdown wrappers or code block markers."""
    code_text = code_text.strip()
    if "```python" in code_text:
        code_text = code_text.split("```python")[1].split("```")[0]
    elif "```" in code_text:
        code_text = code_text.split("```")[1].split("```")[0]
    return code_text.strip()

def validate_and_generate(api_key, prompt, model_name, output_file, retries=2):
    """Generates code with Gemini API and validates syntax, retrying on errors."""
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
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
