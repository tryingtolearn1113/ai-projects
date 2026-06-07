"""
Universal EPICS to Python Converter
Converts EPICS display files (.edl, .adl, .opi, .bob) into Python monitoring dashboards.
Uses the Gemini API (google.genai) to generate clean, styled code.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Import our modular components
from universal_converter.parsers import get_parser
from universal_converter.core.compressor import compress_repeated_pvs
from universal_converter.core.style_guide import STYLE_GUIDE
from universal_converter.core.ai_client import build_prompt, validate_and_generate

def main():
    parser = argparse.ArgumentParser(description="Universal EPICS Display to Python Dashboard Converter")
    parser.add_argument("display_files", nargs="+", help="EPICS display files to convert (.edl, .adl, .opi, .bob)")
    parser.add_argument("--ref", help="Reference Python file for style learning")
    parser.add_argument("--framework", choices=["tkinter", "pyqt6"], default="tkinter", help="UI framework (default: tkinter)")
    parser.add_argument("--output", default="generated_dashboard.py", help="Output filename (default: generated_dashboard.py)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name (default: gemini-2.5-flash)")
    parser.add_argument("--retries", type=int, default=2, help="Number of retries on syntax error (default: 2)")
    
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set. Please create a .env file.")
        sys.exit(1)
        
    # 1. Load extra reference
    ref_code = None
    if args.ref:
        if os.path.exists(args.ref):
            with open(args.ref, 'r', encoding='utf-8', errors='ignore') as f:
                ref_code = f.read()
            print(f"Reference file loaded: {args.ref} ({len(ref_code)} bytes)")
        else:
            print(f"Warning: reference file not found: {args.ref}")
            
    # 2. Parse Display Files
    all_widgets = []
    combined_screen_info = {"title": "EPICS Universal Monitor", "w": 0, "h": 0}
    
    print("\nParsing EPICS display files...")
    for filepath in args.display_files:
        try:
            parser_inst = get_parser(filepath)
            widgets, s_info = parser_inst.parse_file(filepath)
            all_widgets.extend(widgets)
            print(f"  {filepath}: {len(widgets)} widgets found")
            
            # Combine screen bounds
            combined_screen_info["w"] = max(combined_screen_info["w"], s_info.get("w", 0))
            combined_screen_info["h"] = max(combined_screen_info["h"], s_info.get("h", 0))
            if s_info.get("title") and s_info["title"] != "EPICS Monitor":
                combined_screen_info["title"] = s_info["title"]
        except ValueError as e:
            print(f"  Skipping {filepath}: {e}")
            
    if not all_widgets:
        print("Error: No valid widgets found across all files. Cannot generate dashboard.")
        sys.exit(1)
        
    print(f"Total widget elements parsed: {len(all_widgets)}")
    
    # 3. Compress PVs
    compressed_widgets = compress_repeated_pvs(all_widgets)
    print(f"Compressed elements for LLM context: {len(compressed_widgets)}")
    
    # 4. Build prompt & generate
    prompt = build_prompt(compressed_widgets, combined_screen_info, args.framework, STYLE_GUIDE, ref_code)
    
    generated_code = validate_and_generate(api_key, prompt, args.model, args.output, args.retries)
    
    # 5. Save final file
    if generated_code:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(generated_code)
            
        print(f"\nDone! Dashboard code saved to: {args.output}")
        print("Run the dashboard in simulation mode:")
        print(f"  python {args.output}")
        print("Run the dashboard connecting to real EPICS PVs:")
        print(f"  python {args.output} --online")
    else:
        print("\nFailed to generate code.")

if __name__ == "__main__":
    main()
