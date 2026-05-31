"""
EPICS EDL to Python Converter
Uses Gemini AI to analyze PV names and generate dashboard
Works with ANY .edl file!
"""
import re
import os
import sys
from google import genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# ═══════════════════════════════════
# STEP 1 - Generic EDL PV Extractor
# ═══════════════════════════════════
def extract_pvs_from_edl(filepath):
    """Extract ALL PV names from any .edl file"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []

    with open(filepath, 'r',
              encoding='utf-8',
              errors='ignore') as f:
        content = f.read()

    pvs = set()

    # Find all PV patterns
    patterns = [
        r'controlPv\s+"([^"]+)"',
        r'visPv\s+"([^"]+)"',
        r'indicatorPv\s+"([^"]+)"',
        r'readPv\s+"([^"]+)"',
        r'writePv\s+"([^"]+)"',
        r'xPv\s*\{\s*\d+\s+"([^"]+)"',
        r'yPv\s*\{\s*\d+\s+"([^"]+)"',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content)
        for pv in matches:
            pv = pv.strip()
            if pv and len(pv) > 2:
                pvs.add(pv)

    return sorted(list(pvs))


# ═══════════════════════════════════
# STEP 2 - AI Analyzes PV Names
# ═══════════════════════════════════
def analyze_pvs_with_ai(pvs):
    """Send PV names to Gemini for analysis"""
    pv_list = "\n".join(pvs)

    prompt = f"""You are an EPICS control system expert.
    
I have extracted these PV (Process Variable) names from an EDL file 
at a particle accelerator facility.

Please analyze each PV name and categorize them into these groups:
- beam_metrics: beam current, energy, lifetime, tune
- cell_status: storage ring cell status indicators
- interlocks: machine protection system PVs
- bpm: beam position monitors
- rf_system: RF cavity and frequency PVs
- shutters: beamline shutter PVs
- id_gaps: insertion device gap PVs
- heartbeat: system heartbeat monitors
- other: anything else

PV Names:
{pv_list}

Respond in this exact format for each PV:
PV_NAME | CATEGORY | SHORT_DESCRIPTION

Example:
SR:G00:BEAMCURRENT_T | beam_metrics | Storage ring beam current
LI:BPM_T01:X | bpm | LINAC BPM T01 X position
"""

    print("Asking Gemini to analyze PVs...")
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text


# ═══════════════════════════════════
# STEP 3 - Parse AI Response
# ═══════════════════════════════════
def parse_ai_response(ai_text):
    """Parse Gemini's categorization into dict"""
    categories = {
        'beam_metrics': [],
        'cell_status': [],
        'interlocks': [],
        'bpm': [],
        'rf_system': [],
        'shutters': [],
        'id_gaps': [],
        'heartbeat': [],
        'other': []
    }

    for line in ai_text.strip().split('\n'):
        if '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 2:
            continue

        pv = parts[0].strip()
        category = parts[1].strip().lower()
        description = parts[2].strip() if len(parts) > 2 else ""

        # Match to our categories
        matched = False
        for cat in categories.keys():
            if cat in category or category in cat:
                categories[cat].append({
                    'pv': pv,
                    'description': description
                })
                matched = True
                break

        if not matched:
            categories['other'].append({
                'pv': pv,
                'description': description
            })

    return categories


# ═══════════════════════════════════
# STEP 4 - Print Summary
# ═══════════════════════════════════
def print_summary(categories):
    """Print categorized PV summary"""
    print("\n=== AI Analysis Results ===\n")
    total = 0
    for cat, pvs in categories.items():
        if pvs:
            print(f"{cat:20s}: {len(pvs)} PVs")
            total += len(pvs)
    print(f"\n{'TOTAL':20s}: {total} PVs")

    print("\n=== Detailed Categories ===")
    for cat, pvs in categories.items():
        if not pvs:
            continue
        print(f"\n── {cat.upper()} ──")
        for item in pvs[:3]:
            print(f"  {item['pv']}")
            if item['description']:
                print(f"    → {item['description']}")
        if len(pvs) > 3:
            print(f"  ... and {len(pvs)-3} more")


# ═══════════════════════════════════
# MAIN
# ═══════════════════════════════════
def main():
    # Default files
    left  = "SurvLeft_20240818.edl"
    right = "SurvRight_20240818.edl"

    # Allow command line arguments
    if len(sys.argv) > 1:
        left = sys.argv[1]
    if len(sys.argv) > 2:
        right = sys.argv[2]

    print("=== EPICS EDL AI Converter ===\n")

    # Extract PVs from both files
    all_pvs = set()
    for filepath in [left, right]:
        if os.path.exists(filepath):
            print(f"Reading: {filepath}")
            pvs = extract_pvs_from_edl(filepath)
            all_pvs.update(pvs)
            print(f"Found {len(pvs)} PVs\n")

    if not all_pvs:
        print("No PVs found!")
        return

    print(f"Total unique PVs: {len(all_pvs)}")

    # AI Analysis
    ai_response = analyze_pvs_with_ai(
        sorted(list(all_pvs))
    )

    # Parse and display results
    categories = parse_ai_response(ai_response)
    print_summary(categories)

    # Save results
    output = "pv_analysis.txt"
    with open(output, 'w') as f:
        f.write("=== PV Analysis Results ===\n\n")
        for cat, pvs in categories.items():
            if pvs:
                f.write(f"\n── {cat.upper()} ──\n")
                for item in pvs:
                    f.write(f"  {item['pv']}")
                    if item['description']:
                        f.write(f" | {item['description']}")
                    f.write("\n")

    print(f"\n✅ Results saved to {output}")


if __name__ == "__main__":
    main()