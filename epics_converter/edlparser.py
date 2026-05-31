"""
EDL Parser - Reads EPICS EDL files and extracts PV information
"""
import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class EDLWidget:
    widget_type: str
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    pv_name: str = ""
    vis_pv: str = ""
    label: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

class EDLParser:
    def __init__(self):
        self.widgets: List[EDLWidget] = []
        self.screen_props: Dict = {}

    def parse_file(self, filepath: str) -> List[EDLWidget]:
        """Parse an EDL file and return list of widgets"""
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return []

        with open(filepath, 'r',
                  encoding='utf-8',
                  errors='ignore') as f:
            content = f.read()

        widgets = []
        # Split into object blocks
        blocks = re.split(r'(?=^# \()', content,
                         flags=re.MULTILINE)

        for block in blocks:
            widget = self._parse_block(block)
            if widget:
                widgets.append(widget)

        return widgets

    def _parse_block(self,
                     block: str) -> Optional[EDLWidget]:
        """Parse a single widget block"""
        # Get widget type from comment
        type_match = re.search(r'# \((.+?)\)', block)
        if not type_match:
            return None

        widget_type = type_match.group(1)

        # Get properties
        props = {}
        # x, y, w, h
        for prop in ['x', 'y', 'w', 'h']:
            m = re.search(rf'^{prop} (\d+)',
                         block, re.MULTILINE)
            if m:
                props[prop] = int(m.group(1))

        # PV names
        for pv_key in ['controlPv', 'visPv',
                       'indicatorPv']:
            m = re.search(rf'{pv_key} "(.+?)"', block)
            if m:
                props[pv_key] = m.group(1)

        # Label/value text
        value_match = re.search(
            r'value \{[^}]*"(.+?)"[^}]*\}',
            block, re.DOTALL)
        if value_match:
            props['label'] = value_match.group(1)

        # Colors
        for color_key in ['fgColor', 'bgColor',
                          'fillColor', 'lineColor']:
            m = re.search(
                rf'{color_key} index (\d+)', block)
            if m:
                props[color_key] = int(m.group(1))

        # Visibility
        if 'visInvert' in block:
            props['visInvert'] = True

        widget = EDLWidget(
            widget_type=widget_type,
            x=props.get('x', 0),
            y=props.get('y', 0),
            w=props.get('w', 0),
            h=props.get('h', 0),
            pv_name=props.get('controlPv',
                    props.get('indicatorPv', '')),
            vis_pv=props.get('visPv', ''),
            label=props.get('label', ''),
            properties=props
        )
        return widget

    def extract_all_pvs(self,
                        widgets: List[EDLWidget]) -> Dict:
        """Categorize all PVs found in widgets"""
        pvs = {
            'cell_status': {},
            'interlocks': {},
            'bpm': [],
            'pbpm': [],
            'heartbeat': [],
            'rf': [],
            'beam_metrics': [],
            'shutters': [],
            'other': []
        }

        for w in widgets:
            pv = w.pv_name or w.vis_pv
            if not pv:
                continue

            # Categorize by PV name pattern
            if re.match(r'SR:MISC\d+', pv):
                pvs['cell_status'][pv] = w

            elif 'SSS' in pv:
                # Shutters (BL:MISxxBxx format)
                pvs['shutters'].append(pv)

            elif re.match(r'BL:PBPM', pv):
                # PBPM graphs
                if pv not in pvs['pbpm']:
                    pvs['pbpm'].append(pv)

            elif re.match(r'(LI|SR):BPM', pv):
                # BPM position data
                if pv not in pvs['bpm']:
                    pvs['bpm'].append(pv)

            elif any(x in pv for x in
                    ['HEART', 'SOFB', 'FOFB']):
                # Heartbeats
                if pv not in pvs['heartbeat']:
                    pvs['heartbeat'].append(pv)

            elif re.match(r'SR:RF', pv):
                # RF system
                if pv not in pvs['rf']:
                    pvs['rf'].append(pv)

            elif any(x in pv for x in
                    ['MISPIJ', 'MISLINAC', 'MISBTL',
                    'MISSR', 'MISRFINT', 'MISIDBPM',
                    'MISGV', 'MISMPS4']):
                # Real interlocks
                if pv not in pvs['interlocks']:
                    pvs['interlocks'][pv] = w

            elif any(x in pv for x in
                    ['BEAMCURRENT', 'ENERGY',
                    'LIFETIME', 'TUNE', 'BEAMSIZE',
                    'TOPUP', 'EFFI', 'INJ']):
                # Beam metrics
                if pv not in pvs['beam_metrics']:
                    pvs['beam_metrics'].append(pv)

            elif re.match(r'SR:ID', pv):
                # ID gaps
                if 'id_gaps' not in pvs:
                    pvs['id_gaps'] = []
                if pv not in pvs['id_gaps']:
                    pvs['id_gaps'].append(pv)

            else:
                if pv not in pvs['other']:
                    pvs['other'].append(pv)

        return pvs

def analyze_edl_files(left_path: str,
                      right_path: str):
    """Analyze both EDL files and print summary"""
    parser = EDLParser()

    print("=== EDL File Analyzer ===\n")

    all_widgets = []
    for path, name in [(left_path, "LEFT"),
                       (right_path, "RIGHT")]:
        print(f"Reading {name}: {path}")
        widgets = parser.parse_file(path)
        all_widgets.extend(widgets)
        print(f"Found {len(widgets)} widgets\n")

    # Extract and categorize PVs
    pvs = parser.extract_all_pvs(all_widgets)

    print("=== PV Summary ===")
    for category, items in pvs.items():
        count = (len(items) if isinstance(items, list)
                else len(items))
        print(f"{category:20s}: {count} PVs")

    print("\n=== Detailed PV List ===")
    for category, items in pvs.items():
        if not items:
            continue
        print(f"\n── {category.upper()} ──")
        if isinstance(items, dict):
            for pv in list(items.keys())[:5]:
                print(f"  {pv}")
            if len(items) > 5:
                print(f"  ... and {len(items)-5} more")
        else:
            for pv in items[:5]:
                print(f"  {pv}")
            if len(items) > 5:
                print(f"  ... and {len(items)-5} more")

    return all_widgets, pvs

if __name__ == "__main__":
    LEFT  = "SurvLeft_20240818.edl"
    RIGHT = "SurvRight_20240818.edl"
    analyze_edl_files(LEFT, RIGHT)