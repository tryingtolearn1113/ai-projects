import os
import re
from .base_parser import BaseParser, PV_PATTERNS

class EDLParser(BaseParser):
    """Robust EDL file parser handling nesting structure (Groups) and properties."""
    
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
        block_content = "\\n".join(block_lines)
        info = {
            "type": wtype,
            "class": wclass,
        }
        
        # Dimensions
        for prop in ['x', 'y', 'w', 'h']:
            m = re.search(rf'^{prop}\s+(\\d+)', block_content, re.MULTILINE)
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
            m_val = re.search(r'value\s+\\{[^}]*"([^"]+)"', block_content, re.DOTALL)
            if m_val:
                info["label"] = m_val.group(1)
                
        # Colors
        for color_key in ['fgColor', 'bgColor', 'fillColor', 'lineColor']:
            m = re.search(rf'{color_key}\s+index\s+(\\d+)', block_content)
            if m:
                info[color_key] = int(m.group(1))
                
        # Precision
        m_prec = re.search(r'precision\s+(\\d+)', block_content)
        if m_prec:
            info["precision"] = int(m_prec.group(1))
            
        # Visibility
        m_vis = re.search(r'visPv\s+"([^"]+)"', block_content)
        if m_vis:
            info["visPv"] = m_vis.group(1)
            for vis_prop in ['visMin', 'visMax']:
                m_v = re.search(rf'{vis_prop}\s+"?([0-9\\.-]+)"?', block_content)
                if m_v:
                    info[vis_prop] = float(m_v.group(1))
            if 'visInvert' in block_content:
                info['visInvert'] = True
                
        # Related displays
        m_disp = re.search(r'displayFileName\s+\\{\s*\\d*\s*"([^"]+)"', block_content)
        if m_disp:
            info["displayFileName"] = m_disp.group(1)
        else:
            m_disp_simple = re.search(r'displayFileName\s+"([^"]+)"', block_content)
            if m_disp_simple:
                info["displayFileName"] = m_disp_simple.group(1)
                
        if "pv" in info or "label" in info or "displayFileName" in info:
            return info
        return None
