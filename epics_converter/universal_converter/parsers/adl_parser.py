import os
import re
from .base_parser import BaseParser

class ADLParser(BaseParser):
    """Parser for MEDM .adl files. Uses a stack to handle nested { } blocks."""
    
    def parse_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"Error: file not found: {filepath}")
            return [], self.screen_info

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        widgets = []
        screen_info = self.screen_info.copy()
        
        # Tokenize by finding blocks
        # We will iterate through lines. When we see a word followed by { we push to stack.
        # When we see }, we pop.
        lines = content.splitlines()
        
        stack = []  # List of {"type": str, "lines": []}
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if block starts: e.g. 'text {' or '"message button" {'
            m_start = re.match(r'^("?[a-zA-Z0-9_ ]+"?)\s*\{', line)
            if m_start:
                block_type = m_start.group(1).strip('"')
                stack.append({"type": block_type, "lines": []})
            elif line == "}":
                if stack:
                    obj = stack.pop()
                    # If this was a display block, get screen info
                    if obj["type"] == "display":
                        screen_info = self._parse_display_block(obj["lines"], screen_info)
                    elif not stack: # Top-level widget block
                        parsed = self._parse_widget_block(obj["type"], obj["lines"])
                        if parsed:
                            widgets.append(parsed)
                    else:
                        # Nested block, append its content to the parent block's lines
                        parent = stack[-1]
                        parent["lines"].append(obj["type"] + " {")
                        parent["lines"].extend(obj["lines"])
                        parent["lines"].append("}")
            else:
                if stack:
                    stack[-1]["lines"].append(line)
                    
            i += 1
            
        self.widgets = widgets
        self.screen_info = screen_info
        return widgets, screen_info

    def _parse_display_block(self, lines, screen_info):
        block_content = "\\n".join(lines)
        m_w = re.search(r'width=(\\d+)', block_content)
        if m_w: screen_info["w"] = int(m_w.group(1))
        m_h = re.search(r'height=(\\d+)', block_content)
        if m_h: screen_info["h"] = int(m_h.group(1))
        return screen_info

    def _parse_widget_block(self, wtype, lines):
        block_content = "\\n".join(lines)
        info = {
            "type": wtype,
            "class": "ADLWidget",
        }
        
        # Dimensions
        for prop in ['x', 'y', 'width', 'height']:
            m = re.search(rf'^{prop}=(\\d+)', block_content, re.MULTILINE)
            if m:
                # Convert width/height to w/h for consistency
                key = 'w' if prop == 'width' else 'h' if prop == 'height' else prop
                info[key] = int(m.group(1))
                
        # PV: 'chan="PV"'
        m_chan = re.search(r'chan="([^"]+)"', block_content)
        if m_chan:
            info["pv"] = m_chan.group(1)
            
        # Label: 'textix="label"' or 'label="label"'
        m_label = re.search(r'(textix|label)="([^"]+)"', block_content)
        if m_label:
            info["label"] = m_label.group(2)
            
        # Visibility
        m_vis = re.search(r'vis="([^"]+)"', block_content)
        if m_vis:
            info["visRule"] = m_vis.group(1)
            
        # Colors
        for color_key in ['clr', 'bclr']:
            m = re.search(rf'{color_key}=(\\d+)', block_content)
            if m:
                info[color_key] = int(m.group(1))
                
        if "pv" in info or "label" in info:
            return info
        return None
