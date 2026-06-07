import os
import xml.etree.ElementTree as ET
from .base_parser import BaseParser

class BOBParser(BaseParser):
    """Parser for Phoebus .bob XML files."""
    
    def parse_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"Error: file not found: {filepath}")
            return [], self.screen_info

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Error parsing BOB XML: {e}")
            return [], self.screen_info

        widgets = []
        screen_info = self.screen_info.copy()
        
        # BOB files have display dimensions at the root level
        if root.tag == "display":
            w = root.find("width")
            if w is not None and w.text: screen_info["w"] = int(w.text)
            h = root.find("height")
            if h is not None and h.text: screen_info["h"] = int(h.text)
            title = root.find("name")
            if title is not None and title.text: screen_info["title"] = title.text

        # Recursively find all widgets
        for widget_node in root.findall(".//widget"):
            parsed = self._parse_widget_node(widget_node)
            if parsed:
                widgets.append(parsed)

        self.widgets = widgets
        self.screen_info = screen_info
        return widgets, screen_info

    def _parse_widget_node(self, node):
        wtype = node.get("type", "Unknown")
        
        info = {
            "type": wtype,
            "class": "BOBWidget"
        }

        # Dimensions
        for prop in ['x', 'y', 'width', 'height']:
            elem = node.find(prop)
            if elem is not None and elem.text:
                key = 'w' if prop == 'width' else 'h' if prop == 'height' else prop
                try:
                    info[key] = int(float(elem.text))
                except ValueError:
                    pass

        # PV Name
        pv_elem = node.find("pv_name")
        if pv_elem is not None and pv_elem.text:
            info["pv"] = pv_elem.text

        # Label/Text
        text_elem = node.find("text")
        if text_elem is not None and text_elem.text:
            info["label"] = text_elem.text

        # Rules for visibility
        rules = node.find("rules")
        if rules is not None:
            for rule in rules.findall("rule"):
                if rule.get("prop_id") == "visible":
                    pv = rule.find("pv_name")
                    if pv is not None and pv.text:
                        info["visPv"] = pv.text

        if "pv" in info or "label" in info:
            return info
        return None
