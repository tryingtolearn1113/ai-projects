import os
import re

# Standard PV attribute regex patterns (for EDL/ADL)
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

class BaseParser:
    """Abstract base class for EPICS display parsers."""
    
    def __init__(self):
        self.widgets = []
        self.screen_info = {"title": "EPICS Monitor", "w": 800, "h": 600}
        
    def parse_file(self, filepath):
        """
        Parses the display file and returns a tuple: (widgets, screen_info).
        Widgets is a list of dictionaries.
        Screen info is a dictionary with 'title', 'w', and 'h'.
        """
        raise NotImplementedError("Subclasses must implement parse_file()")
