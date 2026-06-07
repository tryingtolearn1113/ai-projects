import os
from .edl_parser import EDLParser
from .adl_parser import ADLParser
from .opi_parser import OPIParser
from .bob_parser import BOBParser

def get_parser(filepath):
    """Returns the appropriate parser based on the file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.edl':
        return EDLParser()
    elif ext == '.adl':
        return ADLParser()
    elif ext == '.opi':
        return OPIParser()
    elif ext == '.bob':
        return BOBParser()
    else:
        raise ValueError(f"Unsupported EPICS file extension: {ext}")
