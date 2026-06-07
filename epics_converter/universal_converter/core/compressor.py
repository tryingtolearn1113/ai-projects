import re
from collections import defaultdict

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
