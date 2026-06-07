STYLE_GUIDE = """
STYLE GUIDE (always follow exactly - extracted from a real working EPICS dashboard):

== COLORS (use these exact hex values) ==
BG='#07090f', PANEL='#0d1120', PANEL2='#0f1525', BORDER='#1a2540'
HDR='#0a1530', GOLD='#F5C518', CYAN='#22D3EE', LIME='#4ADE80'
WHITE='#E2E8F0', GRAY='#4B5563', DIM='#1F2937'
GREEN='#16A34A', RED='#DC2626', YELLOW='#CA8A04'
BLUE='#2563EB', AMBER='#D97706', PINK='#DB2777'
TEAL='#2DD4BF', LBLUE='#60A5FA'

== ARCHITECTURE - use these exact classes ==

1. PVManager class:
   - self.values = {}
   - self.history = defaultdict(lambda: deque(maxlen=1200))
   - self._dirty = set()  - only update widgets when values change
   - self._lock = threading.Lock()  - thread safety for EPICS callbacks
   - Methods: get(pv), get_history(pv, n), take_dirty(), connect_all(pvs), _cb(callback)
   - Simulator: separate daemon thread, updates every 0.5s
     use math.sin/cos for slow drift + random.uniform for noise
     binary PVs mostly 1 (OK), rarely flip to 0
     heartbeat PVs increment by 1, wrap at 10000

2. SectionFrame class (panel with scrollable title bar):
   - PANEL background, BORDER highlight 1px
   - Title bar: HDR background, 3px accent color strip on left, Consolas 7 bold
   - Optional scrollable body: canvas + scrollbar + mousewheel binding
     CRITICAL: Bind mousewheel ONLY to the local scrollable canvas, NOT globally, to avoid scroll conflicts.
   - Usage: SectionFrame(parent, "TITLE", accent=CYAN, scrollable=True)

3. MiniPlot class (auto-resizing trend chart):
   - Canvas-based, fills parent, bind Configure for resize
   - Background '#050810', filled polygon + smooth line on top
   - GRAY grid lines (dashed), GOLD last value top-right corner
   - Y-axis labels in GRAY Consolas 5

4. DockablePanel class (floating, draggable, resizable):
   - Drag by title bar (cursor='fleur'), binds ButtonPress-1 and B1-Motion
   - Resize grip bottom-right corner triangle (cursor='sizing')
   - Auto-scales all child fonts on resize using font registry
   - Saves layout to ~/.dashboard_layout.json on window close
   - Ctrl+R resets layout to DEFAULT_LAYOUT

5. HeartbeatWidget (inline, fits in subheader bar):
   - LED canvas oval: GREEN when alive, fades to YELLOW when stalled
   - EKG mini-canvas shows value history as line
   - tick() method called every 500ms

6. BPMStation widget:
   - Compact: label + X bar canvas + Y bar canvas + numeric labels
   - Bar shows deflection from center, RED if > threshold
   - TEAL for X, LBLUE for Y

== LAYOUT ==
- Header bar: HDR background, GOLD bold title left, CYAN clock Consolas 11 bold right
  Lightning bolt emoji before title, CYAN 1px underline strip below header
- Subheader bar: PANEL background, sections divided by BORDER lines
  Contains: Tune & Beam | Injection beam params | Heartbeats - all in one row
- Main workspace: tk.Frame with all DockablePanel instances placed with .place(x=,y=)
- DEFAULT_LAYOUT dict: defines x, y, w, h for each panel name at 1920x1080

== UPDATE PATTERN (critical - always use this pattern) ==
- 500ms tick: root.after(500, self._tick)
- self._tick calls self._tick_inner() inside try/except to prevent crashes
- dirty = pvm.take_dirty() - atomically get changed PVs
- Only update widgets whose PV is in dirty set - never update everything every tick
- For numeric labels: compare new text to last text, skip .config() if same
- Heartbeats tick every cycle regardless of dirty set

== WIDGET PATTERNS ==
- Key metric: GRAY Consolas 6 label -> LIME Consolas 16 bold value -> DIM unit label
  all in a vertical Frame, separated by BORDER 1px dividers
- Status/interlock indicator: tk.Label, bg=GREEN if value==1 else RED, bg=DIM if None
  width=3-7, no border, relief='flat'
- Shutter: tk.Canvas rectangle 26x9, GREEN='OPN' or RED='CLS' text in BG color Consolas 5
- RF LED: small Canvas oval 7x7, GREEN if abs(value)<threshold else YELLOW
- Scrollable list (ID gaps, shutters, RF): Canvas + Scrollbar + mousewheel binding
  inner Frame holds rows, each row is label + value label packed LEFT/RIGHT

== FONTS (always Consolas, always monospace) ==
- 5: axis tick labels
- 6: small labels, units, secondary info
- 7: section titles, subheader labels
- 8-9: normal values, BPM labels
- 11-12: header clock and title
- 16: key metrics values
- Always bold for values, titles, and status text

== CODE STRUCTURE (follow this order exactly) ==
1. imports (os, time, json, math, threading, logging, collections, tkinter, epics)
2. logging setup
3. Color constants
4. PV definition lists and dicts (KEY_METRICS, INTERLOCKS, etc.)
5. PVManager class
6. Widget classes: SectionFrame, MiniPlot, DockablePanel, HeartbeatWidget, BPMStation
7. DEFAULT_LAYOUT dict
8. Dashboard class with _build(), _tick(), panel content methods
9. if __name__ == "__main__": block with simulator/online mode selection

== SIMULATOR MODE ==
- Default: simulate=True
- Command line: python script.py --online switches to real EPICS
- Print clear INFO message showing which mode is active
- try: import epics / except ImportError: EPICS_AVAILABLE = False
"""
