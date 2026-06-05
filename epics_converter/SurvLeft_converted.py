#!/usr/bin/env python3
"""
PLS-II Operation Status Dashboard v5
Layout matches user's Image 2:
  - Header + clock
  - Metrics bar
  - Subheader: Tune | Injection | Heartbeats
  - Main 2-column body that fills ALL remaining height:
      LEFT:  beam chart / ID gaps / shutters / RF   (fixed width)
      RIGHT: cell status / interlocks / PBPM        (expands)
  - BPM: LINAC + BTL rows are INSIDE the LEFT column at the bottom
    (never cut off, because they're inside the body not below it)
"""
import os, time, json, threading, logging
from collections import defaultdict, deque

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('pls_monitor')

os.environ.setdefault('EPICS_CA_REPEATER_PORT', '5065')
from epics import PV as EpicsPV
import tkinter as tk

LAYOUT_FILE = os.path.join(os.path.expanduser('~'), '.pls_monitor_layout.json')

# ═══════════════════════════════════════════════════════════════
# PV DEFINITIONS  (v4 working PV names preserved exactly)
# ═══════════════════════════════════════════════════════════════
KEY_METRICS = [
    ("SR Energy",     "SR:ENERGY",            "GeV", 1),
    ("Lifetime",      "SR:G00:LIFETIME_T",    "hr",  3),
    ("Beam Current",  "SR:G00:BEAMCURRENT_T", "mA",  3),
    ("Linac Energy",  "LI:ENERGY",            "GeV", 1),
    ("Inj Eff (BPM)", "INJ:EFFI:BPM",         "%",   2),
    ("Inj Eff (SUM)", "INJ:EFFI:SUM",         "%",   2),
    ("Inj Rate",      "INJ:RATE",             "mA/s",3), # Added from new EDL
    ("Topup Ctr",     "BL:TOPUP:COUNT",        "s",   0),
]
ID_GAPS = [
    ("1C SFA",      "SR:ID1C:GAP_R"),
    ("2A EPU72",    "SR:ID2A:IN_GAP"),
    ("3A REVOLVER", "ID:3A:GAP"),
    ("3C SFA",      "SR:ID3C:UPSTREAM_GAP_RB"),
    ("4A EPU114",   "SR:ID4A:VAxis[0]_ActPos"),
    ("4C SFA",      "SR:ID4C:UPSTREAM_GAP_RB"),
    ("5A SFA",      "SR:ID5A:UPSTREAM_GAP_RB"),
    ("5C SFA",      "SR:ID5C:UPSTREAM_GAP_RB"),
    ("6A MPK EPU",  "SR:ID6A:IN_GAP"),
    ("6C MPW14",    "SR:ID6C:UPSTREAM_GAP_RB"),
    ("7A ADC",      "SR:ID7A:UPSTREAM_GAP_RB"),
    ("7C SFA",      "SR:ID7C:UPSTREAM_GAP_RB"),
    ("8A U68",      "SR:ID8A:GAP_RB"),
    ("8C SFA",      "SR:ID8C:UPSTREAM_GAP_RB"),
    ("9A SSRF",     "SR:ID9A:UPSTREAM_GAP_R"),
    ("9C SFA",      "SR:ID9C:UPSTREAM_GAP_RB"),
    ("10A EPU72",   "SR:ID10A:IN_GAP"),
    ("10C MPW10",   "SR:ID10C:GAP_RB"),
    ("11C SFA",     "SR:ID11C:UPSTREAM_GAP_RB"),
]
SHUTTER_GRID = {
    1:  [None,                "BL:MIS1BB01TB1SSS","BL:MIS1CB01TB2SSS","BL:MIS1DB01TB2SSS"],
    2:  ["BL:MIS2AB02TB1SSS", None,               "BL:MIS2CB02TB2SSS","BL:MIS2DB02TB2SSS"],
    3:  ["BL:MIS3AB03TB1SSS", None,               "BL:MIS3CB03TB2SSS","BL:MIS3DB03TB2SSS"],
    4:  ["BL:MIS4AB04TB1SSS","BL:MIS4BB04TB1SSS", "BL:MIS4CB04TB2SSS","BL:MIS4DB04TB2SSS"],
    5:  ["BL:MIS5AB05TB1SSS", None,               "BL:MIS5CB05TB2SSS","BL:MIS5DB05TB2SSS"],
    6:  ["BL:MIS6AB06TB1SSS", None,               "BL:MIS6CB06TB2SSS","BL:MIS6DB06TB2SSS"],
    7:  ["BL:MIS7AB07TB1SSS","BL:MIS7BB07TB1SSS", "BL:MIS7CB07TB2SSS","BL:MIS7DB07TB2SSS"],
    8:  ["BL:MIS8AB08TB1SSS", None,               "BL:MIS8CB08TB2SSS","BL:MIS8DB08TB2SSS"],
    9:  ["BL:MIS9AB09TB1SSS","BL:MIS9BB09TB1SSS", "BL:MIS9CB09TB2SSS","BL:MIS9DB09TB2SSS"],
    10: ["BL:MIS10AB10TB1SSS",None,               "BL:MIS10CB10TB2SSS","BL:MIS10DB10TB2SSS"],
    11: ["BL:MIS11AB11TB1SSS",None,               "BL:MIS11CB11TB2SSS","BL:MIS11DB11TB1SSS"],
    12: [None,None,None,None],
}
TUNE_BEAM = [
    ("Tune X",  "SR:G00:TUNE_X",    ""),
    ("Tune Y",  "SR:G00:TUNE_Y",    ""),
    ("BSize X", "SR:G00:BEAMSIZE_X","μm"),
    ("BSize Y", "SR:G00:BEAMSIZE_Y","μm"),
]
RF_MAIN = [
    ("Freq.",    "SR:RF:FREQ",        7, "MHz"),
    ("Gapvolt.", "SR:RF:GAP_VOLTAGE", 3, "MV"),
]
RF_CAVITIES = [
    {"n":"#1","gap":"SR:RF:FCGAPV1","phase":"SR:RF:LRF1:PMES","deta":"SR:RF:LRF1:DETA"},
    {"n":"#2","gap":"SR:RF:FCGAPV2","phase":"SR:RF:LRF2:PMES","deta":"SR:RF:LRF2:DETA"},
    {"n":"#3","gap":"SR:RF:FCGAPV3","phase":"SR:RF:LRF3:PMES","deta":"SR:RF:LRF3:DETA"},
]
CELL_TYPES  = ["CPST","BG_ST","VACIP_ALLOK","MPS_INT","MGN_INT"]
CELL_LABELS = ["Cool","BmGate","Vacuum","MPS","Magnet"]
def cell_pv(n, t):
    nn = f"{n:02d}"
    return {"CPST":f"SR:MISC{nn}CPST","BG_ST":f"SR:MISC{nn}BG_ST",
            "VACIP_ALLOK":f"SR:MISVACIP{nn}ALLOK","MPS_INT":f"SR:MISMPS{nn}_INT",
            "MGN_INT":f"SR:MISMGN{nn}_INT"}[t]
INTERLOCKS = [
    ("PRE-INJ","LI:MISPIJ_MPSST"),   ("LINAC",  "LI:MISLINAC_MPSST"),
    ("BTL MPS","LI:MISBTL_MPSST"),   ("BTL BD", "LI:MISBTL_MPSINT"),
    ("SR INJ", "SR:MISSR_INJ_ST"),   ("SR BD",  "SR:MISMPSBLDG_ST"),
    ("RF",     "SR:MISRFINT01DI1"),  ("BPM",    "SR:MISIDBPMALLOK"),
    ("LCW",    "SR:MISSR_LCW_ST"),   ("G/V",    "SR:MISGVALLST"),
    ("IDA IR1","SR:MISMPS4AIDASCIR1"),
]
PBPM_GRAPHS = [
    ("3C Vert", "BL:PBPM:3C:signals:sa.Y",   600),
    ("5A Vert", "BL:PBPM:5A:VERT",           600),
    ("9A Vert", "BL:PBPM:9A:SA:SA_Y_MONITOR",600),
    ("10D Vert","BL:PBPM:10D:VERT",          600),
]
HEARTBEATS = [("SOFB","SR:SOFB:HEART_1"), ("FOFB","SR:FOFB:HEART")]

# ── BPM: EXACT same PV pattern as v4 which worked ──────────────
LINAC_BPM = [(f"L{i:02d}", f"LI:BPM_L{i:02d}:X", f"LI:BPM_L{i:02d}:Y") for i in range(1,11)]
BTL_BPM   = [(f"T{i:02d}", f"LI:BPM_T{i:02d}:X", f"LI:BPM_T{i:02d}:Y") for i in range(1,15)]
BPM_THRESHOLD = 3.0

INJ_BEAM = [("Charge","LI:AVG_PC",3),("Jitter","LI:energy_Jitter",3),("Spread","LI:BTOTR1:Energy_Spread",3)]
BCUR_PV = "SR:G00:BEAMCURRENT_T"

# ═══════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════
BG    = '#07090f'; PANEL = '#0d1120'; PANEL2= '#0f1525'; BORDER= '#1a2540'
HDR   = '#0a1530'; GOLD  = '#F5C518'; CYAN  = '#22D3EE'; LIME  = '#4ADE80'
WHITE = '#E2E8F0'; GRAY  = '#4B5563'; DIM   = '#1F2937'
GREEN = '#16A34A'; RED   = '#DC2626'; YELLOW= '#CA8A04'
BLUE  = '#2563EB'; AMBER = '#D97706'; PINK  = '#DB2777'
TEAL  = '#2DD4BF'; LBLUE = '#60A5FA'

# ═══════════════════════════════════════════════════════════════
# PV MANAGER
# ═══════════════════════════════════════════════════════════════
class PVManager:
    def __init__(self):
        self.values  = {}
        self.history = defaultdict(lambda: deque(maxlen=1200))
        self._pvs    = {}
        self._dirty  = set()
        self._lock   = threading.Lock()   # EPICS callbacks come from bg threads

    def get(self, pv):            return self.values.get(pv)

    def get_history(self, pv, n):
        """Return last *n* items without copying the entire deque."""
        d = self.history[pv]
        if len(d) <= n:
            return list(d)
        return list(d)[-n:]              # still bounded, never > 1200

    def take_dirty(self):
        """Atomically swap dirty set and return old one."""
        with self._lock:
            d = self._dirty
            self._dirty = set()
            return d

    def connect_all(self, pvs):
        for p in pvs:
            if p and p not in self._pvs:
                try:
                    self._pvs[p] = EpicsPV(p, callback=self._cb, auto_monitor=True)
                except Exception as e:
                    log.warning('PV connect failed: %s – %s', p, e)

    def _cb(self, pvname=None, value=None, **kw):
        if pvname and value is not None:
            with self._lock:
                self.values[pvname] = value
                self.history[pvname].append(value)
                self._dirty.add(pvname)

# ═══════════════════════════════════════════════════════════════
# WIDGETS
# ═══════════════════════════════════════════════════════════════
class SectionFrame(tk.Frame):
    """Panel with title bar and optional scrollable body."""
    def __init__(self, master, title, accent=CYAN, scrollable=False, **kw):
        super().__init__(master, bg=PANEL, highlightbackground=BORDER,
                         highlightthickness=1, **kw)
        bar = tk.Frame(self, bg=HDR, height=16); bar.pack(fill=tk.X); bar.pack_propagate(False)
        tk.Frame(bar, bg=accent, width=3).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(bar, text=f" {title}", font=('Consolas',7,'bold'),
                 bg=HDR, fg=accent, anchor='w').pack(side=tk.LEFT, padx=2, pady=1)
        if scrollable:
            container = tk.Frame(self, bg=PANEL)
            container.pack(fill=tk.BOTH, expand=True)
            self._canvas = tk.Canvas(container, bg=PANEL, highlightthickness=0)
            self._scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=self._canvas.yview)
            self.body = tk.Frame(self._canvas, bg=PANEL, padx=3, pady=2)
            self.body.bind('<Configure>', lambda e: self._canvas.configure(scrollregion=self._canvas.bbox('all')))
            self._canvas.create_window((0,0), window=self.body, anchor='nw')
            self._canvas.configure(yscrollcommand=self._scrollbar.set)
            self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            # mousewheel scrolling
            def _on_mousewheel(event):
                self._canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
            self._canvas.bind('<MouseWheel>', _on_mousewheel)
            self.body.bind('<MouseWheel>', _on_mousewheel)
            # also bind to children as they're created
            self._mw_func = _on_mousewheel
        else:
            self.body = tk.Frame(self, bg=PANEL, padx=3, pady=2)
            self.body.pack(fill=tk.BOTH, expand=True)
            self._mw_func = None

    def bind_scroll(self, widget):
        """Bind mousewheel scrolling to a child widget."""
        if self._mw_func:
            widget.bind('<MouseWheel>', self._mw_func)


class MiniPlot(tk.Frame):
    """Auto-resizing plot that fills its parent."""
    def __init__(self, master, title="", width=300, height=65, color=BLUE):
        super().__init__(master, bg=PANEL)
        self._title=title; self.plot_w=width; self.plot_h=height
        self._color=color; self._last_len=0; self._last_data=None
        self.canvas = tk.Canvas(self, width=width, height=height, bg='#050810', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self.canvas.bind('<Configure>', self._on_resize)

    def _on_resize(self, event):
        if event.width > 10 and event.height > 10:
            new_w, new_h = event.width, event.height
            if abs(new_w - self.plot_w) > 5 or abs(new_h - self.plot_h) > 5:
                self.plot_w = new_w
                self.plot_h = new_h
                if self._last_data:
                    self._last_len = 0  # force redraw
                    self.update_data(self._last_data)

    def update_data(self, data):
        if len(data)==self._last_len: return
        self._last_len=len(data)
        self._last_data=data
        if len(data)<2: return
        c=self.canvas; c.delete("all")
        pt,pb,pl,pr=13,3,36,3
        pw=self.plot_w-pl-pr; ph=self.plot_h-pt-pb
        if pw < 10 or ph < 10: return
        ymin,ymax=min(data),max(data)
        if ymax-ymin<1e-9: ymin-=1; ymax+=1
        yr=ymax-ymin
        for i in range(3):
            yy=pt+(i/2)*ph
            c.create_line(pl,yy,pl+pw,yy, fill='#111d30', dash=(2,4))
            c.create_text(pl-2,yy, text=f"{ymax-(i/2)*yr:.1f}", fill=GRAY, font=('Consolas',5), anchor='e')
        c.create_text(pl+2,5, text=self._title, fill=GRAY, font=('Consolas',6,'bold'), anchor='w')
        c.create_text(self.plot_w-pr-2,5, text=f"{data[-1]:.3f}", fill=GOLD, font=('Consolas',6,'bold'), anchor='e')
        n=len(data); poly=[]; pts=[]
        for i,v in enumerate(data):
            x=pl+(i/(n-1))*pw; y=pt+ph-((v-ymin)/yr)*ph
            poly.extend([x,y]); pts.extend([x,y])
        poly+=[pl+pw,pt+ph,pl,pt+ph]
        c.create_polygon(poly, fill='#162040', outline='')
        c.create_line(pts, fill=self._color, width=1.5, smooth=True)
        lx=pl+pw; ly=pt+ph-((data[-1]-ymin)/yr)*ph
        c.create_oval(lx-2,ly-2,lx+2,ly+2, fill=self._color, outline=WHITE, width=1)


class BPMStation(tk.Frame):
    """Compact BPM: label + X bar + Y bar + ±mm values. Same logic as v4."""
    BAR_W=20; BAR_H=50; YRANGE=25.0

    def __init__(self, master, label, x_pv, y_pv, threshold=3.0):
        super().__init__(master, bg=PANEL, padx=1)
        self._xpv=x_pv; self._ypv=y_pv; self._thresh=threshold; self._last={}

        tk.Label(self, text=label, font=('Consolas',8,'bold'), bg=PANEL, fg=CYAN).pack(pady=(1,0))
        bars=tk.Frame(self, bg=PANEL); bars.pack()
        self._xcv=tk.Canvas(bars, width=self.BAR_W, height=self.BAR_H, bg='#0a1020',
                            highlightthickness=1, highlightbackground='#1e3050')
        self._xcv.pack(side=tk.LEFT, padx=1)
        self._ycv=tk.Canvas(bars, width=self.BAR_W, height=self.BAR_H, bg='#0a1020',
                            highlightthickness=1, highlightbackground='#1e3050')
        self._ycv.pack(side=tk.LEFT, padx=1)
        self._xlbl=tk.Label(self, text="—", font=('Consolas',8), bg=PANEL, fg=TEAL,  width=5)
        self._xlbl.pack()
        self._ylbl=tk.Label(self, text="—", font=('Consolas',8), bg=PANEL, fg=LBLUE, width=5)
        self._ylbl.pack(pady=(0,1))
        self._draw_bar(self._xcv, None, TEAL)
        self._draw_bar(self._ycv, None, LBLUE)

    def _draw_bar(self, cv, val, base_color):
        h,w=self.BAR_H,self.BAR_W; mid=h//2
        cv.delete("all")
        cv.create_line(0,mid,w,mid, fill='#2a3a50', width=1)
        tf=mid-int((self._thresh/self.YRANGE)*mid)
        tn=mid+int((self._thresh/self.YRANGE)*mid)
        cv.create_line(0,tf,w,tf, fill='#4a2020', dash=(1,2))
        cv.create_line(0,tn,w,tn, fill='#4a2020', dash=(1,2))
        if val is None: return
        clamped=max(-self.YRANGE,min(self.YRANGE,val))
        tip=mid-int((clamped/self.YRANGE)*mid)
        y0,y1=min(mid,tip),max(mid,tip); y1=max(y1,y0+2)
        cv.create_rectangle(1,y0,w-1,y1, fill=RED if abs(val)>self._thresh else base_color, outline='')

    def update(self, pv, val):
        if self._last.get(pv)==val: return
        self._last[pv]=val
        alarm=val is not None and abs(val)>self._thresh
        if pv==self._xpv:
            self._draw_bar(self._xcv, val, TEAL)
            self._xlbl.config(text=f"{val:+.1f}" if val is not None else "—", fg=RED if alarm else TEAL)
        elif pv==self._ypv:
            self._draw_bar(self._ycv, val, LBLUE)
            self._ylbl.config(text=f"{val:+.1f}" if val is not None else "—", fg=RED if alarm else LBLUE)


class HeartbeatWidget(tk.Frame):
    """Inline heartbeat: label + large LED + status + EKG."""
    _FADE=8
    def __init__(self, master, label, pv, pvm):
        super().__init__(master, bg=PANEL, padx=6, pady=2)
        self.pv=pv; self.pvm=pvm
        self._last_val=None; self._flash_steps=0; self._dot_color=DIM
        self._trace=deque(maxlen=40)
        tk.Label(self, text=label, font=('Consolas',8,'bold'), bg=PANEL, fg=CYAN, width=5, anchor='w').pack(side=tk.LEFT)
        self._led=tk.Canvas(self, width=20, height=20, bg=PANEL, highlightthickness=0); self._led.pack(side=tk.LEFT, padx=(2,4))
        self._status=tk.Label(self, text="WAIT", font=('Consolas',8,'bold'), bg=PANEL, fg=GRAY, width=7, anchor='w'); self._status.pack(side=tk.LEFT)
        self._ekg=tk.Canvas(self, width=100, height=18, bg='#050810', highlightthickness=0); self._ekg.pack(side=tk.LEFT, padx=(4,0))
        self._draw(DIM,"WAIT",GRAY)

    def _draw(self, color, txt, tcol):
        if color!=self._dot_color:
            self._led.delete("all")
            if color not in (DIM,GRAY): self._led.create_oval(1,1,19,19, outline=color, width=1)
            self._led.create_oval(3,3,17,17, fill=color, outline='')
            self._dot_color=color
        self._status.config(text=txt, fg=tcol)
        c=self._ekg; c.delete("all")
        d=list(self._trace)
        if len(d)>=2:
            mn,mx=min(d),max(d)
            if mx-mn>1e-9:
                pts=[]; n=len(d)
                for i,v in enumerate(d): pts.extend([(i/(n-1))*100, 16-((v-mn)/(mx-mn))*14])
                c.create_line(pts, fill=GREEN, width=1.5, smooth=True)

    def _lerp(self, f):
        r=int(0x16+(0x1F-0x16)*(1-f)); g=int(0xA3+(0x29-0xA3)*(1-f)); b=int(0x4A+(0x37-0x4A)*(1-f))
        return f'#{r:02X}{g:02X}{b:02X}'

    def tick(self):
        v=self.pvm.get(self.pv); changed=False
        if v is not None:
            try:
                fv=float(v)
                if fv!=self._last_val: changed=True; self._last_val=fv; self._flash_steps=self._FADE; self._trace.append(fv)
            except: pass
        if not changed and self._flash_steps==0 and v is not None: return
        if self._flash_steps>0: self._flash_steps-=1
        if v is None:             self._draw(DIM,   "NO DATA",GRAY)
        elif self._flash_steps>0: self._draw(self._lerp(self._flash_steps/self._FADE),"  ALIVE",GREEN)
        else:                     self._draw(YELLOW,"STALLED",YELLOW)


# ═══════════════════════════════════════════════════════════════
# DOCKABLE PANEL — drag-to-move, corner-resize, auto-scale content
# ═══════════════════════════════════════════════════════════════
class DockablePanel(tk.Frame):
    """A floating panel that can be freely dragged, resized, and auto-scales content."""
    GRIP_SIZE = 12
    MIN_W = 80
    MIN_H = 40

    def __init__(self, master, name, title, accent=CYAN, width=300, height=200):
        super().__init__(master, bg=PANEL, highlightbackground=BORDER,
                         highlightthickness=1, width=width, height=height)
        self.pack_propagate(False)
        self._name = name
        self._accent = accent
        self._default_w = width
        self._default_h = height
        self._font_registry = []   # [(widget, family, base_size, weight)] for scaling
        self._last_scale = 1.0

        # ── Title bar (drag handle) ───────────────────────────
        bar = tk.Frame(self, bg=HDR, height=18, cursor='fleur')
        bar.pack(fill=tk.X); bar.pack_propagate(False)
        tk.Frame(bar, bg=accent, width=3).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(bar, text=f" {title}", font=('Consolas',7,'bold'),
                 bg=HDR, fg=accent, anchor='w').pack(side=tk.LEFT, padx=2, pady=1)
        # Drag bindings on bar
        bar.bind('<ButtonPress-1>', self._drag_start)
        bar.bind('<B1-Motion>', self._drag_move)
        for child in bar.winfo_children():
            child.bind('<ButtonPress-1>', self._drag_start)
            child.bind('<B1-Motion>', self._drag_move)

        # ── Body (content goes here) ──────────────────────────
        self.body = tk.Frame(self, bg=PANEL, padx=3, pady=2)
        self.body.pack(fill=tk.BOTH, expand=True)

        # ── Resize grip (bottom-right corner) ─────────────────
        grip = tk.Canvas(self, width=self.GRIP_SIZE, height=self.GRIP_SIZE,
                         bg=PANEL, highlightthickness=0, cursor='sizing')
        grip.place(relx=1.0, rely=1.0, anchor='se')
        gs = self.GRIP_SIZE
        grip.create_polygon(gs, 0, gs, gs, 0, gs, fill=GRAY, outline='')
        grip.create_line(gs-3, gs, gs, gs-3, fill=WHITE, width=1)
        grip.create_line(gs-6, gs, gs, gs-6, fill=WHITE, width=1)
        grip.create_line(gs-9, gs, gs, gs-9, fill=WHITE, width=1)
        grip.bind('<ButtonPress-1>', self._resize_start)
        grip.bind('<B1-Motion>', self._resize_move)

        self._drag_data = {}
        self.bind('<Configure>', self._on_configure)

    def register_fonts(self):
        """Call after content is built to register all Label fonts for scaling."""
        self._font_registry.clear()
        self._scan_fonts(self.body)

    def _scan_fonts(self, widget):
        try:
            font = widget.cget('font')
            if font and isinstance(font, str):
                # parse tkinter font string like 'Consolas 7 bold'
                parts = font.split()
                if len(parts) >= 2:
                    family = parts[0]
                    size = int(parts[1])
                    weight = parts[2] if len(parts) > 2 else ''
                    self._font_registry.append((widget, family, size, weight))
            elif font and isinstance(font, tuple):
                family = font[0]
                size = font[1] if len(font) > 1 else 8
                weight = font[2] if len(font) > 2 else ''
                self._font_registry.append((widget, family, size, weight))
        except: pass
        for child in widget.winfo_children():
            self._scan_fonts(child)

    def _on_configure(self, event):
        if event.widget != self: return
        w = event.width
        h = event.height
        scale = min(w / self._default_w, h / self._default_h)
        scale = max(0.4, min(3.0, scale))  # clamp
        if abs(scale - self._last_scale) < 0.05: return
        self._last_scale = scale
        self._apply_scale(scale)

    def _apply_scale(self, scale):
        for widget, family, base_size, weight in self._font_registry:
            try:
                new_size = max(4, int(base_size * scale))
                if weight:
                    widget.configure(font=(family, new_size, weight))
                else:
                    widget.configure(font=(family, new_size))
            except: pass

    def _drag_start(self, event):
        self._drag_data['x'] = event.x_root
        self._drag_data['y'] = event.y_root
        self.lift()

    def _drag_move(self, event):
        dx = event.x_root - self._drag_data['x']
        dy = event.y_root - self._drag_data['y']
        x = self.winfo_x() + dx
        y = self.winfo_y() + dy
        self.place(x=x, y=y)
        self._drag_data['x'] = event.x_root
        self._drag_data['y'] = event.y_root

    def _resize_start(self, event):
        self._drag_data['x'] = event.x_root
        self._drag_data['y'] = event.y_root
        self._drag_data['w'] = self.winfo_width()
        self._drag_data['h'] = self.winfo_height()

    def _resize_move(self, event):
        dx = event.x_root - self._drag_data['x']
        dy = event.y_root - self._drag_data['y']
        new_w = max(self.MIN_W, self._drag_data['w'] + dx)
        new_h = max(self.MIN_H, self._drag_data['h'] + dy)
        self.configure(width=int(new_w), height=int(new_h))

    def get_geometry(self):
        return {'x': self.winfo_x(), 'y': self.winfo_y(),
                'w': self.winfo_width(), 'h': self.winfo_height()}

    def set_geometry(self, x, y, w, h):
        self.configure(width=int(w), height=int(h))
        self.place(x=int(x), y=int(y))


# ═══════════════════════════════════════════════════════════════
# DEFAULT LAYOUT — positions/sizes for 1920x1080 (approximate)
# ═══════════════════════════════════════════════════════════════
DEFAULT_LAYOUT = {
    'metrics':   {'x': 4,   'y': 2,   'w': 860, 'h': 70},
    'beam':      {'x': 4,   'y': 75,  'w': 500, 'h': 160},
    'id_gaps':   {'x': 4,   'y': 238, 'w': 140, 'h': 280},
    'shutters':  {'x': 148, 'y': 238, 'w': 155, 'h': 280},
    'rf':        {'x': 307, 'y': 238, 'w': 197, 'h': 280},
    'bpm':       {'x': 4,   'y': 521, 'w': 500, 'h': 150},
    'cells':     {'x': 508, 'y': 75,  'w': 355, 'h': 285},
    'interlocks':{'x': 508, 'y': 363, 'w': 355, 'h': 100},
    'pbpm':      {'x': 508, 'y': 466, 'w': 355, 'h': 210},
}


# ═══════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════
class Dashboard:
    def __init__(self, root, pvm):
        self.root=root; self.pvm=pvm
        root.title("PLS-II Operation Status  v5")
        root.configure(bg=BG)
        root.state('zoomed')
        self.val_lbl={}; self.ind_cv={}; self.plots={}; self.bpm_widgets={}; self.hb_widgets=[]
        self._panels={}          # name -> DockablePanel
        self._connect_all()
        self._build()
        self._restore_layout()
        self._tick()
        root.bind('<Control-r>', lambda e: self._reset_layout())
        root.bind('<Control-R>', lambda e: self._reset_layout())
        root.protocol('WM_DELETE_WINDOW', self._on_close)

    def _on_close(self):
        self._save_layout()
        self.root.destroy()

    def _save_layout(self):
        cfg = {}
        for name, panel in self._panels.items():
            cfg[name] = panel.get_geometry()
        try:
            with open(LAYOUT_FILE, 'w') as f: json.dump(cfg, f, indent=2)
        except: pass

    def _restore_layout(self):
        if not os.path.exists(LAYOUT_FILE): return
        try:
            with open(LAYOUT_FILE) as f: cfg = json.load(f)
        except: return
        def apply():
            for name, geom in cfg.items():
                panel = self._panels.get(name)
                if panel and all(k in geom for k in ('x','y','w','h')):
                    panel.set_geometry(geom['x'], geom['y'], geom['w'], geom['h'])
        self.root.after(200, apply)

    def _reset_layout(self):
        try: os.remove(LAYOUT_FILE)
        except: pass
        for name, panel in self._panels.items():
            g = DEFAULT_LAYOUT.get(name, {'x':10,'y':10,'w':300,'h':200})
            panel.set_geometry(g['x'], g['y'], g['w'], g['h'])

    def _connect_all(self):
        pvs=set()
        for _,p,*_ in KEY_METRICS: pvs.add(p)
        for _,p in ID_GAPS:        pvs.add(p)
        for row in SHUTTER_GRID.values():
            for p in row:
                if p: pvs.add(p)
        for _,p,_ in TUNE_BEAM:    pvs.add(p)
        for _,p,_,_ in RF_MAIN:    pvs.add(p)
        for cav in RF_CAVITIES:    pvs.update([cav["gap"],cav["phase"],cav["deta"]])
        for c in range(1,13):
            for t in CELL_TYPES:   pvs.add(cell_pv(c,t))
        for _,p in INTERLOCKS:     pvs.add(p)
        for _,p,_ in PBPM_GRAPHS:  pvs.add(p)
        for _,p in HEARTBEATS:     pvs.add(p)
        for _,xp,yp in LINAC_BPM: pvs.update([xp,yp])
        for _,xp,yp in BTL_BPM:   pvs.update([xp,yp])
        for _,p,_ in INJ_BEAM:     pvs.add(p)
        pvs.add(BCUR_PV)
        self.pvm.connect_all(pvs)

    def _make_panel(self, name, title, accent=CYAN, **kw):
        g = DEFAULT_LAYOUT.get(name, {'x':10,'y':10,'w':300,'h':200})
        panel = DockablePanel(self._workspace, name, title, accent=accent,
                              width=g['w'], height=g['h'])
        panel.place(x=g['x'], y=g['y'])
        self._panels[name] = panel
        return panel

    def _build(self):
        self.content=tk.Frame(self.root, bg=BG)
        self.content.pack(fill=tk.BOTH, expand=True)

        self._build_header()
        self._build_subheader()   # Tune | Injection | Heartbeats

        # ── Workspace: a frame that holds all floating panels ──
        self._workspace = tk.Frame(self.content, bg=BG)
        self._workspace.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        # ── Create all floating panels ────────────────────────
        p = self._make_panel('metrics', 'Key Metrics', accent=GOLD)
        self._build_metrics_content(p.body)
        p.register_fonts()

        p = self._make_panel('beam', 'BEAM CURRENT HISTORY', accent=CYAN)
        self._beam_plot = MiniPlot(p.body, "", width=480, height=120, color=BLUE)
        self._beam_plot.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        p = self._make_panel('id_gaps', 'ID Gaps (mm)', accent=LIME)
        self._build_id_gaps_content(p.body)
        p.register_fonts()

        p = self._make_panel('shutters', 'Shutters  A·B·C·D', accent=AMBER)
        self._build_shutters_content(p.body)
        p.register_fonts()

        p = self._make_panel('rf', 'RF System', accent=PINK)
        self._build_rf_content(p.body)
        p.register_fonts()

        p = self._make_panel('bpm', f'BPM  ·  teal=X  blue=Y  red=|val|>{BPM_THRESHOLD:.0f}mm', accent=BLUE)
        self._build_bpm_content(p.body)
        p.register_fonts()

        p = self._make_panel('cells', 'Cell Status (1–12)', accent=LIME)
        self._build_cells_content(p.body)
        p.register_fonts()

        p = self._make_panel('interlocks', 'Interlocks', accent=RED)
        self._build_interlocks_content(p.body)
        p.register_fonts()

        p = self._make_panel('pbpm', 'BeamLine PBPM History', accent=AMBER)
        self._build_pbpm_content(p.body)

    # ── Header ───────────────────────────────────────────────
    def _build_header(self):
        h=tk.Frame(self.content, bg='#080d1e', height=36)
        h.pack(fill=tk.X, padx=3, pady=(3,1)); h.pack_propagate(False)
        lf=tk.Frame(h, bg='#080d1e'); lf.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(lf, text="⚡", font=('Segoe UI',16), bg='#080d1e', fg=GOLD).pack(side=tk.LEFT, padx=(8,2))
        tf=tk.Frame(lf, bg='#080d1e'); tf.pack(side=tk.LEFT)
        tk.Label(tf, text="PLS-II  OPERATION STATUS", font=('Consolas',12,'bold'), bg='#080d1e', fg=WHITE).pack(anchor='w')
        tk.Label(tf, text="Pohang Light Source — Live EPICS", font=('Consolas',6), bg='#080d1e', fg=GRAY).pack(anchor='w')
        self.time_lbl=tk.Label(h, text="", font=('Consolas',11,'bold'), bg='#080d1e', fg=CYAN)
        self.time_lbl.pack(side=tk.RIGHT, padx=10)
        tk.Frame(self.content, bg=CYAN, height=1).pack(fill=tk.X, padx=3)

    def _build_metrics_content(self, parent):
        """Key metrics as floating panel content."""
        for i,(label,pv,unit,prec) in enumerate(KEY_METRICS):
            if i: tk.Frame(parent, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=2)
            f=tk.Frame(parent, bg=PANEL, padx=6, pady=2); f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            tk.Label(f, text=label, font=('Consolas',6), bg=PANEL, fg=GRAY).pack()
            vl=tk.Label(f, text="—", font=('Consolas',16,'bold'), bg=PANEL, fg=LIME); vl.pack()
            tk.Label(f, text=unit, font=('Consolas',6), bg=PANEL, fg=DIM).pack()
            self.val_lbl[pv]=[vl,prec,None]

    def _build_subheader(self):
        """Full-width single row: Tune & Beam | Injection | Heartbeats"""
        bar=tk.Frame(self.content, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        bar.pack(fill=tk.X, padx=3, pady=(0,2))

        def div(): tk.Frame(bar, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=3)

        tk.Frame(bar, bg=AMBER, width=3).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(bar, text=" Tune & Beam ", font=('Consolas',7,'bold'), bg=PANEL, fg=CYAN).pack(side=tk.LEFT)
        for label,pv,unit in TUNE_BEAM:
            tk.Label(bar, text=label, font=('Consolas',6), bg=PANEL, fg=GRAY).pack(side=tk.LEFT, padx=(5,1))
            vl=tk.Label(bar, text="—", font=('Consolas',9,'bold'), bg=PANEL, fg=CYAN); vl.pack(side=tk.LEFT)
            if unit: tk.Label(bar, text=unit, font=('Consolas',6), bg=PANEL, fg=GRAY).pack(side=tk.LEFT, padx=(1,3))
            self.val_lbl[pv]=[vl,4,None]

        div()
        tk.Label(bar, text=" Injection ", font=('Consolas',7,'bold'), bg=PANEL, fg=AMBER).pack(side=tk.LEFT)
        for label,pv,prec in INJ_BEAM:
            tk.Label(bar, text=label, font=('Consolas',6), bg=PANEL, fg=GRAY).pack(side=tk.LEFT, padx=(4,1))
            vl=tk.Label(bar, text="—", font=('Consolas',9,'bold'), bg=PANEL, fg=AMBER); vl.pack(side=tk.LEFT, padx=(0,3))
            self.val_lbl[pv]=[vl,prec,None]

        div()
        tk.Label(bar, text=" Heartbeats ", font=('Consolas',7,'bold'), bg=PANEL, fg=GREEN).pack(side=tk.LEFT)
        for label,pv in HEARTBEATS:
            hw=HeartbeatWidget(bar, label, pv, self.pvm)
            hw.pack(side=tk.LEFT)
            self.hb_widgets.append(hw)

    # ── Panel content builders ───────────────────────────────
    def _build_id_gaps_content(self, parent):
        canvas = tk.Canvas(parent, bg=PANEL, highlightthickness=0)
        sb = tk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg=PANEL)
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0,0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        def mw(event): canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
        canvas.bind('<MouseWheel>', mw); inner.bind('<MouseWheel>', mw)
        for label,pv in ID_GAPS:
            r=tk.Frame(inner, bg=PANEL); r.pack(fill=tk.X)
            l=tk.Label(r, text=label, font=('Consolas',6), bg=PANEL, fg=GRAY, width=12, anchor='w')
            l.pack(side=tk.LEFT); l.bind('<MouseWheel>', mw)
            vl=tk.Label(r, text="—", font=('Consolas',8,'bold'), bg='#050810', fg=LIME, width=6, anchor='e', padx=1)
            vl.pack(side=tk.RIGHT); vl.bind('<MouseWheel>', mw)
            r.bind('<MouseWheel>', mw)
            self.val_lbl[pv]=[vl,1,None]

    def _build_shutters_content(self, parent):
        canvas = tk.Canvas(parent, bg=PANEL, highlightthickness=0)
        sb = tk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg=PANEL)
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0,0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        def mw(event): canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
        canvas.bind('<MouseWheel>', mw); inner.bind('<MouseWheel>', mw)
        hdr=tk.Frame(inner, bg=PANEL); hdr.pack(fill=tk.X)
        tk.Label(hdr, text="#", font=('Consolas',6,'bold'), bg=PANEL, fg=GRAY, width=3).pack(side=tk.LEFT)
        for c in ['A','B','C','D']:
            tk.Label(hdr, text=c, font=('Consolas',6,'bold'), bg=PANEL, fg=AMBER, width=4).pack(side=tk.LEFT)
        for cell in range(1,13):
            row=tk.Frame(inner, bg=PANEL); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=str(cell), font=('Consolas',6,'bold'), bg=PANEL, fg=GOLD, width=3).pack(side=tk.LEFT)
            row.bind('<MouseWheel>', mw)
            for pv in SHUTTER_GRID[cell]:
                if not pv:
                    tk.Frame(row, width=26, height=9, bg=DIM).pack(side=tk.LEFT, padx=1)
                else:
                    cv=tk.Canvas(row, width=26, height=9, bg='#1a2030', highlightthickness=0)
                    cv.pack(side=tk.LEFT, padx=1)
                    cv.bind('<MouseWheel>', mw)
                    self.ind_cv[f"SH:{pv}"]=[cv,None]

    def _build_rf_content(self, parent):
        canvas = tk.Canvas(parent, bg=PANEL, highlightthickness=0)
        sb = tk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg=PANEL)
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0,0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        def mw(event): canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
        canvas.bind('<MouseWheel>', mw); inner.bind('<MouseWheel>', mw)
        for label,pv,prec,unit in RF_MAIN:
            r=tk.Frame(inner, bg=PANEL); r.pack(fill=tk.X, pady=1)
            tk.Label(r, text=label, font=('Consolas',7), bg=PANEL, fg=WHITE, anchor='w').pack(side=tk.LEFT)
            tk.Label(r, text=unit, font=('Consolas',6), bg=PANEL, fg=DIM).pack(side=tk.RIGHT)
            vl=tk.Label(r, text="—", font=('Consolas',9,'bold'), bg='#050810', fg=PINK, anchor='e', padx=2)
            vl.pack(side=tk.RIGHT)
            r.bind('<MouseWheel>', mw)
            self.val_lbl[pv]=[vl,prec,None]
        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=2)
        for cav in RF_CAVITIES:
            cl=tk.Label(inner, text=f"Cav {cav['n']}", font=('Consolas',7,'bold'), bg=PANEL, fg=PINK)
            cl.pack(anchor='w', pady=(2,0)); cl.bind('<MouseWheel>', mw)
            for sub,key in [("Gap","gap"),("Phase","phase"),("DETA","deta")]:
                pv=cav[key]
                r=tk.Frame(inner, bg=PANEL); r.pack(fill=tk.X)
                tk.Label(r, text=f"  {sub}", font=('Consolas',6), bg=PANEL, fg=GRAY, anchor='w').pack(side=tk.LEFT)
                led=tk.Canvas(r, width=7, height=7, bg=PANEL, highlightthickness=0); led.pack(side=tk.RIGHT, padx=2)
                self.ind_cv[f"RF:{pv}"]=[led,None]
                vl=tk.Label(r, text="—", font=('Consolas',7,'bold'), bg='#050810', fg=LIME, anchor='e', padx=2)
                vl.pack(side=tk.RIGHT)
                r.bind('<MouseWheel>', mw)
                self.val_lbl[pv]=[vl,2,None]

    def _build_bpm_content(self, parent):
        def make_row(group_label, stations):
            outer=tk.Frame(parent, bg=PANEL); outer.pack(fill=tk.X, pady=(1,0))
            tk.Label(outer, text=group_label, font=('Consolas',6,'bold'),
                     bg=PANEL, fg=GRAY, width=6, anchor='w').pack(side=tk.LEFT, padx=(0,2))
            for lbl,xpv,ypv in stations:
                s=BPMStation(outer, lbl, xpv, ypv, threshold=BPM_THRESHOLD)
                s.pack(side=tk.LEFT, padx=1)
                self.bpm_widgets[xpv]=s
                self.bpm_widgets[ypv]=s
        make_row("LINAC", LINAC_BPM)
        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, pady=1)
        make_row(" BTL ", BTL_BPM)

    def _build_cells_content(self, parent):
        hdr=tk.Frame(parent, bg=PANEL); hdr.pack(fill=tk.X)
        tk.Label(hdr, text=" #", font=('Consolas',6,'bold'), bg=PANEL, fg=GRAY, width=3, anchor='w').pack(side=tk.LEFT)
        for sl in CELL_LABELS:
            tk.Label(hdr, text=sl, font=('Consolas',6,'bold'), bg=PANEL, fg=CYAN, width=7, anchor='center').pack(side=tk.LEFT, padx=1)
        for c in range(1,13):
            row=tk.Frame(parent, bg=PANEL); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"{c:2d}", font=('Consolas',7,'bold'), bg=PANEL, fg=GOLD, width=3, anchor='e').pack(side=tk.LEFT)
            for t in CELL_TYPES:
                pv=cell_pv(c,t)
                lbl=tk.Label(row, text="", bg=DIM, width=7, height=1, relief='flat')
                lbl.pack(side=tk.LEFT, padx=2, pady=1)
                self.ind_cv[pv]=[lbl,None]

    def _build_interlocks_content(self, parent):
        grid=tk.Frame(parent, bg=PANEL); grid.pack(anchor='w')
        for i,(label,pv) in enumerate(INTERLOCKS):
            sub=tk.Frame(grid, bg=PANEL)
            sub.grid(row=i//3, column=i%3, padx=4, pady=3, sticky='w')
            ind=tk.Label(sub, text="", bg=DIM, width=3, height=1, relief='flat'); ind.pack(side=tk.LEFT)
            tk.Label(sub, text=label, font=('Consolas',7), bg=PANEL, fg=WHITE).pack(side=tk.LEFT, padx=3)
            self.ind_cv[pv]=[ind,None]

    def _build_pbpm_content(self, parent):
        r1=tk.Frame(parent, bg=PANEL); r1.pack(fill=tk.BOTH, expand=True)
        r2=tk.Frame(parent, bg=PANEL); r2.pack(fill=tk.BOTH, expand=True)
        for i,(title,pv,npts) in enumerate(PBPM_GRAPHS):
            p=MiniPlot([r1,r1,r2,r2][i], title, width=160, height=60, color=AMBER)
            p.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,2 if i%2==0 else 0))
            self.plots[pv]=p

    # ── TICK ─────────────────────────────────────────────────
    def _tick(self):
        try:
            self._tick_inner()
        except Exception as e:
            log.exception('Tick error: %s', e)
        self.root.after(500, self._tick)   # always reschedule

    def _tick_inner(self):
        self.time_lbl.config(text=time.strftime("%Y-%m-%d  %H:%M:%S"))
        dirty = self.pvm.take_dirty()      # thread-safe swap
        if not dirty:
            # still tick heartbeats (they track staleness)
            for hw in self.hb_widgets: hw.tick()
            return

        for pv,(lbl,prec,last) in self.val_lbl.items():
            if pv not in dirty: continue
            v=self.pvm.get(pv)
            if v is None: continue
            try:    txt=f"{float(v):.{prec}f}"
            except (TypeError, ValueError): txt=str(v)
            if txt!=last: lbl.config(text=txt); self.val_lbl[pv][2]=txt

        for key,(widget,last_color) in self.ind_cv.items():
            if key.startswith("RF:"):
                pv=key[3:]
                if pv not in dirty: continue
                v=self.pvm.get(pv); color=DIM
                if v is not None:
                    try: color=GREEN if abs(float(v))<100 else YELLOW
                    except (TypeError, ValueError): pass
                if color==last_color: continue
                widget.delete("all"); widget.create_oval(1,1,6,6, fill=color, outline='')
                self.ind_cv[key][1]=color
            elif key.startswith("SH:"):
                pv=key[3:]
                if pv not in dirty: continue
                v=self.pvm.get(pv); color=DIM
                if v is not None:
                    try: color=RED if int(v)==1 else GREEN
                    except (TypeError, ValueError): pass
                if color==last_color: continue
                widget.delete("all")
                widget.create_rectangle(0,0,26,9, fill=color, outline='')
                widget.create_text(13,4.5, text='CLS' if color==RED else 'OPN', fill=BG, font=('Consolas',5,'bold'))
                self.ind_cv[key][1]=color
            else:
                if key not in dirty: continue
                v=self.pvm.get(key); color=DIM
                if v is not None:
                    try: color=GREEN if int(v)==1 else RED
                    except (TypeError, ValueError): pass
                if color==last_color: continue
                widget.config(bg=color); self.ind_cv[key][1]=color

        for hw in self.hb_widgets: hw.tick()

        # Beam history — only update when new data arrived
        if BCUR_PV in dirty:
            hist=self.pvm.get_history(BCUR_PV,1200)
            if hist: self._beam_plot.update_data(hist)

        for _,pv,n in PBPM_GRAPHS:
            if pv in dirty and pv in self.plots:
                h=self.pvm.get_history(pv,n)
                if h: self.plots[pv].update_data(h)

        for pv,station in self.bpm_widgets.items():
            if pv not in dirty: continue
            v=self.pvm.get(pv)
            if v is not None:
                try: station.update(pv,float(v))
                except (TypeError, ValueError): pass




if __name__=="__main__":
    pvm=PVManager()
    root=tk.Tk()
    app=Dashboard(root,pvm)
    root.mainloop()