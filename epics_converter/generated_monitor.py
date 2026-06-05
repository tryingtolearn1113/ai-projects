import sys
import random
import time
from datetime import datetime

# Optional PyEpics Import
try:
    import epics
    EPICS_AVAILABLE = True
except ImportError:
    EPICS_AVAILABLE = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QTabWidget, QLabel, QPushButton, QFrame, 
    QScrollArea, QStatusBar, QSizePolicy, QGroupBox
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QFont, QPalette, QPainter, QRadialGradient

# -----------------------------------------------------------------------------
# CONSTANTS & STYLES
# -----------------------------------------------------------------------------
WINDOW_TITLE = "SYNCHROTRON CONTROL SYSTEM MONITOR"
FACILITY_NAME = "PAL-II Synchrotron Light Source"

DARK_STYLE = """
    QMainWindow {
        background-color: #121212;
    }
    QWidget {
        background-color: #1a1a1a;
        color: #e0e0e0;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    QTabWidget::pane {
        border: 1px solid #333333;
        background: #1a1a1a;
        margin-top: -1px;
    }
    QTabBar::tab {
        background: #252525;
        border: 1px solid #333333;
        padding: 8px 16px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        color: #a0a0a0;
        font-weight: bold;
    }
    QTabBar::tab:selected {
        background: #1a1a1a;
        border-bottom-color: #1a1a1a;
        color: #00ffcc;
    }
    QTabBar::tab:hover {
        background: #2d2d2d;
        color: #ffffff;
    }
    QGroupBox {
        border: 1px solid #3a3a3a;
        border-radius: 6px;
        margin-top: 12px;
        font-weight: bold;
        font-size: 13px;
        color: #00ffcc;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 5px;
    }
    QScrollArea {
        border: none;
        background-color: #1a1a1a;
    }
    QScrollBar:vertical {
        border: none;
        background: #151515;
        width: 10px;
        margin: 0px;
    }
    QScrollBar::handle:vertical {
        background: #333;
        min-height: 20px;
        border-radius: 5px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        border: none;
        background: none;
    }
"""

# -----------------------------------------------------------------------------
# THEME COLORS
# -----------------------------------------------------------------------------
COLOR_BG_CARD = "#222222"
COLOR_BORDER_CARD = "#333333"
COLOR_VAL_OK = "#00ff66"
COLOR_VAL_WARN = "#ffcc00"
COLOR_VAL_ALARM = "#ff3333"
COLOR_TEXT_MUTED = "#888888"

# -----------------------------------------------------------------------------
# CORE EPICS/SIMULATION MANAGER
# -----------------------------------------------------------------------------
class EPICSManager(QObject):
    pv_updated = pyqtSignal(str, object)

    def __init__(self, use_simulator=True):
        super().__init__()
        self.use_simulator = use_simulator
        self.pvs = {}
        self.pv_values = {}
        
        # Internal cache of PV properties for simulation
        self.sim_configs = {}

    def register_pv(self, pv_name, default_val=0.0, sim_type="static", bounds=(0, 100)):
        self.pv_values[pv_name] = default_val
        self.sim_configs[pv_name] = {"type": sim_type, "bounds": bounds, "val": default_val}
        
        if not self.use_simulator and EPICS_AVAILABLE:
            try:
                # Setup non-blocking Epics PV connection
                self.pvs[pv_name] = epics.PV(pv_name, callback=self._epics_callback)
            except Exception as e:
                print(f"Error connecting to PV {pv_name}: {e}")

    def _epics_callback(self, pvname=None, value=None, **kwargs):
        if pvname and value is not None:
            self.pv_values[pvname] = value
            self.pv_updated.emit(pvname, value)

    def get_value(self, pv_name):
        return self.pv_values.get(pv_name, 0.0)

    def set_simulator_mode(self, enabled):
        self.use_simulator = enabled
        if not enabled and EPICS_AVAILABLE:
            # Reconnect all real PVs if switching to EPICS
            for pv_name in self.pv_values.keys():
                if pv_name not in self.pvs:
                    try:
                        self.pvs[pv_name] = epics.PV(pv_name, callback=self._epics_callback)
                    except Exception as e:
                        print(f"Error: {e}")

    def simulation_step(self):
        """Generates realistic physical drifts and behaviors for the EPICS PVs"""
        if not self.use_simulator:
            return

        for pv, cfg in self.sim_configs.items():
            t = cfg["type"]
            b = cfg["bounds"]
            curr = cfg["val"]

            if t == "static":
                new_val = curr
            elif t == "noise":
                # White noise centered around midpoint
                mid = (b[0] + b[1]) / 2.0
                span = b[1] - b[0]
                new_val = curr + random.normalvariate(0, span * 0.01)
                new_val = max(b[0], min(b[1], new_val))
            elif t == "decay":
                # Simulated beam current decay
                new_val = curr - random.uniform(0.001, 0.005)
                if new_val < b[0]:
                    new_val = b[1] # "Injection" reset
            elif t == "sinusoid":
                # Smooth variation over time
                new_val = ((b[1] - b[0]) / 2.0) * (1.0 + 0.1 * random.uniform(-1, 1)) + b[0]
            elif t == "binary_ok":
                # Mostly 1 (OK), very rarely trips to 0
                new_val = 1 if random.random() > 0.002 else 0
            elif t == "binary_random":
                new_val = random.choice([0, 1])
            else:
                new_val = curr

            self.sim_configs[pv]["val"] = new_val
            self.pv_values[pv] = new_val
            self.pv_updated.emit(pv, new_val)

# -----------------------------------------------------------------------------
# CUSTOM WIDGETS
# -----------------------------------------------------------------------------
class LEDWidget(QWidget):
    """Custom circular vector LED showing system status state"""
    def __init__(self, parent=None, size=18):
        super().__init__(parent)
        self.setMinimumSize(size, size)
        self.setMaximumSize(size, size)
        self.color = QColor(100, 100, 100) # Default gray

    def set_state(self, status):
        """Set state: 'ok', 'warning', 'alarm', 'off'"""
        if status == 'ok':
            self.color = QColor(0, 255, 102) # Bright Green
        elif status == 'warning':
            self.color = QColor(255, 204, 0) # Amber
        elif status == 'alarm':
            self.color = QColor(255, 51, 51) # Red
        else:
            self.color = QColor(60, 60, 60) # Dark Gray
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Glowing effect with radial gradient
        gradient = QRadialGradient(self.width()/2, self.height()/2, self.width()/2)
        gradient.setColorAt(0.0, QColor(255, 255, 255, 200))
        gradient.setColorAt(0.2, self.color)
        gradient.setColorAt(1.0, self.color.darker(250))
        
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(1, 1, self.width()-2, self.height()-2)

class PVCard(QFrame):
    """Modern modular block to display single PV value, description and limits"""
    def __init__(self, title, pv_name, unit="", format_str="{:.3f}", parent=None):
        super().__init__(parent)
        self.pv_name = pv_name
        self.format_str = format_str
        self.unit = unit

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            PVCard {{
                background-color: {COLOR_BG_CARD};
                border: 1px solid {COLOR_BORDER_CARD};
                border-radius: 4px;
            }}
            PVCard:hover {{
                border-color: #555555;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px; font-weight: bold;")
        self.title_lbl.setWordWrap(True)

        self.val_lbl = QLabel("---")
        self.val_lbl.setStyleSheet(f"color: {COLOR_VAL_OK}; font-size: 18px; font-weight: bold;")
        
        self.pv_lbl = QLabel(pv_name)
        self.pv_lbl.setStyleSheet(f"color: #555555; font-size: 9px;")
        self.pv_lbl.setWordWrap(True)

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.val_lbl)
        layout.addWidget(self.pv_lbl)

    def update_value(self, val):
        if val is None:
            self.val_lbl.setText("---")
            return
            
        try:
            val_formatted = self.format_str.format(val)
            self.val_lbl.setText(f"{val_formatted} {self.unit}")
        except Exception:
            self.val_lbl.setText(str(val))

class StatusRow(QWidget):
    """Simple row combining an LED indicator, Description and short PV identifier"""
    def __init__(self, description, pv_name, invert_logic=False, parent=None):
        super().__init__(parent)
        self.pv_name = pv_name
        self.invert_logic = invert_logic # e.g. status code 0 means active error, or 1 means OK.

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(10)

        self.led = LEDWidget(self, size=14)
        self.led.set_state('off')

        self.desc_lbl = QLabel(description)
        self.desc_lbl.setStyleSheet("font-size: 12px; color: #d0d0d0;")
        
        self.pv_lbl = QLabel(pv_name)
        self.pv_lbl.setStyleSheet("font-size: 9px; color: #555555;")
        self.pv_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.led)
        layout.addWidget(self.desc_lbl)
        layout.addStretch()
        layout.addWidget(self.pv_lbl)

    def update_state(self, val):
        is_ok = (int(val) != 0) if not self.invert_logic else (int(val) == 0)
        self.led.set_state('ok' if is_ok else 'alarm')

# -----------------------------------------------------------------------------
# MAIN APPLICATION INTERFACE
# -----------------------------------------------------------------------------
class MonitorDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1280, 800)
        self.setStyleSheet(DARK_STYLE)

        # Initialize EPICS Manager with Simulator ON by default
        self.manager = EPICSManager(use_simulator=True)
        
        # Structure track tables/widgets
        self.pv_widgets = {}

        self.init_pv_definitions()
        self.init_ui()

        # Update Timers
        self.data_timer = QTimer(self)
        self.data_timer.timeout.connect(self.manager.simulation_step)
        self.data_timer.start(500) # EPICS / Simulation refresh rate at 500ms

        self.gui_timer = QTimer(self)
        self.gui_timer.timeout.connect(self.update_gui)
        self.gui_timer.start(250) # GUI drawing timer

    def init_pv_definitions(self):
        """Map actual PV list variables to target physical profiles and models"""
        
        # System Core PVs
        self.manager.register_pv("SR:G00:BEAMCURRENT_T", default_val=399.85, sim_type="decay", bounds=(350, 400))
        self.manager.register_pv("SR:G00:LIFETIME_T", default_val=18.42, sim_type="noise", bounds=(15.0, 20.0))
        self.manager.register_pv("SR:ENERGY", default_val=3.0, sim_type="static")
        self.manager.register_pv("TOPUP:COUNT", default_val=120, sim_type="decay", bounds=(0, 180))
        self.manager.register_pv("INJ:RATE", default_val=1.24, sim_type="noise", bounds=(0.5, 2.5))
        self.manager.register_pv("INJ:EFFI:BPM", default_val=98.4, sim_type="noise", bounds=(95, 100))
        
        # Storage Ring Physics
        self.manager.register_pv("SR:G00:TUNE_X", default_val=0.228, sim_type="noise", bounds=(0.22, 0.23))
        self.manager.register_pv("SR:G00:TUNE_Y", default_val=0.174, sim_type="noise", bounds=(0.17, 0.18))
        self.manager.register_pv("SR:G00:BEAMSIZE_X", default_val=124.5, sim_type="noise", bounds=(120.0, 130.0))
        self.manager.register_pv("SR:G00:BEAMSIZE_Y", default_val=12.2, sim_type="noise", bounds=(10.0, 14.0))
        self.manager.register_pv("SR:FOFB:HEART", default_val=1, sim_type="binary_ok")
        self.manager.register_pv("SR:MISIDBPMALLOK", default_val=1, sim_type="binary_ok")
        self.manager.register_pv("SR:MISGVALLST", default_val=1, sim_type="binary_ok")

        # Linac Parameters
        self.manager.register_pv("LI:ENERGY", default_val=3.0, sim_type="static")
        self.manager.register_pv("LI:AVG_PC", default_val=1.52, sim_type="noise", bounds=(1.4, 1.7))
        self.manager.register_pv("LI:energy_Jitter", default_val=0.04, sim_type="noise", bounds=(0.01, 0.08))
        self.manager.register_pv("LI:BTOTR1:Energy_Spread", default_val=0.12, sim_type="noise", bounds=(0.08, 0.15))
        
        # MPS Systems
        self.manager.register_pv("LI:MISLINAC_MPSST", default_val=1, sim_type="binary_ok")
        self.manager.register_pv("LI:MISBTL_MPSINT", default_val=1, sim_type="binary_ok")
        self.manager.register_pv("LI:MISPIJ_MPSST", default_val=1, sim_type="binary_ok")

        # BPM positions (Linac L01-L10 and Transfer T01-T14)
        for i in range(1, 11):
            self.manager.register_pv(f"LI:BPM_L{i:02d}:X", default_val=0.01, sim_type="noise", bounds=(-0.5, 0.5))
            self.manager.register_pv(f"LI:BPM_L{i:02d}:Y", default_val=-0.02, sim_type="noise", bounds=(-0.5, 0.5))

        for i in range(1, 15):
            self.manager.register_pv(f"LI:BPM_T{i:02d}:X", default_val=-0.03, sim_type="noise", bounds=(-0.5, 0.5))
            self.manager.register_pv(f"LI:BPM_T{i:02d}:Y", default_val=0.04, sim_type="noise", bounds=(-0.5, 0.5))

        # Insertion Devices (IDs)
        self.manager.register_pv("ID:3A:GAP", default_val=18.5, sim_type="sinusoid", bounds=(15.0, 45.0))
        for id_sec in ["1C", "2A", "3C", "4A", "4C", "5A", "5C", "6A", "6C", "7A", "7C", "8A", "8C", "9A", "9C", "10A", "10C", "11C"]:
            if "GAP_R" in id_sec or "ActPos" in id_sec:
                pass
            pv_gap = f"SR:ID{id_sec}:UPSTREAM_GAP_RB" if "A" in id_sec or "C" in id_sec else f"SR:ID{id_sec}:GAP_R"
            if id_sec == "ID4A":
                pv_gap = "SR:ID4A:VAxis[0]_ActPos"
            elif id_sec == "1C":
                pv_gap = "SR:ID1C:GAP_R"
            elif id_sec in ["2A", "6A", "10A"]:
                self.manager.register_pv(f"SR:ID{id_sec}:IN_GAP", default_val=1, sim_type="binary_random")
                pv_gap = f"SR:ID{id_sec}:GAP_RB" if id_sec == "10A" else f"SR:ID{id_sec}:GAP_RB" # standard fallback
            
            self.manager.register_pv(pv_gap, default_val=22.4, sim_type="sinusoid", bounds=(16.0, 80.0))

        # Vacuum & Beam Gate Controllers (SR Cells 01-12)
        for i in range(1, 13):
            self.manager.register_pv(f"SR:MISC{i:02d}BG_ST", default_val=1, sim_type="binary_ok")
            self.manager.register_pv(f"SR:MISC{i:02d}CPST", default_val=1, sim_type="binary_ok")

        # Beamline Miscellaneous States (BL:MIS*)
        # Adding subset of key BLs from PV list:
        bls = ["10AB10TB1", "10CB10TB2", "10DB10TB2", "11AB11TB1", "11CB11TB2", "11DB11TB1", 
               "1BB01TB1", "1CB01TB2", "1DB01TB2", "2AB02TB1", "2CB02TB2", "2DB02TB2",
               "3AB03TB1", "3CB03TB2", "3DB03TB2", "4AB04TB1", "4BB04TB1", "4CB04TB2",
               "4DB04TB2", "5AB05TB1", "5CB05TB2", "5DB05TB2", "6AB06TB1", "6CB06TB2",
               "6DB06TB2", "7AB07TB1", "7BB07TB1", "7CB07TB2", "7DB07TB2", "8AB08TB1",
               "8CB08TB2", "8DB08TB2", "9AB09TB1", "9BB09TB1", "9CB09TB2", "9DB09TB2"]
        for b_id in bls:
            self.manager.register_pv(f"BL:MIS{b_id}SSS", default_val=1, sim_type="binary_ok")

    def init_ui(self):
        # Base Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ---------------------------------------------------------------------
        # HEADER BAR
        # ---------------------------------------------------------------------
        header_layout = QHBoxLayout()
        
        # Logo/Identity
        logo_layout = QVBoxLayout()
        facility_lbl = QLabel(FACILITY_NAME)
        facility_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ffcc;")
        title_lbl = QLabel("DIAGNOSTICS & SYSTEM STATUS BOARD")
        title_lbl.setStyleSheet("font-size: 11px; color: #888888; letter-spacing: 1px;")
        logo_layout.addWidget(facility_lbl)
        logo_layout.addWidget(title_lbl)
        header_layout.addLayout(logo_layout)
        
        header_layout.addStretch()

        # Engine selector (Simulated vs Live)
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(6)
        
        self.sim_btn = QPushButton("SIMULATION ON")
        self.sim_btn.setCheckable(True)
        self.sim_btn.setChecked(True)
        self.sim_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                border: 1px solid #ff6659;
                border-radius: 4px;
                color: white;
                font-weight: bold;
                padding: 6px 14px;
            }
            QPushButton:checked {
                background-color: #388e3c;
                border: 1px solid #6abf69;
            }
        """)
        self.sim_btn.toggled.connect(self.toggle_simulation_mode)
        mode_layout.addWidget(self.sim_btn)

        self.live_btn = QPushButton("EPICS PROD")
        self.live_btn.setEnabled(EPICS_AVAILABLE)
        self.live_btn.setStyleSheet("""
            QPushButton {
                background-color: #2c2c2c;
                border: 1px solid #444;
                border-radius: 4px;
                color: #888;
                font-weight: bold;
                padding: 6px 14px;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                border: 1px solid #222;
                color: #444;
            }
        """)
        mode_layout.addWidget(self.live_btn)
        
        header_layout.addLayout(mode_layout)

        # Clock
        self.clock_lbl = QLabel()
        self.clock_lbl.setStyleSheet("font-family: monospace; font-size: 16px; color: #00ffcc; padding-left: 15px;")
        header_layout.addWidget(self.clock_lbl)

        main_layout.addLayout(header_layout)

        # ---------------------------------------------------------------------
        # MAIN TAB ENGINE
        # ---------------------------------------------------------------------
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Build individual dashboard pages
        self.create_summary_tab()
        self.create_linac_tab()
        self.create_ring_tab()
        self.create_insertion_devices_tab()
        self.create_beamlines_tab()

        # Footer Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System initialized. Running in Simulator Mode.")

        # Tick clock at initialization
        self.update_clock()
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

    def toggle_simulation_mode(self, enabled):
        if enabled:
            self.sim_btn.setText("SIMULATION ON")
            self.manager.set_simulator_mode(True)
            self.status_bar.showMessage("Engine switched to simulation mode.")
        else:
            self.sim_btn.setText("SIMULATION OFF")
            self.manager.set_simulator_mode(False)
            self.status_bar.showMessage("Engine linked to Epics Production environment.")

    def update_clock(self):
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.clock_lbl.setText(curr_time)

    # ---------------------------------------------------------------------
    # TAB GENERATORS
    # ---------------------------------------------------------------------
    def create_summary_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left Panel: Core Machine Readouts
        left_panel = QVBoxLayout()
        
        beam_group = QGroupBox("Core Machine Parameters")
        grid = QGridLayout(beam_group)
        
        pvs = [
            ("Beam Current", "SR:G00:BEAMCURRENT_T", "mA", "{:.2f}"),
            ("Beam Lifetime", "SR:G00:LIFETIME_T", "hrs", "{:.2f}"),
            ("Storage Ring Energy", "SR:ENERGY", "GeV", "{:.1f}"),
            ("Top-Up Timer", "TOPUP:COUNT", "s", "{:.0f}"),
            ("Injection Rate", "INJ:RATE", "Hz", "{:.2f}"),
            ("Inj Efficiency BPM", "INJ:EFFI:BPM", "%", "{:.2f}")
        ]
        
        for idx, (lbl, name, unit, fmt) in enumerate(pvs):
            card = PVCard(lbl, name, unit, fmt)
            self.pv_widgets[name] = card
            grid.addWidget(card, idx // 2, idx % 2)

        left_panel.addWidget(beam_group)
        
        # Right Panel: Security & Overall status
        right_panel = QVBoxLayout()
        mps_group = QGroupBox("Safety & Interlock Overview")
        mps_layout = QVBoxLayout(mps_group)
        
        interlocks = [
            ("Linac MPS State", "LI:MISLINAC_MPSST"),
            ("BTL MPS Interlock State", "LI:MISBTL_MPSINT"),
            ("Pre-Injector State", "LI:MISPIJ_MPSST"),
            ("SR Fast Orbit Feedback", "SR:FOFB:HEART"),
            ("BPM System Status", "SR:MISIDBPMALLOK"),
            ("Global Gate Valves Status", "SR:MISGVALLST")
        ]
        
        for desc, pv in interlocks:
            row = StatusRow(desc, pv)
            self.pv_widgets[pv] = row
            mps_layout.addWidget(row)
            
        mps_layout.addStretch()
        right_panel.addWidget(mps_group)

        layout.addLayout(left_panel, 3)
        layout.addLayout(right_panel, 2)
        self.tab_widget.addTab(tab, "Main Overview")

    def create_linac_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Metrics Row
        metrics_layout = QHBoxLayout()
        pvs = [
            ("Linac Energy", "LI:ENERGY", "GeV", "{:.2f}"),
            ("Average Photo-Cathode Current", "LI:AVG_PC", "nC", "{:.2f}"),
            ("Energy Jitter", "LI:energy_Jitter", "%", "{:.3f}"),
            ("Linac Energy Spread", "LI:BTOTR1:Energy_Spread", "%", "{:.3f}")
        ]
        for title, pv, unit, fmt in pvs:
            card = PVCard(title, pv, unit, fmt)
            self.pv_widgets[pv] = card
            metrics_layout.addWidget(card)
        layout.addLayout(metrics_layout)

        # BPM Overview Area
        bpm_group = QGroupBox("Linac & Transfer Line BPM Array (mm)")
        bpm_layout = QGridLayout(bpm_group)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_grid = QGridLayout(scroll_content)
        scroll_grid.setSpacing(6)

        # Headers
        scroll_grid.addWidget(QLabel("BPM Index"), 0, 0)
        scroll_grid.addWidget(QLabel("X Position"), 0, 1)
        scroll_grid.addWidget(QLabel("Y Position"), 0, 2)
        scroll_grid.addWidget(QLabel("BPM Index"), 0, 4)
        scroll_grid.addWidget(QLabel("X Position"), 0, 5)
        scroll_grid.addWidget(QLabel("Y Position"), 0, 6)
        
        # Add Linac BPMs (L01-L10) and T-line BPMs (T01-T14)
        bpms = [f"L{i:02d}" for i in range(1, 11)] + [f"T{i:02d}" for i in range(1, 15)]
        
        for idx, bpm in enumerate(bpms):
            col_offset = 4 if idx >= 12 else 0
            row_idx = (idx % 12) + 1
            
            lbl = QLabel(f"BPM {bpm}")
            lbl.setStyleSheet("font-weight: bold; color: #888888;")
            scroll_grid.addWidget(lbl, row_idx, col_offset + 0)
            
            pv_x = f"LI:BPM_{bpm}:X"
            pv_y = f"LI:BPM_{bpm}:Y"
            
            card_x = PVCard("", pv_x, "mm", "{:.4f}")
            card_y = PVCard("", pv_y, "mm", "{:.4f}")
            
            self.pv_widgets[pv_x] = card_x
            self.pv_widgets[pv_y] = card_y
            
            scroll_grid.addWidget(card_x, row_idx, col_offset + 1)
            scroll_grid.addWidget(card_y, row_idx, col_offset + 2)
            
            # Spacer spacer
            if col_offset == 0:
                spacer = QFrame()
                spacer.setFrameShape(QFrame.Shape.VLine)
                spacer.setStyleSheet("color: #333333;")
                scroll_grid.addWidget(spacer, row_idx, 3)

        scroll.setWidget(scroll_content)
        bpm_layout.addWidget(scroll)
        layout.addWidget(bpm_group)
        self.tab_widget.addTab(tab, "Linac & Transfer Line")

    def create_ring_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Physics Diagnostic Area
        left_layout = QVBoxLayout()
        physics_group = QGroupBox("Beam Parameters & Size Profiles")
        physics_grid = QGridLayout(physics_group)
        
        pvs = [
            ("Horizontal Tune (Qx)", "SR:G00:TUNE_X", "", "{:.4f}"),
            ("Vertical Tune (Qy)", "SR:G00:TUNE_Y", "", "{:.4f}"),
            ("H-Beam Size (σx)", "SR:G00:BEAMSIZE_X", "μm", "{:.2f}"),
            ("V-Beam Size (σy)", "SR:G00:BEAMSIZE_Y", "μm", "{:.2f}")
        ]
        for idx, (title, pv, unit, fmt) in enumerate(pvs):
            card = PVCard(title, pv, unit, fmt)
            self.pv_widgets[pv] = card
            physics_grid.addWidget(card, idx // 2, idx % 2)
            
        left_layout.addWidget(physics_group)
        
        # Interlocks status block
        status_gp = QGroupBox("Global Interlock Profiles")
        status_lay = QVBoxLayout(status_gp)
        status_lay.addWidget(StatusRow("SR Interlock Monitor", "SR:BPM12-3:ENV:ENV_INTERLOCK_MONITOR"))
        status_lay.addWidget(StatusRow("Machine Beam Gate Master Status", "SR:MISGVALLST"))
        status_lay.addStretch()
        left_layout.addWidget(status_gp)

        # Cell Status Block
        right_layout = QVBoxLayout()
        cell_group = QGroupBox("Cell Vacuum & Gate Status")
        cell_grid = QGridLayout(cell_group)
        
        cell_grid.addWidget(QLabel("Cell"), 0, 0)
        cell_grid.addWidget(QLabel("Gate State (BG)"), 0, 1)
        cell_grid.addWidget(QLabel("Cryo Pump (CP)"), 0, 2)
        cell_grid.addWidget(QLabel("Cell"), 0, 4)
        cell_grid.addWidget(QLabel("Gate State (BG)"), 0, 5)
        cell_grid.addWidget(QLabel("Cryo Pump (CP)"), 0, 6)
        
        # Divider line
        for r in range(1, 7):
            divider = QFrame()
            divider.setFrameShape(QFrame.Shape.VLine)
            divider.setStyleSheet("color: #333333;")
            cell_grid.addWidget(divider, r, 3)

        for i in range(1, 13):
            # Split to 2 columns of cells
            col_offset = 4 if i > 6 else 0
            row_idx = (i - 1) % 6 + 1
            
            lbl = QLabel(f"Cell {i:02d}")
            lbl.setStyleSheet("font-weight: bold; color: #888888;")
            cell_grid.addWidget(lbl, row_idx, col_offset + 0)
            
            pv_bg = f"SR:MISC{i:02d}BG_ST"
            pv_cp = f"SR:MISC{i:02d}CPST"
            
            row_bg = StatusRow("", pv_bg)
            row_cp = StatusRow("", pv_cp)
            
            self.pv_widgets[pv_bg] = row_bg
            self.pv_widgets[pv_cp] = row_cp
            
            cell_grid.addWidget(row_bg, row_idx, col_offset + 1)
            cell_grid.addWidget(row_cp, row_idx, col_offset + 2)

        right_layout.addWidget(cell_group)
        
        layout.addLayout(left_layout, 2)
        layout.addLayout(right_layout, 3)
        self.tab_widget.addTab(tab, "Storage Ring Controls")

    def create_insertion_devices_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        
        # Section Header labels
        grid.addWidget(QLabel("ID Sector"), 0, 0)
        grid.addWidget(QLabel("Gap Position"), 0, 1)
        grid.addWidget(QLabel("ID Status"), 0, 2)
        grid.addWidget(QLabel("ID Sector"), 0, 4)
        grid.addWidget(QLabel("Gap Position"), 0, 5)
        grid.addWidget(QLabel("ID Status"), 0, 6)

        # Unique lists extracted from user configuration files
        id_sectors = ["1C", "2A", "3C", "4A", "4C", "5A", "5C", "6A", "6C", "7A", "7C", "8A", "8C", "9A", "9C", "10A", "10C", "11C"]
        
        for idx, sec in enumerate(id_sectors):
            col_offset = 4 if idx >= 9 else 0
            row_idx = (idx % 9) + 1
            
            lbl = QLabel(f"Insertion Device {sec}")
            lbl.setStyleSheet("font-weight: bold; color: #888888;")
            grid.addWidget(lbl, row_idx, col_offset + 0)
            
            # Reconstruct PV Gap based on formatting
            if sec == "4A":
                pv_gap = "SR:ID4A:VAxis[0]_ActPos"
            elif sec == "1C":
                pv_gap = "SR:ID1C:GAP_R"
            elif "A" in sec or "C" in sec:
                pv_gap = f"SR:ID{sec}:UPSTREAM_GAP_RB"
            else:
                pv_gap = f"SR:ID{sec}:GAP_R"
                
            card_gap = PVCard("", pv_gap, "mm", "{:.4f}")
            self.pv_widgets[pv_gap] = card_gap
            grid.addWidget(card_gap, row_idx, col_offset + 1)

            # In Gap status tracking for specialized elements
            if sec in ["2A", "6A", "10A"]:
                pv_state = f"SR:ID{sec}:IN_GAP"
                state_led = StatusRow("", pv_state)
                self.pv_widgets[pv_state] = state_led
                grid.addWidget(state_led, row_idx, col_offset + 2)
            else:
                placeholder = QLabel("ACTIVE")
                placeholder.setStyleSheet("color: #00ff66; font-size: 11px;")
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(placeholder, row_idx, col_offset + 2)

            # Center divider line
            if col_offset == 0:
                v_line = QFrame()
                v_line.setFrameShape(QFrame.Shape.VLine)
                v_line.setStyleSheet("color: #333333;")
                grid.addWidget(v_line, row_idx, 3)

        scroll.setWidget(container)
        layout.addWidget(scroll)
        self.tab_widget.addTab(tab, "Insertion Devices (IDs)")

    def create_beamlines_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        
        bls = [
            "10AB10TB1", "10CB10TB2", "10DB10TB2", "11AB11TB1", "11CB11TB2", "11DB11TB1", 
            "1BB01TB1", "1CB01TB2", "1DB01TB2", "2AB02TB1", "2CB02TB2", "2DB02TB2",
            "3AB03TB1", "3CB03TB2", "3DB03TB2", "4AB04TB1", "4BB04TB1", "4CB04TB2",
            "4DB04TB2", "5AB05TB1", "5CB05TB2", "5DB05TB2", "6AB06TB1", "6CB06TB2",
            "6DB06TB2", "7AB07TB1", "7BB07TB1", "7CB07TB2", "7DB07TB2", "8AB08TB1",
            "8CB08TB2", "8DB08TB2", "9AB09TB1", "9BB09TB1", "9CB09TB2", "9DB09TB2"
        ]
        
        # Grid arrangement: 3 columns of details
        col_width = 12
        for idx, bl in enumerate(bls):
            col_offset = (idx // col_width) * 3
            row_idx = idx % col_width
            
            lbl = QLabel(f"BL-{bl[:3]}")
            lbl.setStyleSheet("font-weight: bold; color: #888888;")
            grid.addWidget(lbl, row_idx, col_offset + 0)
            
            pv_name = f"BL:MIS{bl}SSS"
            status_wdg = StatusRow("Valve/Interlock", pv_name)
            self.pv_widgets[pv_name] = status_wdg
            grid.addWidget(status_wdg, row_idx, col_offset + 1)
            
            # Vertical separator lines
            if col_offset < 6:
                v_line = QFrame()
                v_line.setFrameShape(QFrame.Shape.VLine)
                v_line.setStyleSheet("color: #333333;")
                grid.addWidget(v_line, row_idx, col_offset + 2)

        scroll.setWidget(container)
        layout.addWidget(scroll)
        self.tab_widget.addTab(tab, "Beamline Safety Interfaces (BL:MIS)")

    # ---------------------------------------------------------------------
    # DATA AND GUI SYNC
    # ---------------------------------------------------------------------
    def update_gui(self):
        """Pulls updated values from the manager and pushes updates down to the UI widgets"""
        for pv, widget in self.pv_widgets.items():
            val = self.manager.get_value(pv)
            if isinstance(widget, PVCard):
                widget.update_value(val)
            elif isinstance(widget, StatusRow):
                widget.update_state(val)

# -----------------------------------------------------------------------------
# APPLICATION ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Force dark fusion style colors to eliminate any standard OS system override
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 26))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(18, 18, 18))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(26, 26, 26))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(42, 42, 42))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 255, 204))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)

    dashboard = MonitorDashboard()
    dashboard.show()
    sys.exit(app.exec())