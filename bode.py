"""
Requirements:
    - numpy
    - PyQt6
    - matplotlib
    - seaborn
    - mplcyberpunk

Install with:
    pip install numpy PyQt6 matplotlib seaborn mplcyberpunk

Note: mplcyberpunk is required for the 'cyberpunk' matplotlib style.
"""

import sys
import numpy as np
import seaborn as sns
import mplcyberpunk
import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QSizePolicy, 
    QSplitter,
    QPushButton,
    QGroupBox,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QSlider,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Set cyberpunk style globally
plt.style.use("cyberpunk")


# ---------------------------
# Utility: polynomial -> string
# ---------------------------

def poly_to_str(coeffs, var="s", precision=3):
    """
    Convert polynomial coefficients into a human-readable string.

    coeffs: array-like, [a_n, ..., a_0]
    returns: "a_n s^n + ... + a_0"
    """
    coeffs = np.array(coeffs, dtype=float)
    # strip leading ~zeros
    eps = 1e-10
    nz = np.where(np.abs(coeffs) > eps)[0]
    if len(nz) == 0:
        return "0"
    coeffs = coeffs[nz[0]:]
    n = len(coeffs) - 1

    terms = []
    for i, a in enumerate(coeffs):
        power = n - i
        if abs(a) < eps:
            continue
        # sign and magnitude
        sign = "-" if a < 0 else "+"
        mag = abs(a)

        if power == 0:
            term_core = f"{mag:.{precision}g}"
        elif power == 1:
            if np.isclose(mag, 1.0, atol=10**(-precision)):
                term_core = var
            else:
                term_core = f"{mag:.{precision}g}{var}"
        else:
            if np.isclose(mag, 1.0, atol=10**(-precision)):
                term_core = f"{var}^{power}"
            else:
                term_core = f"{mag:.{precision}g}{var}^{power}"

        terms.append((sign, term_core))

    # build final string
    if not terms:
        return "0"

    first_sign, first_core = terms[0]
    if first_sign == "-":
        s = "-" + first_core
    else:
        s = first_core

    for sign, core in terms[1:]:
        s += f" {sign} {core}"

    return s


def poly_to_latex(coeffs, var="s", precision=3):
    """
    Convert polynomial coefficients into LaTeX format.

    coeffs: array-like, [a_n, ..., a_0]
    returns: LaTeX string for the polynomial
    """
    coeffs = np.array(coeffs, dtype=float)
    # strip leading ~zeros
    eps = 1e-10
    nz = np.where(np.abs(coeffs) > eps)[0]
    if len(nz) == 0:
        return "0"
    coeffs = coeffs[nz[0]:]
    n = len(coeffs) - 1

    terms = []
    for i, a in enumerate(coeffs):
        power = n - i
        if abs(a) < eps:
            continue
        # sign and magnitude
        sign = "-" if a < 0 else "+"
        mag = abs(a)

        if power == 0:
            term_core = f"{mag:.{precision}g}"
        elif power == 1:
            if np.isclose(mag, 1.0, atol=10**(-precision)):
                term_core = var
            else:
                term_core = f"{mag:.{precision}g}{var}"
        else:
            if np.isclose(mag, 1.0, atol=10**(-precision)):
                term_core = f"{var}^{{{power}}}"
            else:
                term_core = f"{mag:.{precision}g}{var}^{{{power}}}"

        terms.append((sign, term_core))

    # build final string
    if not terms:
        return "0"

    first_sign, first_core = terms[0]
    if first_sign == "-":
        s = "-" + first_core
    else:
        s = first_core

    for sign, core in terms[1:]:
        s += f" {sign} {core}"

    return s


# ---------------------------
# Root control widget
# ---------------------------

class RootControl(QWidget):
    """
    Widget to control one root (zero or pole).
    Represents either:
      - a real root (if Im == 0)
      - a complex-conjugate pair (if Im != 0) in the math model

    Fields:
      - Enabled checkbox
      - Real slider + label
      - Imag slider + label
      - Multiplicity spinbox
      - Remove button
    """
    changed = pyqtSignal()           # emitted whenever any parameter changes
    removed = pyqtSignal(QWidget)    # emitted when user clicks Remove

    def __init__(self, kind="zero", parent=None):
        super().__init__(parent)
        assert kind in ("zero", "pole")
        self.kind = kind

        layout = QGridLayout()
        layout.setSpacing(2)                # was default ~8–12
        layout.setContentsMargins(4, 4, 4, 4)

        row = 0

        # Title row
        self.enabled_cb = QCheckBox(f"{kind.capitalize()}")
        self.enabled_cb.setChecked(True)
        layout.addWidget(self.enabled_cb, row, 0, 1, 2)

        self.remove_btn = QPushButton("Remove")
        layout.addWidget(self.remove_btn, row, 2, 1, 1)
        row += 1

        # Real part
        layout.addWidget(QLabel("Re:"), row, 0)
        self.re_slider = self._make_slider()
        self.re_value_label = QLabel("0.0")
        layout.addWidget(self.re_slider, row, 1)
        layout.addWidget(self.re_value_label, row, 2)
        row += 1

        # Imag part
        layout.addWidget(QLabel("Im:"), row, 0)
        self.im_slider = self._make_im_slider()
        self.im_value_label = QLabel("0.0")
        layout.addWidget(self.im_slider, row, 1)
        layout.addWidget(self.im_value_label, row, 2)
        row += 1

        # Multiplicity
        layout.addWidget(QLabel("Mult:"), row, 0)
        self.mult_spin = QSpinBox()
        self.mult_spin.setRange(1, 10)
        self.mult_spin.setValue(1)
        layout.addWidget(self.mult_spin, row, 1)

        # stretch
        layout.setColumnStretch(1, 1)
        self.setLayout(layout)

        # Set white text styling for this widget and its children
        self.setStyleSheet("""
            QLabel { color: white; font-size: 11pt; }
            QCheckBox { color: white; font-size: 11pt; }
            QCheckBox::indicator { 
                border: 2px solid white; 
                background-color: #1a1a2e; 
                width: 18px; 
                height: 18px; 
                border-radius: 3px;
            }
            QCheckBox::indicator:checked { 
                background-color: #87BBA2; 
                border: 2px solid #87BBA2;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #A0D4B8;
                border: 2px solid #A0D4B8;
            }
            QPushButton { color: white; background-color: #2a3a5a; font-size: 11pt; }
            QPushButton:hover { background-color: #3a4a6a; }
            QSpinBox { color: white; background-color: #1a1a2e; font-size: 11pt; }
            QSlider::groove:horizontal { background: #1a1a2e; }
            QSlider::handle:horizontal { background: white; }
        """)

        # connections
        self.enabled_cb.stateChanged.connect(self._emit_changed)
        self.re_slider.valueChanged.connect(self._on_re_slider)
        self.im_slider.valueChanged.connect(self._on_im_slider)
        self.mult_spin.valueChanged.connect(self._emit_changed)
        self.remove_btn.clicked.connect(self._on_remove)

        # initialize values
        self.re_slider.setValue(0)
        self.im_slider.setValue(0)

    def _make_slider(self):
        # map [-100, 100] -> [-10.0, 10.0] with step 0.1
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(-100, 100)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setValue(0)
        return slider

    def _make_im_slider(self):
        # map [0, 100] -> [0.0, 10.0] with step 0.1 (imaginary values are non-negative)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setValue(0)
        return slider

    @staticmethod
    def slider_to_float(v):
        return v / 10.0

    @staticmethod
    def im_slider_to_float(v):
        # map [0, 100] -> [0.0, 10.0]
        return v / 10.0

    @staticmethod
    def float_to_slider(x):
        x = max(min(x, 10.0), -10.0)
        return int(round(x * 10))

    @staticmethod
    def im_float_to_slider(x):
        # map [0.0, 10.0] -> [0, 100], clamp to non-negative
        x = max(min(abs(x), 10.0), 0.0)
        return int(round(x * 10))

    def _on_re_slider(self, v):
        f = self.slider_to_float(v)
        self.re_value_label.setText(f"{f:.1f}")
        self._emit_changed()

    def _on_im_slider(self, v):
        f = self.im_slider_to_float(v)
        self.im_value_label.setText(f"{f:.1f}")
        self._emit_changed()

    def _on_remove(self):
        self.removed.emit(self)

    def _emit_changed(self, *args):
        self.changed.emit()

    # public getters
    def is_enabled(self):
        return self.enabled_cb.isChecked()

    def get_root(self):
        """
        Returns (real, imag, multiplicity).
        imag != 0 will correspond to a conjugate pair in the model.
        imag is always non-negative (0 to 10.0).
        """
        re = self.slider_to_float(self.re_slider.value())
        im = self.im_slider_to_float(self.im_slider.value())
        mult = self.mult_spin.value()
        return re, im, mult


# ---------------------------
# Main window
# ---------------------------

class BodeWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interactive Bode Plot Visualization")

        # Data
        self.zeros_controls = []
        self.poles_controls = []

        # Central widget with layout
        central = QWidget()
        # Set overall window background to dark-ish color
        central.setStyleSheet("background-color: #14182A;")
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setCentralWidget(central)

        # Left: control panel
        control_panel = self._create_control_panel()
        control_panel.setMinimumWidth(400)  # Make control panel wider
        main_layout.addWidget(control_panel, stretch=0)

        # Right: matplotlib figure
        self.canvas = self._create_figure()
        main_layout.addWidget(self.canvas, stretch=1)

        # Initial update
        self.update_bode()

    # ---------- control panel ----------

    def _create_control_panel(self):
        panel = QWidget()
        panel.setStyleSheet("""
            QWidget { background-color: #212946; color: white; }
            QLabel { color: white; font-size: 11pt; }
            QGroupBox {
                color: white; border: 1px solid #555; border-radius: 5px;
                margin-top: 10px; padding-top: 10px; font-size: 12pt;
            }
            QGroupBox::title {
                color: white; subcontrol-origin: margin; left: 10px;
                padding: 0 5px; font-size: 12pt;
            }
            QPushButton {
                color: white; background-color: #2a3a5a; border: 1px solid #555;
                padding: 5px; font-size: 11pt;
            }
            QPushButton:hover { background-color: #3a4a6a; }
            QPushButton:pressed { background-color: #1a2a4a; }
            QCheckBox { color: white; font-size: 11pt; }
            QCheckBox::indicator {
                border: 2px solid white; background-color: #1a1a2e;
                width: 18px; height: 18px; border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #87BBA2; border: 2px solid #87BBA2;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #A0D4B8; border: 2px solid #A0D4B8;
            }
            QSpinBox, QDoubleSpinBox {
                color: white; background-color: #1a1a2e; border: 1px solid #555;
                font-size: 11pt;
            }
            QSlider::groove:horizontal {
                background: #1a1a2e; height: 8px; border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: white; width: 18px; margin: -5px 0; border-radius: 9px;
            }
            QScrollArea { color: white; background-color: #212946; border: 1px solid #555; }
        """)

        layout = QVBoxLayout(panel)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # -------------------------
        # Global settings group
        # -------------------------
        gf_group = QGroupBox("Global Settings")
        gf_layout = QGridLayout(gf_group)
        gf_layout.setHorizontalSpacing(8)
        gf_layout.setVerticalSpacing(6)

        # Gain
        gf_layout.addWidget(QLabel("Gain K:"), 0, 0)
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(1e-3, 1e3)
        self.gain_spin.setDecimals(3)
        self.gain_spin.setValue(1.0)
        self.gain_spin.setSingleStep(0.1)
        gf_layout.addWidget(self.gain_spin, 0, 1)

        # Frequency range (log10)
        gf_layout.addWidget(QLabel("log10(ω_min):"), 1, 0)
        self.wmin_spin = QDoubleSpinBox()
        self.wmin_spin.setRange(-4.0, 4.0)
        self.wmin_spin.setDecimals(2)
        self.wmin_spin.setValue(-2.0)
        gf_layout.addWidget(self.wmin_spin, 1, 1)

        gf_layout.addWidget(QLabel("log10(ω_max):"), 2, 0)
        self.wmax_spin = QDoubleSpinBox()
        self.wmax_spin.setRange(-4.0, 4.0)
        self.wmax_spin.setDecimals(2)
        self.wmax_spin.setValue(2.0)
        gf_layout.addWidget(self.wmax_spin, 2, 1)

        layout.addWidget(gf_group)

        # -------------------------
        # Zeros group
        # -------------------------
        zeros_group = QGroupBox("Zeros")
        zeros_layout = QVBoxLayout(zeros_group)
        zeros_layout.setContentsMargins(6, 6, 6, 6)
        zeros_layout.setSpacing(6)

        self.zeros_container = QWidget()
        self.zeros_container_layout = QVBoxLayout(self.zeros_container)
        self.zeros_container_layout.setSpacing(4)
        self.zeros_container_layout.setContentsMargins(2, 2, 2, 2)

        self.zeros_scroll = QScrollArea()
        self.zeros_scroll.setWidgetResizable(True)
        self.zeros_scroll.setWidget(self.zeros_container)
        self.zeros_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.zeros_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.zeros_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        zeros_layout.addWidget(self.zeros_scroll, stretch=1)

        self.add_zero_btn = QPushButton("Add Zero")
        zeros_layout.addWidget(self.add_zero_btn)
        zeros_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # -------------------------
        # Poles group
        # -------------------------
        poles_group = QGroupBox("Poles")
        poles_layout = QVBoxLayout(poles_group)
        poles_layout.setContentsMargins(6, 6, 6, 6)
        poles_layout.setSpacing(6)

        self.poles_container = QWidget()
        self.poles_container_layout = QVBoxLayout(self.poles_container)
        self.poles_container_layout.setSpacing(4)
        self.poles_container_layout.setContentsMargins(2, 2, 2, 2)

        self.poles_scroll = QScrollArea()
        self.poles_scroll.setWidgetResizable(True)
        self.poles_scroll.setWidget(self.poles_container)
        self.poles_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.poles_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.poles_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        poles_layout.addWidget(self.poles_scroll, stretch=1)

        self.add_pole_btn = QPushButton("Add Pole")
        poles_layout.addWidget(self.add_pole_btn)
        poles_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # -------------------------
        # Adaptive Z/P splitter
        # -------------------------
        zp_splitter = QSplitter(Qt.Orientation.Vertical)
        zp_splitter.setChildrenCollapsible(False)
        zp_splitter.addWidget(zeros_group)
        zp_splitter.addWidget(poles_group)
        zp_splitter.setStretchFactor(0, 1)
        zp_splitter.setStretchFactor(1, 1)
        zp_splitter.setSizes([300, 300])  # initial bias; user can drag divider if they want

        layout.addWidget(zp_splitter, stretch=1)

        # -------------------------
        # Reset button
        # -------------------------
        self.reset_btn = QPushButton("Reset (clear zeros & poles)")
        layout.addWidget(self.reset_btn)

        layout.addStretch(0)

        # -------------------------
        # Connections
        # -------------------------
        self.add_zero_btn.clicked.connect(self.add_zero_control)
        self.add_pole_btn.clicked.connect(self.add_pole_control)
        self.reset_btn.clicked.connect(self.reset_model)
        self.gain_spin.valueChanged.connect(self.update_bode)
        self.wmin_spin.valueChanged.connect(self.update_bode)
        self.wmax_spin.valueChanged.connect(self.update_bode)

        # -------------------------
        # Default zeros & poles
        # -------------------------
        self.add_zero_control()

        pole1 = self.add_pole_control()
        pole1.re_slider.setValue(RootControl.float_to_slider(-1.0))
        pole1.im_slider.setValue(RootControl.im_float_to_slider(2.0))
        pole1.re_value_label.setText("-1.0")
        pole1.im_value_label.setText("2.0")

        self.add_pole_control()

        return panel


    def add_zero_control(self):
        ctrl = RootControl(kind="zero")
        self.zeros_controls.append(ctrl)
        self.zeros_container_layout.addWidget(ctrl)
        ctrl.changed.connect(self.update_bode)
        ctrl.removed.connect(self.remove_zero_control)
        self.update_bode()
        return ctrl

    def add_pole_control(self):
        ctrl = RootControl(kind="pole")
        self.poles_controls.append(ctrl)
        self.poles_container_layout.addWidget(ctrl)
        ctrl.changed.connect(self.update_bode)
        ctrl.removed.connect(self.remove_pole_control)
        self.update_bode()
        return ctrl

    def remove_zero_control(self, ctrl_widget):
        if ctrl_widget in self.zeros_controls:
            self.zeros_controls.remove(ctrl_widget)
            ctrl_widget.setParent(None)
            ctrl_widget.deleteLater()
            self.update_bode()

    def remove_pole_control(self, ctrl_widget):
        if ctrl_widget in self.poles_controls:
            self.poles_controls.remove(ctrl_widget)
            ctrl_widget.setParent(None)
            ctrl_widget.deleteLater()
            self.update_bode()

    def reset_model(self):
        for ctrl in self.zeros_controls:
            ctrl.setParent(None)
            ctrl.deleteLater()
        for ctrl in self.poles_controls:
            ctrl.setParent(None)
            ctrl.deleteLater()
        self.zeros_controls.clear()
        self.poles_controls.clear()
        
        # Add default zeros and poles
        # Zero: 0+0i
        self.add_zero_control()
        
        # Pole 1: -1-2i (imaginary part is 2.0, positive)
        pole1 = self.add_pole_control()
        pole1.re_slider.setValue(RootControl.float_to_slider(-1.0))
        pole1.im_slider.setValue(RootControl.im_float_to_slider(2.0))
        pole1.re_value_label.setText("-1.0")
        pole1.im_value_label.setText("2.0")
        
        # Pole 2: 0+0i (already default)
        self.add_pole_control()
        
        self.gain_spin.setValue(1.0)
        self.wmin_spin.setValue(-2.0)
        self.wmax_spin.setValue(2.0)
        self.update_bode()

    # ---------- matplotlib figure ----------

    def _create_figure(self):
        self.fig = Figure(figsize=(6, 6))
        # Apply cyberpunk style to the figure
        with plt.style.context("cyberpunk"):
            self.ax_mag, self.ax_phase = self.fig.add_subplot(2, 1, 1), self.fig.add_subplot(2, 1, 2)


        self.ax_mag.set_title("Bode Magnitude", fontsize=13)
        self.ax_mag.set_ylabel("Magnitude (dB)", fontsize=12)
        self.ax_mag.set_xscale("log")
        self.ax_mag.tick_params(labelsize=11)
        self.ax_mag.grid(True)

        self.ax_phase.set_title("Bode Phase", fontsize=13)
        self.ax_phase.set_ylabel("Phase (deg)", fontsize=12)
        self.ax_phase.set_xlabel("Frequency ω (rad/s)", fontsize=12)
        self.ax_phase.set_xscale("log")
        self.ax_phase.tick_params(labelsize=11)
        self.ax_phase.grid(True)

        # initial dummy data
        w = np.logspace(-2, 2, 100)
        mag = np.zeros_like(w)
        phase = np.zeros_like(w)

        # Get flare color palette for plotting
        self.palette = sns.color_palette("flare", 10)
        
        # Use custom color for magnitude and phase plots
        plot_color = "#C9E4CA"
        (self.mag_line,) = self.ax_mag.semilogx(w, mag, color=plot_color, label='Magnitude')
        (self.phase_line,) = self.ax_phase.semilogx(w, phase, color=plot_color, label='Phase')

        # text for TF - using LaTeX rendering, more prominent
        # Place it at the very top of the figure, above the magnitude plot title
        # Use cyberpunk-themed colors
        # Left box for transfer function
        tfspace = .13
        self.tf_text = self.fig.text(
            tfspace,
            0.98,
            "",
            transform=self.fig.transFigure, 
            ha="center",
            va="top",
            fontsize=16,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#212946", alpha=0.9, edgecolor="#87BBA2", linewidth=1.5),
        )
        
        # Four separate boxes for system parameters on the right side
        # Position them in a single row
        space = .13
        self.tau_text = self.fig.text(
            tfspace + 1.5 * space,
            0.97,
            "",
            transform=self.fig.transFigure, 
            ha="center",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#212946", alpha=0.9, edgecolor="#87BBA2", linewidth=1.5),
        )
        
        self.omega_n_text = self.fig.text(
            tfspace + 2.5 * space,
            0.97,
            "",
            transform=self.fig.transFigure, 
            ha="center",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#212946", alpha=0.9, edgecolor="#87BBA2", linewidth=1.5),
        )
        
        self.omega_d_text = self.fig.text(
            tfspace + 3.5 * space,
            0.97,
            "",
            transform=self.fig.transFigure, 
            ha="center",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#212946", alpha=0.9, edgecolor="#87BBA2", linewidth=1.5),
        )
        
        self.zeta_text = self.fig.text(
            tfspace + 4.5 * space,
            0.97,
            "",
            transform=self.fig.transFigure, 
            ha="center",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#212946", alpha=0.9, edgecolor="#87BBA2", linewidth=1.5),
        )
        
        canvas = FigureCanvas(self.fig)
        return canvas

    # ---------- model & computation ----------

    def _collect_roots(self):
        """
        From the zero/pole controls, build lists of complex roots.
        Imaginary != 0 will create a conjugate pair.
        Returns:
          zeros: list of complex
          poles: list of complex
        """
        zeros = []
        poles = []

        eps_im = 1e-8

        # zeros
        for ctrl in self.zeros_controls:
            if not ctrl.is_enabled():
                continue
            re, im, mult = ctrl.get_root()
            if abs(im) < eps_im:
                # real root, multiplicity many
                zeros.extend([re] * mult)
            else:
                # complex conjugate pair
                z1 = re + 1j * im
                z2 = re - 1j * im
                for _ in range(mult):
                    zeros.append(z1)
                    zeros.append(z2)

        # poles
        for ctrl in self.poles_controls:
            if not ctrl.is_enabled():
                continue
            re, im, mult = ctrl.get_root()
            if abs(im) < eps_im:
                poles.extend([re] * mult)
            else:
                p1 = re + 1j * im
                p2 = re - 1j * im
                for _ in range(mult):
                    poles.append(p1)
                    poles.append(p2)

        return zeros, poles

    def _build_tf_polynomials(self):
        """
        Build numerator and denominator polynomials from zeros and poles.
        Returns:
          num, den (numpy arrays)
        """
        zeros, poles = self._collect_roots()

        if len(zeros) > 0:
            num = np.poly(zeros)
        else:
            num = np.array([1.0])

        if len(poles) > 0:
            den = np.poly(poles)
        else:
            den = np.array([1.0])

        num = np.real_if_close(num, tol=1e-8)
        den = np.real_if_close(den, tol=1e-8)

        return num, den

    def _extract_system_parameters(self):
        """
        
        Extract time constants, natural frequency, damped frequency, and damping ratio from poles.
        Returns:
          time_constants: list of time constants (or None if not applicable)
          natural_freq: natural frequency ωₙ (or None if not applicable)
          damped_freq: damped frequency ωd (or None if not applicable)
          zeta: damping ratio ζ (or None if not applicable)
        """
        zeros, poles = self._collect_roots()
        
        if len(poles) == 0:
            return None, None, None, None
        
        time_constants = []
        natural_freq = None
        damped_freq = None
        zeta = None
        
        eps_im = 1e-8
        
        # Separate real and complex poles
        real_poles = []
        complex_poles = []
        
        for p in poles:
            if abs(p.imag) < eps_im:
                real_poles.append(p.real)
            else:
                complex_poles.append(p)
        
        # Extract time constants from real poles
        # For a real pole at -1/τ, the time constant is τ = -1/pole (if pole < 0)
        for p_real in real_poles:
            if p_real < -eps_im:  # Only negative real poles give positive time constants
                tau = -1.0 / p_real
                if tau > 1e-10:  # Avoid very small or negative time constants
                    time_constants.append(tau)
        
        # Extract natural frequency, damped frequency, and damping ratio from complex conjugate pairs
        # Look for a dominant second-order pair
        if len(complex_poles) >= 2:
            # Find complex conjugate pairs
            used_indices = set()
            for i, p1 in enumerate(complex_poles):
                if i in used_indices:
                    continue
                for j, p2 in enumerate(complex_poles[i+1:], start=i+1):
                    if j in used_indices:
                        continue
                    # Check if they're conjugates
                    if abs(p1.real - p2.real) < eps_im and abs(p1.imag + p2.imag) < eps_im:
                        # Found a conjugate pair
                        sigma = -p1.real  # damping coefficient (σ, can be positive or negative)
                        omega_d = abs(p1.imag)  # damped frequency
                        
                        # Natural frequency: ωₙ = √(σ² + ωd²)
                        omega_n = np.sqrt(sigma**2 + omega_d**2)
                        
                        # Damping ratio: ζ = σ / ωₙ
                        if omega_n > eps_im:  # Avoid division by zero
                            zeta_val = sigma / omega_n
                        else:
                            zeta_val = None
                        
                        # Only set if we haven't found one yet, or if this is the dominant pair (lowest ωₙ)
                        if natural_freq is None or omega_n < natural_freq:
                            natural_freq = omega_n
                            damped_freq = omega_d
                            zeta = zeta_val
                        
                        # Mark both as used
                        used_indices.add(i)
                        used_indices.add(j)
                        break
        
        # If no time constants found, return None
        if len(time_constants) == 0:
            time_constants = None
        else:
            # Sort and remove duplicates (within tolerance)
            time_constants = sorted(set([round(t, 6) for t in time_constants]))
        
        return time_constants, natural_freq, damped_freq, zeta

    def update_bode(self):
        """
        Recompute and redraw the Bode plot.
        Called whenever any control changes.
        """
        # Check if figure has been initialized
        if not hasattr(self, 'mag_line') or not hasattr(self, 'phase_line') or not hasattr(self, 'tf_text') or not hasattr(self, 'tau_text'):
            return

        # frequency range
        wmin_log = float(self.wmin_spin.value())
        wmax_log = float(self.wmax_spin.value())
        if wmin_log >= wmax_log:
            # avoid invalid range
            wmax_log = wmin_log + 0.1
            self.wmax_spin.blockSignals(True)
            self.wmax_spin.setValue(wmax_log)
            self.wmax_spin.blockSignals(False)

        w = np.logspace(wmin_log, wmax_log, 500)

        # transfer function
        num, den = self._build_tf_polynomials()
        K = float(self.gain_spin.value())

        s = 1j * w
        num_vals = np.polyval(num, s)
        den_vals = np.polyval(den, s)

        # avoid division by zero
        eps = 1e-16
        den_vals = np.where(np.abs(den_vals) < eps, eps, den_vals)

        G = K * num_vals / den_vals

        mag = 20 * np.log10(np.maximum(np.abs(G), 1e-12))
        phase = np.angle(G, deg=True)

        # update lines
        self.mag_line.set_data(w, mag)
        self.phase_line.set_data(w, phase)
        
        # Set custom color for magnitude and phase plots
        plot_color = "#C9E4CA"
        self.mag_line.set_color(plot_color)
        self.phase_line.set_color(plot_color)

        # rescale
        self.ax_mag.relim()
        self.ax_mag.autoscale_view()

        self.ax_phase.relim()
        self.ax_phase.autoscale_view()

        # update TF text with LaTeX rendering
        num_latex = poly_to_latex(num, var="s")
        den_latex = poly_to_latex(den, var="s")
        # Format gain nicely
        if abs(K - 1.0) < 1e-6:
            gain_str = ""
        else:
            gain_str = f"{K:.3g}"
        
        # Build LaTeX string for transfer function
        if gain_str:
            tf_latex = f"$G(s) = {gain_str} \\cdot \\frac{{{num_latex}}}{{{den_latex}}}$"
        else:
            tf_latex = f"$G(s) = \\frac{{{num_latex}}}{{{den_latex}}}$"
        
        # Set transfer function text in left box
        self.tf_text.set_text(tf_latex)
        
        # Extract system parameters
        time_constants, natural_freq, damped_freq, zeta = self._extract_system_parameters()
        
        # Set each parameter in its own box
        # Time constant(s)
        if time_constants is not None and len(time_constants) > 0:
            if len(time_constants) == 1:
                tau_str = f"$\\tau = {time_constants[0]:.4g}$"
            else:
                tau_list = ", ".join([f"{tau:.4g}" for tau in time_constants])
                tau_str = f"$\\tau = [{tau_list}]$"
        else:
            tau_str = "$\\tau =$ N/A"
        self.tau_text.set_text(tau_str)
        
        # Natural frequency
        if natural_freq is not None:
            omega_n_str = f"$\\omega_n = {natural_freq:.4g}$"
        else:
            omega_n_str = "$\\omega_n =$ N/A"
        self.omega_n_text.set_text(omega_n_str)
        
        # Damped frequency
        if damped_freq is not None:
            omega_d_str = f"$\\omega_d = {damped_freq:.4g}$"
        else:
            omega_d_str = "$\\omega_d =$ N/A"
        self.omega_d_text.set_text(omega_d_str)
        
        # Damping ratio
        if zeta is not None:
            zeta_str = f"$\\zeta = {zeta:.4g}$"
        else:
            zeta_str = "$\\zeta =$ N/A"
        self.zeta_text.set_text(zeta_str)

        # redraw
        self.canvas.draw_idle()


# ---------------------------
# main
# ---------------------------

def main():
    app = QApplication(sys.argv)
    win = BodeWindow()
    win.showMaximized()  # Start maximized with title bar visible
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
