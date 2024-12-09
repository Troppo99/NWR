import sys
import os
import importlib
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QImage
from PyQt6.QtWidgets import QScrollArea, QVBoxLayout, QHBoxLayout, QFrame, QPushButton, QLabel, QStyle, QMainWindow, QWidget
import cv2
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "Scripts")
CLEANING_DIR = os.path.join(SCRIPTS_DIR, "Cleaning")
BROOMING_DIR = os.path.join(SCRIPTS_DIR, "Brooming")

CLEANING_SCRIPTS = [f for f in os.listdir(CLEANING_DIR) if f.endswith(".py")]
BROOMING_SCRIPTS = [f for f in os.listdir(BROOMING_DIR) if f.endswith(".py")]

COLORS = {
    "background": "rgba(13, 13, 43, 0.8)",  # Deep cosmic blue with opacity
    "card_bg": "rgba(26, 26, 61, 0.8)",  # Darker blue for cards with opacity
    "accent": "#8A2BE2",  # Electric purple
    "accent_secondary": "#FF1493",  # Deep pink
    "accent_tertiary": "#4B0082",  # Indigo
    "success": "#00FFB2",  # Turquoise
    "warning": "#FF4081",  # Pink-red
    "error": "#FF0000",  # Red
    "text_primary": "#FFFFFF",
    "text_secondary": "#B8B8E0",  # Light purple
    "border": "#333366",  # Dark purple border
    "gradient_start": "#8A2BE2",  # Electric purple
    "gradient_end": "#4B0082",  # Indigo
    "ai_blue": "#00BFFF",  # Deep Sky Blue
    "cv_green": "#7FFF00",  # Chartreuse
    "dark_green": "#1C2C21",  # Dark green
    "dark_blue": "#1C304C",  # Dark blue
    "medium_green": "#1C5D32",  # Medium green
    "blue_green": "#2F7097",  # Blue-green
    "pale_green": "#ABEEE3",  # Pale green
}

STYLES = {
    "navbar_button": f"""
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #84fab0,
                                      stop:0.5 #8fd3f4,
                                      stop:1 #84fab0);
            border: none;
            color: {COLORS['text_primary']};
            padding: 12px 15px;
            font-weight: bold;
            border-radius: 10px;
            text-align: left;
            padding-left: 15px;
            padding-right: 15px;
        }}
        QPushButton:checked {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #8fd3f4,
                                      stop:0.5 #84fab0,
                                      stop:1 #8fd3f4);
        }}
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #8fd3f4,
                                      stop:1 #84fab0);
            border: 2px solid #ffffff;
        }}
    """,
    "run_button": f"""
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #18c729,
                                      stop:1 #FEF500);
            color: {COLORS['text_primary']};
            border: none;
            padding: 10px 20px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 8px;
        }}
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #1ed832,
                                      stop:1 #fff94c);
            margin-top: -1px;
        }}
        QPushButton:pressed {{
            margin-top: 1px;
        }}
    """,
    "stop_button": f"""
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #ff0000,
                                      stop:0.5 #ff4d4d,
                                      stop:1 #ff9999);
            color: {COLORS['text_primary']};
            border: none;
            padding: 10px 20px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 8px;
        }}
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #ff1a1a,
                                      stop:0.5 #ff6666,
                                      stop:1 #ffb3b3);
            margin-top: -1px;
        }}
        QPushButton:pressed {{
            margin-top: 1px;
        }}
        QPushButton:disabled {{
            background: #666666;
            color: #999999;
        }}
    """,
    "toggle_button": f"""
        QPushButton {{
            background: transparent;
            color: {COLORS['text_primary']};
            border: 2px solid {COLORS['border']};
            padding: 10px 20px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 8px;
            min-width: 100px;
        }}
        QPushButton:checked {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #4776E6,
                                      stop:1 #8E54E9);
            border: none;
        }}
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #5a87e6,
                                      stop:1 #9d69ea);
            margin-top: -1px;
        }}
        QPushButton:pressed {{
            margin-top: 1px;
        }}
    """,
}


class ModernFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modernFrame")
        self.setStyleSheet(
            f"""
            QFrame#modernFrame {{
                background-color: rgba(26, 26, 61, 0.8);
                border-radius: 15px;
                border: 2px solid {COLORS['border']};
                margin: 10px;
            }}
            QWidget#centralWidget {{
                background: transparent;
            }}
            """
        )
        self.hue = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_border_color)
        self.animation_active = False

    def start_border_animation(self):
        """Start the RGB border animation"""
        self.animation_active = True
        self.timer.start(50)  # Update every 50ms
        print("Border animation started")

    def stop_border_animation(self):
        """Stop the RGB border animation and reset border"""
        self.animation_active = False
        self.timer.stop()
        # Reset border to default color
        self.setStyleSheet(
            f"""
            QFrame#modernFrame {{
                background-color: rgba(26, 26, 61, 0.8);
                border-radius: 15px;
                border: 2px solid {COLORS['border']};
                margin: 10px;
            }}
            """
        )
        print("Border animation stopped")

    def update_border_color(self):
        """Update border color with a hue change for the RGB border effect"""
        if self.animation_active:
            colors = [
                COLORS["dark_green"],
                COLORS["dark_blue"],
                COLORS["medium_green"],
                COLORS["blue_green"],
                COLORS["pale_green"],
            ]
            color_index = (int(self.hue / (360 / len(colors)))) % len(colors)
            self.setStyleSheet(
                f"""
                QFrame#modernFrame {{
                    background-color: rgba(26, 26, 61, 0.8);
                    border-radius: 15px;
                    border: 2px solid {colors[color_index]};
                    margin: 10px;
                }}
                """
            )
            self.hue = (self.hue + 2) % 360


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, str)
    error_signal = pyqtSignal(str)
    started_signal = pyqtSignal(str)

    def __init__(self, monitor, script_name):
        super().__init__()
        self.monitor = monitor
        self.script_name = script_name
        self._running = True
        self.error_count = 0
        self.max_errors = 3
        self.last_frame_time = time.time()
        self.frame_timeout = 10

        # Tambahkan frame recovery
        self.last_valid_frame = None
        self.frame_recovery_attempts = 0
        self.max_recovery_attempts = 3

    def run(self):
        try:
            self.started_signal.emit(self.script_name)
            self._running = True
            self.monitor.frame_ready.connect(self.handle_frame)
            self.monitor.run()
        except Exception as e:
            self.error_signal.emit(f"{self.script_name}: {str(e)}")

    def handle_frame(self, frame):
        """Process each frame received from the video feed"""
        if not self._running:
            return

        try:
            if frame is not None:
                self.last_valid_frame = frame.copy()
                self.frame_recovery_attempts = 0
                self.frame_ready.emit(frame, self.script_name)
            elif self.last_valid_frame is not None and self.frame_recovery_attempts < self.max_recovery_attempts:
                # Gunakan frame terakhir yang valid jika frame current kosong
                self.frame_ready.emit(self.last_valid_frame, self.script_name)
                self.frame_recovery_attempts += 1

            self.last_frame_time = time.time()

        except Exception as e:
            self.error_signal.emit(f"Error processing frame: {str(e)}")

    def stop(self):
        self._running = False
        if self.monitor:
            try:
                self.monitor.frame_ready.disconnect()
            except:
                pass
            self.monitor.stop()
        self.wait()


class LoadingAnimation(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotate)
        self.timer.start(100)  # Adjust for performance

        self.setStyleSheet("background: transparent;")
        self.setFixedSize(75, 75)  # Increased size for better visibility

    def rotate(self):
        self.angle = (self.angle + 10) % 360
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Gradient color for spinner
        gradient = QtGui.QConicalGradient(37.5, 37.5, -self.angle)  # Center adjusted
        gradient.setColorAt(0, QtGui.QColor(COLORS["accent"]))
        gradient.setColorAt(0.5, QtGui.QColor(COLORS["success"]))
        gradient.setColorAt(1, QtGui.QColor(COLORS["accent"]))

        painter.setPen(QtGui.QPen(QtGui.QBrush(gradient), 5))
        painter.drawArc(10, 10, 55, 55, 0, 5760)  # Adjusted for increased size

    def start_animation(self):
        self.show()
        self.timer.start()

    def stop_animation(self):
        self.timer.stop()
        self.hide()


class SetupThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, script_name, category, cleaning_area_visible=False):
        super().__init__()
        self.script_name = script_name
        self.category = category
        self.cleaning_area_visible = cleaning_area_visible

    def run(self):
        try:
            module_name = os.path.splitext(self.script_name)[0]
            full_module_name = f"Scripts.{self.category}.{module_name}"
            if full_module_name in sys.modules:
                del sys.modules[full_module_name]
            spec = importlib.util.spec_from_file_location(
                full_module_name,
                os.path.join(SCRIPTS_DIR, self.category, self.script_name),
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            monitor = module.create_monitor()
            monitor.show_cleaning_area = self.cleaning_area_visible

            self.finished.emit({"monitor": monitor, "script_name": self.script_name})
        except Exception as e:
            self.error.emit(f"{self.script_name}: {str(e)}")


class VideoLabel(QtWidgets.QLabel):
    double_clicked = pyqtSignal(str)

    def __init__(self, script_name, parent=None):
        super().__init__(parent)
        self.script_name = script_name
        self.setMouseTracking(True)
        self.setMinimumSize(450, 300)  # Adjusted for better visibility
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self.script_name)
            event.accept()


class AnimatedButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)

        # Setup highlight effect
        self.highlight_effect = QtWidgets.QGraphicsDropShadowEffect(self)
        self.highlight_effect.setColor(QtGui.QColor(COLORS["accent"]))
        self.highlight_effect.setBlurRadius(20)
        self.highlight_effect.setOffset(0, 0)
        self.setGraphicsEffect(self.highlight_effect)
        self.highlight_effect.setEnabled(False)

        # Setup animation dengan multiple fallback options
        self.highlight_animation = QtCore.QPropertyAnimation(self.highlight_effect, b"blurRadius")
        self.highlight_animation.setDuration(300)

        # Fungsi helper untuk set easing curve dengan fallback
        def set_easing_curve():
            easing_options = [
                lambda: QtCore.QEasingCurve.Type.OutQuad,  # Opsi 1
                lambda: QtCore.QEasingCurve.Type.Linear,  # Opsi 2
            ]

            for option in easing_options:
                try:
                    self.highlight_animation.setEasingCurve(option())
                    break
                except (AttributeError, TypeError):
                    continue

        set_easing_curve()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.highlight_effect.setEnabled(True)
            self.highlight_animation.setStartValue(0)
            self.highlight_animation.setEndValue(20)  # Menentukan nilai akhir
            self.highlight_animation.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.highlight_animation.setStartValue(self.highlight_effect.blurRadius())
            self.highlight_animation.setEndValue(0)  # Menentukan nilai akhir
            self.highlight_animation.start()
            self.highlight_animation.finished.connect(lambda: self.highlight_effect.setEnabled(False))
        super().mouseReleaseEvent(event)


class NavbarButton(QPushButton):
    def __init__(self, text, icon_path=None, parent=None):
        super().__init__(text, parent)
        self.has_custom_icon = icon_path is not None
        self._icon_offset = 0

        # Setup basic button properties
        self.setStyleSheet(STYLES["navbar_button"])
        self.setCheckable(True)

        # Setup icon if provided
        if self.has_custom_icon:
            self.setIcon(QIcon(icon_path))
            self.setIconSize(QtCore.QSize(24, 24))

        # Setup effects
        self.highlight_effect = QtWidgets.QGraphicsDropShadowEffect(self)
        self.highlight_effect.setColor(QtGui.QColor(COLORS["accent"]))
        self.highlight_effect.setBlurRadius(20)
        self.highlight_effect.setOffset(0, 0)
        self.setGraphicsEffect(self.highlight_effect)
        self.highlight_effect.setEnabled(False)

        # Setup animations
        self.setup_animations()

    def setup_animations(self):
        # Highlight animation
        self.highlight_animation = QtCore.QPropertyAnimation(self.highlight_effect, b"blurRadius")
        self.highlight_animation.setDuration(300)

        # Scale animation untuk efek klik
        self.scale_animation = QtCore.QPropertyAnimation(self, b"geometry")
        self.scale_animation.setDuration(100)

        # Setup easing curves dengan proper type
        try:
            # Menggunakan Type enum untuk easing curve
            self.highlight_animation.setEasingCurve(QtCore.QEasingCurve.Type.OutQuad)
            self.scale_animation.setEasingCurve(QtCore.QEasingCurve.Type.OutQuad)

            if self.has_custom_icon:
                self.icon_animation = QtCore.QPropertyAnimation(self, b"iconOffset")
                self.icon_animation.setDuration(150)
                self.icon_animation.setEasingCurve(QtCore.QEasingCurve.Type.OutQuad)

        except AttributeError:
            # Fallback ke Linear jika OutQuad tidak tersedia
            self.highlight_animation.setEasingCurve(QtCore.QEasingCurve.Type.Linear)
            self.scale_animation.setEasingCurve(QtCore.QEasingCurve.Type.Linear)

            if self.has_custom_icon:
                self.icon_animation = QtCore.QPropertyAnimation(self, b"iconOffset")
                self.icon_animation.setDuration(150)
                self.icon_animation.setEasingCurve(QtCore.QEasingCurve.Type.Linear)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            geo = self.geometry()
            pressed_geo = geo.adjusted(2, 2, -2, -2)

            self.scale_animation.setStartValue(geo)
            self.scale_animation.setEndValue(pressed_geo)
            self.scale_animation.start()

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            geo = self.geometry()
            original_geo = geo.adjusted(-2, -2, 2, 2)

            self.scale_animation.setStartValue(geo)
            self.scale_animation.setEndValue(original_geo)
            self.scale_animation.start()

        super().mouseReleaseEvent(event)

    @QtCore.pyqtProperty(int)
    def iconOffset(self):
        return self._icon_offset

    @iconOffset.setter
    def iconOffset(self, value):
        self._icon_offset = value
        self.update()


class ScriptManagerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Set background image
        background_path = os.path.join(BASE_DIR, "Assets", "background.png")
        if os.path.exists(background_path):
            # Convert path separators to forward slashes for Qt stylesheet
            background_path = background_path.replace("\\", "/")
            self.setStyleSheet(
                f"""
                QMainWindow {{
                    background-image: url({background_path});
                    background-position: center center;
                    background-repeat: no-repeat;
                }}
                QWidget#centralWidget {{
                    background: transparent;
                }}
                """
            )

        # Initialize all necessary attributes
        self.processes = {}
        self.video_labels = {}
        self.video_containers = {}
        self.script_widgets = {}
        self.status_timers = {}
        self.status_colors = {}
        self.cleaning_areas_visible = {}
        self.setup_threads = {}
        self.clean_toggle_buttons = {}
        self.loading_animations = {}
        self.video_positions = {}
        self.maximized_script = None
        self.error_counts = {}  # Track error counts per script
        self.max_errors = 3  # Maximum errors before stopping script

        # Set window properties
        self.setWindowTitle("Video Monitoring System")
        self.resize(1920, 1080)  # Updated to 1920x1080
        self.setMinimumSize(1280, 720)  # Set minimum size to prevent terlalu kecil

        # Current category (default: Cleaning)
        self.current_category = "Cleaning"

        # Scripts organized by category
        self.scripts = {"Cleaning": CLEANING_SCRIPTS, "Brooming": BROOMING_SCRIPTS}

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        # Make central widget transparent
        main_widget = QWidget()
        main_widget.setObjectName("centralWidget")
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)  # Outer margins
        main_layout.setSpacing(10)  # Increased spacing between navbar and content
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Navbar
        navbar = QWidget()
        navbar.setFixedHeight(90)  # Increased height for 1920x1080
        navbar.setStyleSheet(
            f"""
            QWidget {{
                background-color: rgba(26, 26, 61, 0.8);
                border-radius: 20px;
                border: 2px solid {COLORS['border']};
                margin: 0px;
            }}
        """
        )
        navbar_layout = QHBoxLayout(navbar)
        navbar_layout.setContentsMargins(20, 20, 20, 20)  # Adjust internal navbar padding
        navbar_layout.setSpacing(70)  # Increased spacing between elements

        # Title in navbar
        title_label = QLabel("VISUAL AI Monitoring System")
        title_font = QtGui.QFont("Arial", 20, QtGui.QFont.Weight.Bold)  # Larger font
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        navbar_layout.addWidget(title_label)

        # Add some spacing between title and buttons
        navbar_layout.addSpacing(50)

        # Cleaning button with icon
        cleaning_btn = NavbarButton("Cleaning", os.path.join(BASE_DIR, "Assets", "clean.png"))
        cleaning_btn.setChecked(True)
        cleaning_btn.clicked.connect(lambda: self.switch_category("Cleaning"))

        # Brooming button with icon
        brooming_btn = NavbarButton("Brooming", os.path.join(BASE_DIR, "Assets", "broom.png"))
        brooming_btn.clicked.connect(lambda: self.switch_category("Brooming"))

        navbar_layout.addWidget(cleaning_btn)
        navbar_layout.addWidget(brooming_btn)
        navbar_layout.addStretch()

        self.category_buttons = {"Cleaning": cleaning_btn, "Brooming": brooming_btn}
        main_layout.addWidget(navbar)

        # Content area with sidebar and container
        self.content_widget = QWidget()
        self.content_layout = QHBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)  # Remove internal margins
        self.content_layout.setSpacing(15)  # Increased spacing between sidebar and content
        main_layout.addWidget(self.content_widget, 1)

        self.build_ui_for_category()

    def switch_category(self, category):
        if category == self.current_category:
            return

        # Update category buttons
        for cat, btn in self.category_buttons.items():
            btn.setChecked(cat == category)

        self.current_category = category

        # Rebuild the UI
        self.build_ui_for_category()

    def build_ui_for_category(self):
        # Clear previous widgets
        for i in reversed(range(self.content_layout.count())):
            widget = self.content_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Clear data structures
        self.video_labels.clear()
        self.video_containers.clear()
        self.script_widgets.clear()
        self.status_timers.clear()
        self.status_colors.clear()
        self.clean_toggle_buttons.clear()
        self.loading_animations.clear()
        self.video_positions.clear()
        self.setup_threads.clear()
        self.cleaning_areas_visible.clear()

        # Rebuild the UI

        # Sidebar setup
        sidebar_scroll_area = QScrollArea()
        sidebar_scroll_area.setFixedWidth(375)  # Adjusted width for 1920x1080
        sidebar_scroll_area.setWidgetResizable(True)
        sidebar_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll_area.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        sidebar_scroll_area.setStyleSheet(
            f"""
            QScrollArea {{
                background-color: rgba(26, 26, 61, 0.8);
                border-radius: 15px;
                border: 2px solid {COLORS['border']};
                margin: 0px;
            }}
            QWidget#centralWidget {{
                background: transparent;
            }}
            """
        )

        sidebar_content = QWidget()
        sidebar_content.setObjectName("centralWidget")
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Logo or AI Icon
        logo_container = QWidget()
        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ai_image_path = os.path.join(BASE_DIR, "Assets", "logoo.png")
        if os.path.exists(ai_image_path):
            ai_pixmap = QPixmap(ai_image_path)

            scaled_ai_pixmap = ai_pixmap.scaled(
                150,
                150,  # Increased size for better visibility
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_label.setPixmap(scaled_ai_pixmap)
        else:
            logo_label.setText("AI")
            logo_label.setStyleSheet("color: #FFFFFF; font-size: 36px;")  # Increased font size
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_layout.addWidget(logo_label, alignment=Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(logo_container)

        # Spacer
        sidebar_layout.addSpacing(20)

        # Control Buttons
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)

        run_all_btn = AnimatedButton("Run All")
        run_all_btn.setFixedHeight(45)  # Increased height
        run_all_btn.setFixedWidth(90)  # Increased width
        run_all_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        run_all_btn.setStyleSheet(STYLES["run_button"])
        run_all_btn.clicked.connect(self.run_all_scripts)

        stop_all_btn = AnimatedButton("Stop All")
        stop_all_btn.setFixedHeight(45)  # Increased height
        stop_all_btn.setFixedWidth(90)  # Increased width
        stop_all_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        stop_all_btn.setStyleSheet(STYLES["stop_button"])
        stop_all_btn.clicked.connect(self.stop_all_scripts)

        control_layout.addWidget(run_all_btn)
        control_layout.addWidget(stop_all_btn)
        sidebar_layout.addWidget(control_panel)

        # Spacer
        sidebar_layout.addSpacing(20)

        # Script Control Panels
        scripts_in_category = self.scripts[self.current_category]
        for script in scripts_in_category:
            script_frame = QFrame()
            script_frame.setObjectName("scriptFrame")
            script_layout = QVBoxLayout(script_frame)
            script_layout.setContentsMargins(15, 15, 15, 15)  # Increased margins
            script_layout.setSpacing(15)  # Increased spacing

            # Script Name with Icon
            script_name_layout = QHBoxLayout()
            icon_label = QLabel()
            code_image_path = os.path.join(BASE_DIR, "Assets", "UI.png")
            if os.path.exists(code_image_path):
                code_pixmap = QPixmap(code_image_path)
                scaled_code_pixmap = code_pixmap.scaled(
                    36,
                    36,  # Increased size for better visibility
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                icon_label.setPixmap(scaled_code_pixmap)
            else:
                icon_label.setText("💻")
                icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            script_name_label = QLabel(script)
            script_name_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 14px;")
            script_name_layout.addWidget(icon_label)
            script_name_layout.addWidget(script_name_label)
            script_layout.addLayout(script_name_layout)

            # Buttons Container
            btn_container = QWidget()
            btn_layout = QHBoxLayout(btn_container)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            btn_layout.setSpacing(10)  # Increased spacing

            run_btn = AnimatedButton("")
            run_btn.setStyleSheet(STYLES["run_button"])
            run_btn.clicked.connect(lambda checked, s=script: self.run_script(s))
            run_btn.setFixedWidth(40)  # Increased width
            run_btn.setFixedHeight(40)  # Increased height
            run_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            stop_btn = AnimatedButton("")
            stop_btn.setStyleSheet(STYLES["stop_button"])
            stop_btn.clicked.connect(lambda checked, s=script: self.stop_script(s))
            stop_btn.setFixedWidth(40)  # Increased width
            stop_btn.setFixedHeight(40)  # Increased height
            stop_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))

            clean_toggle = AnimatedButton("AKTIF")
            clean_toggle.setCheckable(True)
            clean_toggle.setStyleSheet(STYLES["toggle_button"])
            clean_toggle.setFixedWidth(100)  # Increased width
            clean_toggle.setFixedHeight(40)  # Increased height
            clean_toggle.clicked.connect(lambda checked, s=script: self.toggle_cleaning_area(s, checked))
            self.clean_toggle_buttons[script] = clean_toggle

            btn_layout.addWidget(run_btn)
            btn_layout.addWidget(stop_btn)
            btn_layout.addWidget(clean_toggle)
            script_layout.addWidget(btn_container)

            # Status Indicator
            status_layout = QHBoxLayout()
            status_indicator = QLabel()
            status_indicator.setFixedSize(18, 18)  # Increased size
            status_indicator.setStyleSheet(f"background-color: {COLORS['text_secondary']}; border-radius: 9px;")
            status_label = QLabel("Not Running")
            status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
            status_layout.addWidget(status_indicator)
            status_layout.addWidget(status_label)
            status_layout.addStretch()
            script_layout.addLayout(status_layout)

            self.script_widgets[script] = {
                "status_indicator": status_indicator,
                "status_label": status_label,
            }

            sidebar_layout.addWidget(script_frame)

        # Spacer to push content to top
        sidebar_layout.addStretch()

        # Set sidebar content to scroll area
        sidebar_scroll_area.setWidget(sidebar_content)
        self.content_layout.addWidget(sidebar_scroll_area)

        # Content Area with Grid Layout
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content.setStyleSheet("background: transparent;")
        content.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        scroll_area.setStyleSheet(
            f"""
            QScrollArea {{
                background-color: rgba(26, 26, 61, 0.8);
                border-radius: 15px;
                border: 2px solid {COLORS['border']};
                margin: 0px;
            }}
            QWidget#centralWidget {{
                background: transparent;
            }}
            """
        )

        scroll_content = QWidget()
        scroll_content.setObjectName("centralWidget")
        self.grid_layout = QtWidgets.QGridLayout(scroll_content)
        self.grid_layout.setSpacing(15)  # Increased spacing
        self.grid_layout.setContentsMargins(15, 15, 15, 15)  # Increased margins

        # Create video containers
        for i, script in enumerate(scripts_in_category):
            container = ModernFrame()
            container.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(15, 15, 15, 15)  # Increased margins
            container_layout.setSpacing(15)  # Increased spacing

            video_label = VideoLabel(script)
            video_label.setMinimumSize(450, 300)  # Increased size
            video_label.setText(f"Video feed for {script}")
            video_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px;")
            video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            video_label.double_clicked.connect(self.handle_video_double_click)

            container_layout.addWidget(video_label)

            # Miss Detection Panel
            miss_detection_panel = QWidget()
            miss_detection_layout = QHBoxLayout(miss_detection_panel)
            miss_detection_layout.setContentsMargins(0, 0, 0, 0)
            miss_detection_layout.setSpacing(15)  # Increased spacing

            mark_miss_btn = AnimatedButton("Mark Miss")
            mark_miss_btn.setFixedHeight(40)  # Increased height
            mark_miss_btn.setMinimumWidth(120)  # Increased width
            mark_miss_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #6C5DD3;
                    color: white;
                    border-radius: 8px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #7C6DE3;
                }
                """
            )

            clear_miss_btn = AnimatedButton("Clear Miss")
            clear_miss_btn.setFixedHeight(40)  # Increased height
            clear_miss_btn.setMinimumWidth(120)  # Increased width
            clear_miss_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #FF754C;
                    color: white;
                    border-radius: 8px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #FF8B6A;
                }
                """
            )

            miss_status_label = QLabel("No miss detection")
            miss_status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")

            miss_detection_layout.addWidget(mark_miss_btn)
            miss_detection_layout.addWidget(clear_miss_btn)
            miss_detection_layout.addWidget(miss_status_label)
            miss_detection_layout.addStretch()

            container_layout.addWidget(miss_detection_panel)

            # Add to grid layout
            row = i // 2
            col = i % 2
            self.grid_layout.addWidget(container, row, col)
            self.video_positions[script] = (row, col)

            # Store references
            self.video_labels[script] = video_label
            self.video_containers[script] = container
            self.script_widgets[script]["miss_status_label"] = miss_status_label

            # Connect buttons
            mark_miss_btn.clicked.connect(lambda checked, s=script: self.mark_miss_detection(s))
            clear_miss_btn.clicked.connect(lambda checked, s=script: self.clear_miss_detection(s))

        scroll_area.setWidget(scroll_content)
        content_layout.addWidget(scroll_area)
        self.content_layout.addWidget(content, stretch=1)

        # Info Panel Area
        info_panel = QScrollArea()
        info_panel.setFixedWidth(375)  # Adjusted width for 1920x1080
        info_panel.setWidgetResizable(True)
        info_panel.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        info_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        info_panel.setStyleSheet(
            f"""
            QScrollArea {{
                background-color: rgba(26, 26, 61, 0.8);
                border-radius: 15px;
                border: 2px solid {COLORS['border']};
                margin: 0px;
            }}
            QWidget#infoWidget {{
                background: transparent;
            }}
            """
        )

        # Info Panel Content
        info_content = QWidget()
        info_content.setObjectName("infoWidget")
        info_layout = QVBoxLayout(info_content)
        info_layout.setContentsMargins(15, 15, 15, 15)  # Increased margins
        info_layout.setSpacing(15)  # Increased spacing

        # Title untuk Info Panel
        info_title = QLabel("Script Information")
        info_title.setStyleSheet(
            f"""
            color: {COLORS['text_primary']};
            font-size: 20px;
            font-weight: bold;
            padding: 5px;
            """
        )
        info_layout.addWidget(info_title)

        # Placeholder untuk informasi script
        for script in self.scripts[self.current_category]:
            script_info_frame = QFrame()
            script_info_frame.setStyleSheet(
                f"""
                QFrame {{
                    background-color: rgba(38, 38, 85, 0.6);
                    border-radius: 8px;
                    padding: 10px;
                }}
                """
            )
            script_info_layout = QVBoxLayout(script_info_frame)

            # Script name
            script_name_label = QLabel(script)
            script_name_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: bold; font-size: 14px;")
            script_info_layout.addWidget(script_name_label)

            # Placeholder untuk status dan statistik
            status_label = QLabel("Status: Not Running")
            status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
            script_info_layout.addWidget(status_label)

            # Tambahkan ke info layout
            info_layout.addWidget(script_info_frame)

        # Spacer di akhir
        info_layout.addStretch()

        # Set content ke scroll area
        info_panel.setWidget(info_content)

        # Tambahkan info panel ke content layout
        self.content_layout.addWidget(info_panel)

    def run_script(self, script_name):
        if script_name in self.processes:
            return

        try:
            # Setup loading animation
            label = self.video_labels[script_name]
            loading = LoadingAnimation(label)
            loading.move(
                (label.width() - loading.width()) // 2,
                (label.height() - loading.height()) // 2,
            )
            loading.start_animation()
            self.loading_animations[script_name] = loading

            # Update status
            self.update_script_status(script_name, "Starting...", COLORS["accent"])

            # Setup and run in a separate thread
            setup_thread = SetupThread(
                script_name,
                self.current_category,
                self.cleaning_areas_visible.get(script_name, False),
            )
            self.setup_threads[script_name] = setup_thread

            setup_thread.finished.connect(lambda data: self.on_setup_finished(data, script_name))
            setup_thread.error.connect(lambda err: self.on_setup_error(err, script_name))
            setup_thread.start()

        except Exception as e:
            self.handle_script_error(script_name, str(e))

    def on_setup_finished(self, data, script_name):
        monitor = data["monitor"]
        thread = VideoThread(monitor, script_name)
        thread.frame_ready.connect(self.update_video_feed)
        thread.error_signal.connect(self.handle_video_error)
        thread.started_signal.connect(lambda s: self.handle_video_started(s, self.loading_animations[script_name]))

        self.processes[script_name] = {"thread": thread, "monitor": monitor}
        thread.start()

        # Start border animation
        if script_name in self.video_containers:
            container = self.video_containers[script_name]
            container.start_border_animation()

        # Cleanup setup thread
        del self.setup_threads[script_name]

    def on_setup_error(self, error_msg, script_name):
        if script_name in self.loading_animations:
            self.loading_animations[script_name].stop_animation()
        self.handle_video_error(error_msg)
        del self.setup_threads[script_name]

    def handle_video_started(self, script_name, loading):
        """Handle when video successfully starts"""
        loading.stop_animation()
        self.update_script_status(script_name, "Running", COLORS["success"])
        self.create_status_timer(script_name)

        # Start border animation
        if script_name in self.video_containers:
            container = self.video_containers[script_name]
            container.start_border_animation()

        # Reset error count on successful start
        self.error_counts[script_name] = 0

    def update_video_feed(self, frame, script_name):
        """Update video feed for a specific script"""
        try:
            if script_name in self.video_labels:
                label = self.video_labels[script_name]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape

                # Get label size
                label_size = label.size()
                label_width = label_size.width()
                label_height = label_size.height()

                # Calculate scaling factor to fit frame within label, maintaining aspect ratio
                scale_w = label_width / w
                scale_h = label_height / h
                scale = min(scale_w, scale_h)

                new_w = int(w * scale)
                new_h = int(h * scale)

                # Resize frame
                rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Convert to QImage
                bytes_per_line = ch * new_w
                qt_image = QImage(
                    rgb_frame.data,
                    new_w,
                    new_h,
                    bytes_per_line,
                    QImage.Format.Format_RGB888,
                )

                # Convert to pixmap
                pixmap = QPixmap.fromImage(qt_image)

                # Set pixmap to label, centered
                label.setPixmap(pixmap)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        except Exception as e:
            print(f"Error updating video feed: {str(e)}")

    def stop_script(self, script_name):
        if script_name in self.processes:
            print(f"Stopping {script_name}")

            # Stop border animation
            if script_name in self.video_containers:
                container = self.video_containers[script_name]
                container.stop_border_animation()

            # Stop thread dan monitor
            thread = self.processes[script_name]["thread"]
            monitor = self.processes[script_name]["monitor"]

            try:
                thread.frame_ready.disconnect()
                monitor.frame_ready.disconnect()
            except:
                pass

            thread.stop()
            monitor.stop()
            thread.wait()

            # Clear label
            label = self.video_labels[script_name]
            label.clear()
            label.setText(f"Video feed for {script_name}")
            label.setStyleSheet("color: #808191; font-size: 14px;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Update status
            self.update_script_status(script_name, "Not Running", COLORS["text_secondary"])

            # Stop timers
            if script_name in self.status_timers:
                self.status_timers[script_name].stop()
                del self.status_timers[script_name]

            del self.processes[script_name]

    def run_all_scripts(self):
        """Run all scripts"""
        scripts = self.scripts[self.current_category]
        for script in scripts:
            self.run_script(script)

    def stop_all_scripts(self):
        """Stop all running scripts"""
        scripts = self.scripts[self.current_category]
        for script in list(self.processes.keys()):
            self.stop_script(script)

    def toggle_cleaning_area(self, script_name, checked):
        """Toggle visibility of cleaning area polygon"""
        try:
            self.cleaning_areas_visible[script_name] = checked

            if script_name in self.processes:
                monitor = self.processes[script_name]["monitor"]
                monitor.show_cleaning_area = checked

            # Update button state
            if script_name in self.clean_toggle_buttons:
                self.clean_toggle_buttons[script_name].setChecked(checked)

            print(f"Cleaning area for {script_name} {'shown' if checked else 'hidden'}")

        except Exception as e:
            print(f"Error toggling cleaning area: {str(e)}")

    def update_script_status(self, script_name, status, color):
        """Update status label for a script"""
        if script_name in self.script_widgets:
            self.script_widgets[script_name]["status_label"].setText(status)
            self.script_widgets[script_name]["status_label"].setStyleSheet(f"color: {color}; font-size: 14px;")

    def create_status_timer(self, script_name):
        """Create RGB color cycling timer"""
        timer = QTimer()
        self.status_colors[script_name] = 0
        timer.timeout.connect(lambda: self.update_status_color(script_name))
        timer.start(50)  # Update faster for smoother transition
        self.status_timers[script_name] = timer

    def update_status_color(self, script_name):
        """Update status indicator color with color wheel"""
        if script_name in self.script_widgets:
            hue = int(self.status_colors[script_name])  # Convert to integer
            color = QtGui.QColor.fromHsv(hue, 255, 255)
            self.status_colors[script_name] = (hue + 4) % 360  # Use integer increment

            self.script_widgets[script_name]["status_indicator"].setStyleSheet(f"background-color: {color.name()}; border-radius: 9px;")

    def handle_video_double_click(self, script_name):
        """Handle double click event on video label"""
        container = self.video_containers[script_name]
        if self.maximized_script is None:
            # Save original geometry
            self.original_geometry = container.geometry()

            # Calculate new size (1.5x larger for better balance)
            new_width = min(int(container.width() * 1.5), self.width() - 200)  # Adjusted margin
            new_height = min(int(container.height() * 1.5), self.height() - 200)  # Adjusted margin

            # Calculate center position
            parent = container.parent()
            new_x = (parent.width() - new_width) // 2
            new_y = (parent.height() - new_height) // 2

            # Create animation for container
            self.anim = QtCore.QPropertyAnimation(container, b"geometry")
            self.anim.setDuration(300)
            self.anim.setStartValue(container.geometry())
            self.anim.setEndValue(QtCore.QRect(new_x, new_y, new_width, new_height))
            self.anim.start()

            # Bring container to front
            container.raise_()
            self.maximized_script = script_name

        else:
            # Animate back to original size
            container = self.video_containers[self.maximized_script]
            self.anim = QtCore.QPropertyAnimation(container, b"geometry")
            self.anim.setDuration(300)
            self.anim.setStartValue(container.geometry())
            self.anim.setEndValue(self.original_geometry)
            self.anim.start()

            self.maximized_script = None

    def mark_miss_detection(self, script_name):
        """Functionality for Mark Miss button"""
        import pickle

        miss_detection_data = {"script": script_name, "timestamp": time.time()}
        pkl_filename = script_name.replace(".py", "").upper() + ".pkl"  # e.g., 'CLEANINGKANTIN.pkl'
        miss_detection_folder = os.path.join(BASE_DIR, "Miss Detection")
        if not os.path.exists(miss_detection_folder):
            os.makedirs(miss_detection_folder)
        pkl_filepath = os.path.join(miss_detection_folder, pkl_filename)

        with open(pkl_filepath, "wb") as f:
            pickle.dump(miss_detection_data, f)
        print(f"Miss detection data saved to {pkl_filepath}")

        # Update status label
        if script_name in self.script_widgets:
            self.script_widgets[script_name]["miss_status_label"].setText("Miss detection saved")

    def clear_miss_detection(self, script_name):
        """Functionality for Clear Miss button"""
        pkl_filename = script_name.replace(".py", "").upper() + ".pkl"
        miss_detection_folder = os.path.join(BASE_DIR, "Miss Detection")
        pkl_filepath = os.path.join(miss_detection_folder, pkl_filename)
        if os.path.exists(pkl_filepath):
            os.remove(pkl_filepath)
            print(f"Miss detection data cleared from {pkl_filepath}")

        # Update status label
        if script_name in self.script_widgets:
            self.script_widgets[script_name]["miss_status_label"].setText("Miss detection cleared")

    def handle_video_error(self, error_msg):
        """Handle video feed errors"""
        # Extract script name from error message
        script_name = error_msg.split(":")[0].strip()
        error_details = error_msg.split(":", 1)[1].strip()

        # Increment error count
        if script_name not in self.error_counts:
            self.error_counts[script_name] = 0
        self.error_counts[script_name] += 1

        # Update status with error
        self.update_script_status(script_name, f"Error: {error_details}", COLORS["warning"])

        # Stop loading animation jika ada
        if script_name in self.loading_animations:
            self.loading_animations[script_name].stop_animation()

        # Check if max errors reached
        if self.error_counts[script_name] >= self.max_errors:
            print(f"Max errors reached for {script_name}, stopping script")
            self.stop_script(script_name)
            self.error_counts[script_name] = 0  # Reset error count

            # Update status to indicate max errors reached
            self.update_script_status(script_name, "Stopped: Too many errors", COLORS["warning"])
        else:
            print(f"Error in {script_name}: {error_details} (Error {self.error_counts[script_name]}/{self.max_errors})")

    def closeEvent(self, event):
        """Handle application close event to ensure all threads are properly stopped"""
        self.stop_all_scripts()
        event.accept()


class SplashScreen(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Set window flags
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.SplashScreen)

        # Set transparent background
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        # Set the size of the splash screen
        self.setFixedSize(900, 600)  # Adjusted for 1920x1080
        self.center()

        # Create a layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(75, 75, 75, 75)  # Increased margins
        layout.setSpacing(20)  # Increased spacing

        # Initialize logo_label sebagai atribut kelas
        self.logo_label = QLabel(self)  # Initialize first

        # Load and setup the logo
        logo_path = os.path.join(BASE_DIR, "Assets", "logoo.png")
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            self.logo_label.setPixmap(
                logo_pixmap.scaled(
                    350,
                    350,  # Increased size for better visibility
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addStretch()
            layout.addWidget(self.logo_label, alignment=Qt.AlignmentFlag.AlignCenter)

            # Add glow effect to the logo
            glow_effect = QtWidgets.QGraphicsDropShadowEffect()
            glow_effect.setBlurRadius(75)  # Increased blur radius
            glow_effect.setColor(QtGui.QColor(COLORS["accent"]))  # Purple accent
            glow_effect.setOffset(0)
            self.logo_label.setGraphicsEffect(glow_effect)
        else:
            # Handle case when logo doesn't exist
            self.logo_label.setText("LOGO")
            self.logo_label.setStyleSheet("color: #FFFFFF; font-size: 36px;")  # Increased font size
            self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addStretch()
            layout.addWidget(self.logo_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Application name label dengan fade-in animation
        app_name_label = QtWidgets.QLabel("VISUAL AI Monitoring System", self)
        app_name_label.setStyleSheet("color: #FFFFFF; font-size: 36px;")  # Increased font size
        app_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(app_name_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add loading animation
        loading_label = QLabel(self)
        loading_gif_path = os.path.join(BASE_DIR, "Assets", "loading.gif")  # Pastikan path benar
        if os.path.exists(loading_gif_path):
            loading_movie = QtGui.QMovie(loading_gif_path)
            loading_label.setMovie(loading_movie)
            loading_movie.start()
            loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(loading_label, alignment=Qt.AlignmentFlag.AlignCenter)
        else:
            # Jika loading GIF tidak tersedia, tampilkan placeholder
            loading_label.setText("Loading...")
            loading_label.setStyleSheet("color: #FFFFFF; font-size: 24px;")  # Increased font size
            loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(loading_label, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()

        # Add opacity effect to app_name_label
        self.opacity_effect = QtWidgets.QGraphicsOpacityEffect()
        app_name_label.setGraphicsEffect(self.opacity_effect)

        # Create an animation untuk app name label
        self.opacity_animation = QtCore.QPropertyAnimation(self.opacity_effect, b"opacity")
        self.opacity_animation.setDuration(2000)  # 2 seconds
        self.opacity_animation.setStartValue(0)
        self.opacity_animation.setEndValue(1)
        self.opacity_animation.start()

    def center(self):
        # Center the window on the screen
        qr = self.frameGeometry()
        cp = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def paintEvent(self, event):
        # Paint gradient background
        painter = QtGui.QPainter(self)
        rect = self.rect()

        # Convert QRect to float coordinates untuk gradient
        gradient = QtGui.QLinearGradient(float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height()))

        # Set gradient colors for cosmic/galaxy effect
        gradient.setColorAt(0.0, QtGui.QColor("#0D0D2B"))  # Deep cosmic blue
        gradient.setColorAt(0.3, QtGui.QColor("#1A1A3A"))  # Dark purple
        gradient.setColorAt(0.6, QtGui.QColor("#8A2BE2"))  # Electric purple
        gradient.setColorAt(0.8, QtGui.QColor("#FF1493"))  # Deep pink
        gradient.setColorAt(1.0, QtGui.QColor("#4B0082"))  # Indigo

        painter.fillRect(rect, gradient)
        super().paintEvent(event)

    def showEvent(self, event):
        # Fade in the splash screen
        self.setWindowOpacity(0)
        self.fade_in_animation = QtCore.QPropertyAnimation(self, b"windowOpacity")
        self.fade_in_animation.setDuration(1000)  # 1 second
        self.fade_in_animation.setStartValue(0)
        self.fade_in_animation.setEndValue(1)
        self.fade_in_animation.start()
        super().showEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app_icon_path = os.path.join(BASE_DIR, "Assets", "app.png")
    if os.path.exists(app_icon_path):
        app_icon = QIcon(app_icon_path)
        app.setWindowIcon(app_icon)

    app.setStyle("Fusion")  # Modern style

    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(19, 17, 28))
    dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25, 25, 25))
    dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)
    splash = SplashScreen()
    splash.show()
    window = ScriptManagerApp()

    def finish_splash():
        fade_out = QtCore.QPropertyAnimation(splash, b"windowOpacity")
        fade_out.setDuration(1000)
        fade_out.setStartValue(1)
        fade_out.setEndValue(0)
        fade_out.finished.connect(splash.close)
        fade_out.start()
        window.show()

    QtCore.QTimer.singleShot(3000, finish_splash)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
