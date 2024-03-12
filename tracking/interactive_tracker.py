# Typing
from __future__ import annotations
from typing import List

# Numpy
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Tkinter
import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog

# Python
import yaml
import ast
from collections import deque
from enum import Enum

# Tracking
from tracking.tracker import Tracker

# Utils
from tracking.utils.bbox import BBox
from tracking.bbox_tracking import Detection

# Logging
from tracking.utils.loggers import get_logger
logger = get_logger(__name__)


# Function to filter out numpy arrays from dictionary
def filter_dict(d):
    filtered = {}
    for key, value in d.items():
        if isinstance(value, (int, float, str)):
            filtered[key] = value
        elif isinstance(value, dict):
            filtered[key] = filter_dict(value)
    return filtered


class Mode(Enum):
    SOT = 0
    MOT = 1


class InteractiveTracker:

    default_confidence = 0.9

    def __init__(self):

        self.detections: List[Detection] = []
        self.start_point = None
        self.start_point_draw = None

        self.tracker = Tracker()
        self.mode = Mode.MOT

        self.window_closed = False

        self._init_plot()
        self._init_tkinter()


    def _init_plot(self):

        ax = self.tracker.show(show=False, title="")
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event',
                                                      self.on_click)
        self.fig = ax.get_figure()
        self.ax = ax

    def _init_tkinter(self):

        # Create the main window
        self.root = tk.Tk()
        self.root.tk.call('tk', 'scaling', 1.5)

        # Create a canvas for the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas_widget = self.canvas.get_tk_widget()

        # Create a notebook for multiple tabs
        self.notebook = ttk.Notebook(self.root)
        self.tabs = []

        # Buttons
        btn_predict = tk.Button(self.root, text="Prediction Step (P)",
                                command=self.prediction_step)
        btn_match = tk.Button(self.root, text="Matching Step (M)",
                              command=self.matching_step)
        btn_reset = tk.Button(self.root, text="Reset (R)",
                              command=self.reset)
        btn_read = tk.Button(self.root, text="Import Text (T)",
                             command=self.read_tracklet)
        self.btn_switch = tk.Button(self.root, text="Switch to SOT (T)",
                                    command=self.switch_mode)

        # Position the plot, text area, and buttons
        canvas_widget.grid(row=0, column=0, columnspan=3)
        btn_predict.grid(row=1, column=0, sticky="ew")
        btn_match.grid(row=1, column=1, sticky="ew")
        btn_reset.grid(row=1, column=2, sticky="ew")
        self.notebook.grid(row=2, column=0, columnspan=3, sticky="nsew")
        btn_read.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.btn_switch.grid(row=3, column=2, columnspan=1, sticky="ew")

        # Configure grid weights to allow text area to expand
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Key binding
        self.root.bind('<Key>', self.on_key_press)

        # Run the GUI
        self.refresh()
        self.refresh_mode()
        self.root.mainloop()


    def switch_mode(self):
        self.mode = Mode.MOT if self.mode == Mode.SOT else Mode.SOT

        self.reset()
        self.refresh_mode()

        if self.mode == Mode.SOT:
            self.tracker.set_partial_config(
                {
                    "matching": {
                        "association_1_iou": float(0.0),
                        "association_2_iou": float(0.0),
                        "association_3_iou": float(0.0)
                    },
                }
            )
        self.refresh()

    def refresh_mode(self):

        if self.mode == Mode.SOT:
            self.root.title("Interactive SOT")
            self.ax.set_title("Interactive SOT")
            self.btn_switch.config(text="Switch to MOT")
        else:
            self.root.title("Interactive MOT")
            self.ax.set_title("Interactive MOT")
            self.btn_switch.config(text="Switch to SOT")

        plt.draw()


    def add_tab(self, title="NEW TAB"):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=title)

        text_area = scrolledtext.ScrolledText(tab, wrap=tk.WORD,
                                              width=75, height=16)
        text_area.pack(fill="both", expand=True)
        tab.text_area = text_area


    def remove_tab(self):
        self.notebook.forget(self.notebook.tabs()[-1])


    def clear(self):
        for patch in self.ax.patches:
            if isinstance(patch, plt.Rectangle):
                patch.remove()
            if isinstance(patch, FancyArrow):
                patch.remove()
        for text in self.ax.texts:
            text.remove()
        for collection in self.ax.collections:
                collection.remove()


    def refresh(self, keep_detections=False):
        self.start_point=None
        self.update_text()
        self.clear()
        self.tracker.show(axes=self.ax, show=False)
        if not keep_detections:
            self.detections = []
        plt.draw()


    def reset(self):
        self.tracker = Tracker()
        self.refresh()


    def prediction_step(self):
        self.tracker.predict()
        self.refresh()


    def matching_step(self):
        self.tracker.associate_and_measurement_update(self.detections)
        self.refresh()


    def read_tracklet(self):

        # Config read
        config_id = self.notebook.tabs()[-1]
        config_tab = self.notebook.nametowidget(config_id)

        yaml_text = config_tab.text_area.get("1.0", tk.END)
        loaded_data = yaml.safe_load(yaml_text)

        self.tracker.set_partial_config(loaded_data)

        # Tracklets read
        for i, tracklet in enumerate(self.tracker.tracklets):
            tab = self.notebook.nametowidget(self.notebook.tabs()[i])

            text = tab.text_area.get("1.0", tk.END)
            lines = text.split("\n")
            state_list = lines[2].replace("[","").replace("]","").strip().split(",")
            state = np.array(state_list, dtype=float)

            covariance_lists = []
            for i in range(5,13):
                line = lines[i].replace("[","").replace("]","").strip().split(",")
                line = [float(x) for x in line if x != ""]
                covariance_lists.append(line)

            covariance = np.array(covariance_lists, dtype=float)

            tracklet.state = state
            tracklet.covariance = covariance

            history = ast.literal_eval(lines[15])
            tracklet.history = deque(history,
                                     maxlen=tracklet.config["history_maxlen"])

        # Refresh
        self.refresh()


    def update_text(self):

        N = len(self.tracker.tracklets) + 1
        for _ in range(N - len(self.notebook.tabs())):
            self.add_tab()

        for _ in range(len(self.notebook.tabs()) - N):
            self.remove_tab()

        for i, tracklet in enumerate(self.tracker.tracklets):
            self.notebook.tab(i, text=f"{tracklet.id}")

            state_str = np.array2string(
                tracklet.state, precision=1, floatmode='fixed', separator=',',
                formatter={'float': lambda x: f'{x:8.1f}'})
            covariance_str = np.array2string(
                tracklet.covariance, precision=1,floatmode='fixed',
                separator=',', formatter={'float': lambda x: f'{x:8.1f}'})

            history_str = str(list(tracklet.history))

            tab = self.notebook.nametowidget(self.notebook.tabs()[i])
            tab.text_area.delete('1.0', tk.END)
            tab.text_area.insert(tk.END, " |    x   |    y   |    w   |    h   |   vx   |   vy   |   vw   |   vh   |\n")
            tab.text_area.insert(tk.END, "Mean:\n ")
            tab.text_area.insert(tk.END, state_str)
            tab.text_area.insert(tk.END, " \n\nCovariance matrix:\n")
            tab.text_area.insert(tk.END, covariance_str)
            tab.text_area.insert(tk.END, "\n\nDetections matched history: (past --> present)\n")
            tab.text_area.insert(tk.END, history_str)

        # Config tab
        config_id = self.notebook.tabs()[-1]
        config_tab = self.notebook.nametowidget(config_id)
        self.notebook.tab(config_id, text="Config")

        filtered_dict = filter_dict(self.tracker.config)
        config = yaml.dump(filtered_dict, default_flow_style=False)
        config_tab.text_area.delete('1.0', tk.END)
        config_tab.text_area.insert(tk.END, config)


    def draw_detections(self):

        low_detections, mid_detections, high_detections = \
            self.tracker._classify_detections_by_confidence(self.detections)

        for detection in self.detections:

            if detection in low_detections:
                color = self.tracker.config["visualization"] \
                                       ["detection_color"]["low_confidence"]
            elif detection in mid_detections:
                color = self.tracker.config["visualization"] \
                                       ["detection_color"]["mid_confidence"]
            elif detection in high_detections:
                color = self.tracker.config["visualization"] \
                                       ["detection_color"]["high_confidence"]
            else:
                raise ValueError(f"Error when classifying the detections.")

            self.ax.text(*detection.corners()[3],
                f"({detection.confidence:.2f})", fontsize=8,
                color="white", ha='right', va='bottom',
                bbox=dict(facecolor=color, linewidth=0,
                boxstyle="round, pad=-0.05")
            )
            self.ax.texts[-1].set_alpha(1)

            detection.show(axes=self.ax, show_text=False,
                color=color, linestyle="dashed")

        plt.draw()


    def on_click(self, event):

        if event.inaxes != self.ax:
            return

        if event.button == 1:

            if self.start_point is None:
                self.start_point = (event.xdata, event.ydata)
                self.start_point_draw = self.ax.scatter(*self.start_point,
                                                        s = 50, c="k")
                plt.draw()

            else:
                end_point = (event.xdata, event.ydata)

                bbox = BBox.from_two_corners(*self.start_point, *end_point)
                detection = Detection.from_bbox(
                    bbox, 'Detection', InteractiveTracker.default_confidence
                )

                SOT_removed_detections = False
                if self.mode == Mode.SOT:
                    if len(self.detections) > 0:
                        SOT_removed_detections = True
                    self.detections = [detection]
                else:
                    self.detections.append(detection)

                self.start_point_draw.remove()
                self.start_point = None

                color = self.tracker.config["visualization"] \
                                       ["detection_color"]["high_confidence"]

                detection.show(axes=self.ax, show_text=False,
                    color=color, linestyle="dashed")
                plt.draw()

                confidence = self.ask_for_confidence()
                if confidence is None:
                    self.detections.pop()
                    self.refresh(keep_detections=True)
                    self.draw_detections()
                else:
                    for patch in self.ax.patches:
                        if patch.get_linestyle() == 'dashed' and \
                           patch.get_facecolor() != (0, 0, 0, 0):
                            patch.remove()
                    for text in self.ax.texts:
                        if text.get_text()[0] == '(' and text.get_alpha() == 1:
                            text.remove()
                    detection = self.detections[-1]
                    detection.confidence = confidence
                    self.draw_detections()

        elif event.button == 2:
            self.matching_step()

        elif event.button == 3:
            self.prediction_step()


    def on_key_press(self, event):
        focused_widget = self.root.focus_get()
        if type(focused_widget) == scrolledtext.ScrolledText:
            return

        if event.char == 'r':
            self.reset()
        elif event.char == 'p':
            self.prediction_step()
        elif event.char == 't':
            self.read_tracklet()
        elif event.char == 'm':
            self.matching_step()


    def ask_for_confidence(self) -> float:
        value = simpledialog.askfloat(
            "Confidence Score", "Enter confidence score:",
            initialvalue=InteractiveTracker.default_confidence
        )
        if value is None:
            return None
        value = float(value)
        if value < 0 or value > 1:
            del value
            return self.ask_for_confidence()
        return value


if __name__ == "__main__":
    InteractiveTracker()