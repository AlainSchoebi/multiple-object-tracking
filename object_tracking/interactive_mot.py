# Typing
from __future__ import annotations
from typing import List

# Numpy
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Python
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import yaml

# Tracking
from tracklet import Tracklet
from tracker import Tracker

# Utils
from utils.bbox import BBox
from bbox_tracking import Detection

# Logging
from utils.loggers import get_logger
logger = get_logger(__name__)


class InteractiveMOT:

    detection_color = np.array([0.1,0.2,0.8])
    state_color = np.array([0.5,0.5,0.5])


    def __init__(self):

        self.detections: List[Detection] = []
        self.start_point = None
        self.start_point_draw = None

        self.tracker = Tracker()

        self.init_plot()
        self.init_tkinter()


    def init_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.invert_yaxis()

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, fc=InteractiveMOT.state_color, ec=InteractiveMOT.state_color*0.7, lw=1),  # Blue rectangle for detections
            plt.Rectangle((0, 0), 1, 1, fc=InteractiveMOT.detection_color, ec=InteractiveMOT.detection_color*0.7, lw=1)
        ]
        ax.legend(legend_handles, ['Mean State', 'Detections'], loc='lower right')

        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig = fig
        self.ax = ax

    def init_tkinter(self):
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Interactive MOT")

        # Create a canvas for the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas_widget = self.canvas.get_tk_widget()

        # Create a notebook for multiple tabs
        self.notebook = ttk.Notebook(self.root)
        self.tabs = []

        btn_predict = tk.Button(self.root, text="Prediction Step (right click or P)", command=self.prediction_step)
        btn_match = tk.Button(self.root, text="Matching Step (M)", command=self.matching_step)
        btn_reset = tk.Button(self.root, text="Reset (R)", command=self.reset)
        btn_read = tk.Button(self.root, text="Import Text (T)", command=self.read_tracklet)

        # Position the plot, text area, and buttons using the grid layout manager
        canvas_widget.grid(row=0, column=0, columnspan=3)
        btn_predict.grid(row=1, column=0, sticky="ew")
        btn_match.grid(row=1, column=1, sticky="ew")
        btn_reset.grid(row=1, column=2, sticky="ew")
        self.notebook.grid(row=2, column=0, columnspan=3, sticky="nsew")
        btn_read.grid(row=3, column=0, columnspan=3, sticky="ew")

        # Configure grid weights to allow text area to expand
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Key binding
        self.root.bind('<Key>', self.on_key_press)

        # Run the GUI
        self.refresh()
        self.root.mainloop()

    def add_tab(self, title="NEW TAB"):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=title)

        text_area = scrolledtext.ScrolledText(tab, wrap=tk.WORD, width=75, height=18)
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

    def refresh(self):
        self.start_point=None
        self.update_text()
        self.clear()
        for tracklet in self.tracker.tracklets:
            tracklet.show(axes=self.ax, mean_state_color=InteractiveMOT.state_color)
        plt.draw()

    def reset(self):
        self.tracker = Tracker()
        self.refresh()

    def prediction_step(self):
        self.tracker.predict()
        self.refresh()
        self.detections = []

    def matching_step(self):
        self.tracker.associate_and_measurement_update(self.detections)
        self.refresh()
        self.detections = []

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

        # Refresh
        self.refresh()

    def update_text(self):

        N = len(self.tracker.tracklets) + 1
        for _ in range(N - len(self.notebook.tabs())):
            self.add_tab()

        for _ in range(len(self.notebook.tabs()) - N):
            self.remove_tab()

        for i, tracklet in enumerate(self.tracker.tracklets):
            self.notebook.tab(i, text=f"{tracklet.label}")

            state_str = np.array2string(tracklet.state, precision=1, floatmode='fixed', separator=',',
                              formatter={'float': lambda x: f'{x:8.1f}'})
            covariance_str = np.array2string(tracklet.covariance, precision=1, floatmode='fixed', separator=',',
                             formatter={'float': lambda x: f'{x:8.1f}'})

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

    def on_click(self, event):

        if event.inaxes != self.ax:
            return

        if event.button == 1:

            if self.start_point is None:
                self.start_point = (event.xdata, event.ydata)
                self.start_point_draw = self.ax.scatter(*self.start_point, s = 50, c="k")
                plt.draw()

            else:
                end_point = (event.xdata, event.ydata)

                bbox = BBox.from_two_corners(*self.start_point, *end_point)
                detection = Detection.from_bbox(bbox, 'Detection', 1.0)

                self.detections.append(detection)
                detection.show(axes=self.ax, show_text=False, color=InteractiveMOT.detection_color, alpha=0.8)

                self.start_point_draw.remove()
                self.start_point = None
                plt.draw()


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


# Function to filter out numpy arrays from dictionary
def filter_dict(d):
    filtered = {}
    for key, value in d.items():
        if isinstance(value, (int, float, str)):
            filtered[key] = value
        elif isinstance(value, dict):
            filtered[key] = filter_dict(value)
    return filtered

if __name__ == "__main__":
    InteractiveMOT()