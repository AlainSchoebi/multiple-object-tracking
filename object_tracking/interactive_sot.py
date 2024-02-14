# Typing
from __future__ import annotations

# Numpy
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Python
import tkinter as tk
from tkinter import scrolledtext
import ast

# Tracking
from tracklet import Tracklet

# Utils
from utils.bbox import BBox
from bbox_tracking import Detection

# Logging
from utils.loggers import get_logger
logger = get_logger(__name__)


class InteractiveSOT:

    default_tracklet = Tracklet.initiate_from_detection(Detection(40,40,20,20,"cat",0.9))

    detection_color = np.array([0.1,0.2,0.8])
    state_color = np.array([0.5,0.5,0.5])

    def __init__(self, tracklet = None):

        if tracklet is None:
            tracklet = InteractiveSOT.default_tracklet.copy()

        fig = plt.figure()
        ax = fig.add_subplot()

        # Axis labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.invert_yaxis()

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, fc=InteractiveSOT.state_color, ec=InteractiveSOT.state_color*0.7, lw=1),  # Blue rectangle for detections
            plt.Rectangle((0, 0), 1, 1, fc=InteractiveSOT.detection_color, ec=InteractiveSOT.detection_color*0.7, lw=1)
        ]

        # Add the legend with custom handles and labels
        ax.legend(legend_handles, ['Mean State', 'Detections'], loc='lower right')

        self.ax = ax
        self.tracklet: Tracklet = tracklet
        self.start_point = None
        self.start_point_draw = None
        self.rectangle = None
        self.tracklet.show(axes=self.ax, num=50, show_text=False)
        self.tracklet.show_mean_state(axes=self.ax, show_text=False, color=InteractiveSOT.state_color, alpha=0.8)
        plt.draw()

        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Interactive Tracking")

        # Create a canvas for the plot
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas_widget = self.canvas.get_tk_widget()

        # Create a scrolled text area
        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=75, height=18)

        button1 = tk.Button(self.root, text="Prediction Step (right click)", command=self.prediction_step)
        button2 = tk.Button(self.root, text="Reset Tracklet", command=self.reset_tracklet)
        button3 = tk.Button(self.root, text="Read Tracklet and Config from Text", command=self.read_tracklet)

        # Position the plot, text area, and buttons using the grid layout manager
        canvas_widget.grid(row=0, column=0, columnspan=3)
        self.text_area.grid(row=1, column=0, columnspan=3, sticky="nsew")
        button1.grid(row=2, column=0, sticky="ew")
        button2.grid(row=2, column=1, sticky="ew")
        button3.grid(row=2, column=2, sticky="ew")

        # Configure grid weights to allow text area to expand
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Key binding
        self.root.bind('<Key>', self.on_key_press)

        # Run the GUI
        self.update_text()
        self.root.mainloop()

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

    def reset_tracklet(self):
        self.tracklet = InteractiveSOT.default_tracklet.copy()
        self.refresh()

    def prediction_step(self):
        self.tracklet.predict()
        self.refresh()

    def read_tracklet(self):
        text = self.text_area.get("1.0", tk.END)
        lines = text.split("\n")
        state_list = lines[2].replace("[","").replace("]","").strip().split(",")
        state = np.array(state_list, dtype=float)

        covariance_lists = []
        for i in range(5,13):
            line = lines[i].replace("[","").replace("]","").strip().split(",")
            line = [float(x) for x in line if x != ""]
            covariance_lists.append(line)

        covariance = np.array(covariance_lists, dtype=float)

        self.tracklet.state = state
        self.tracklet.covariance = covariance

        for line in lines[15:]:
            list = line.replace(" - ", "").strip().split(":")
            if len(list) == 2:
                key, value = list
                Tracklet.set_config_arg(key, eval(value))

        self.refresh()

    def refresh(self):
        self.start_point=None
        self.update_text()
        self.clear()
        self.tracklet.show(axes=self.ax, num=50, show_text=False)
        self.tracklet.show_mean_state(axes=self.ax, show_text=False, color=InteractiveSOT.state_color, alpha=0.8)
        plt.draw()


    def update_text(self):
        state_str = np.array2string(self.tracklet.state, precision=1, floatmode='fixed', separator=',',
                                    formatter={'float': lambda x: f'{x:8.1f}'})
        covariance_str = np.array2string(self.tracklet.covariance, precision=1, floatmode='fixed', separator=',',
                             formatter={'float': lambda x: f'{x:8.1f}'})

        self.text_area.delete('1.0', tk.END)
        self.text_area.insert(tk.END, " |    x   |    y   |    w   |    h   |   vx   |   vy   |   vw   |   vh   |\n")
        self.text_area.insert(tk.END, "Mean:\n ")
        self.text_area.insert(tk.END, state_str)
        self.text_area.insert(tk.END, " \n\nCovariance matrix:\n")
        self.text_area.insert(tk.END, covariance_str)
        self.text_area.insert(tk.END, f"\n\nConfig:\n")
        for key, value in Tracklet.config.items():
            if type(value) in [float, int, str]:
                self.text_area.insert(tk.END, f" - {key}: {value}\n")

    def on_click(self, event):

        if event.inaxes != self.ax:
            return

        if event.button == 1:

            if self.start_point is None:
                self.start_point = (event.xdata, event.ydata)
                self.start_point_draw = self.ax.scatter(*self.start_point, s = 50, c="k")
                plt.draw()

            else:
                self.clear()

                end_point = (event.xdata, event.ydata)

                bbox = BBox.from_two_corners(*self.start_point, *end_point)
                detection = Detection.from_bbox(bbox, 'Detection', 1.0)

                self.tracklet.update(detection)

                self.refresh()
                detection.show(axes=self.ax, show_text=False, color=InteractiveSOT.detection_color, alpha=0.8)

        elif event.button == 3:
            self.prediction_step()


    def on_key_press(self, event):
        focused_widget = self.root.focus_get()
        if focused_widget == self.text_area:
            return

        if event.char == 'r':
            self.reset_tracklet()
        elif event.char == 'p':
            self.prediction_step()
        elif event.char == 't':
            self.read_tracklet()


if __name__ == "__main__":
    InteractiveSOT()