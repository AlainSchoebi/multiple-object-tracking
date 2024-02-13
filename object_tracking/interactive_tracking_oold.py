# Typing
from __future__ import annotations
from typing import List, Any, NewType, Callable

# Numpy
import numpy as np
from numpy.typing import NDArray

# Tracking
from tracklet import Tracklet

# Utils
from utils.bbox import BBox, XYXYMode
from bbox_tracking import LabeledBBox, Detection

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


class InteractiveTrackerOOLD:
    def __init__(self, tracklet):

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

        self.ax = ax
        self.tracklet: Tracklet = tracklet
        self.start_point = None
        self.start_point_draw = None
        self.rectangle = None
        self.tracklet.show(axes=self.ax, num=50, show_text=False)
        #self.tracklet.mean_state_bbox().show(axes=self.ax, show_text=False, color=np.array([0.5,0.5,0.5]), alpha=0.8)
        self.tracklet.show_mean_state(axes=self.ax, show_text=False, color=np.array([0.5,0.5,0.5]), alpha=0.8)
        self.update_title()
        plt.draw()

        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)


    def clear(self):
        for patch in self.ax.patches:
            if isinstance(patch, plt.Rectangle):
                patch.remove()
            if isinstance(patch, FancyArrowPatch):
                patch.remove()
        for text in self.ax.texts:
            text.remove()

    def update_title(self):
        plt.rcParams['text.usetex'] = False
        self.ax.set_title(
            fr"$\dot x ={self.tracklet.state[Tracklet.VX]:.2f}$, " +
            fr"$\dot y ={self.tracklet.state[Tracklet.VY]:.2f}$, " +
            fr"$\dot w ={self.tracklet.state[Tracklet.VW]:.2f}$, " +
            fr"$\dot h ={self.tracklet.state[Tracklet.VH]:.2f}$"
        )

    def on_press(self, event):

        if event.inaxes != self.ax:
            return

        if event.button == 1:

            if self.start_point is None:
                self.start_point = (event.xdata, event.ydata)
                self.start_point_draw = self.ax.scatter(*self.start_point, s = 50, c="k")

                plt.draw()

            else:
                self.start_point_draw.remove()
                self.clear()

                end_point = (event.xdata, event.ydata)

                bbox = BBox.from_two_corners(*self.start_point, *end_point)
                detection = Detection.from_bbox(bbox, 'Detection', 1.0)
                detection.show(axes=self.ax)

                self.tracklet.update(detection)
                self.tracklet.show(axes=self.ax, num=50, show_text=False)
                self.tracklet.show_mean_state(axes=self.ax, show_text=False, color=np.array([0.5,0.5,0.5]), alpha=0.8)
                self.update_title()

                self.start_point=None

                plt.draw()

        elif event.button == 3:
            self.clear()
            self.tracklet.predict()
            self.tracklet.show(axes=self.ax, num=50, show_text=False)
            self.tracklet.show_mean_state(axes=self.ax, show_text=False, color=np.array([0.5,0.5,0.5]), alpha=0.8)
            self.update_title()
            plt.draw()
