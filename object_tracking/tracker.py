# Typing
from __future__ import annotations
from typing import List, Any, NewType

# Numpy
import numpy as np

# Object Tracking
from tracklet import Tracklet

# Utils
from utils.xywh import XYWH
from xywh_tracking import LabeledXYWH, Detection

# Cython bbox
import cython_bbox

TODOType = NewType("TODOType", Any)

class Tracker:

    default_config = {"asdf": 7}

    def __init__(self, config = None):

        self.config = config
        if self.config is None:
            self.config = Tracker.default_config

        self.tracklets = []

    def update(self, detections: List[Detection]) -> TODOType:

        # Classify or threshold detections

        # Predict the tracklets using a Kalman Filter
        for tracklet in self.tracklets:
            tracklet.predict()

        #
        Tracker.associate(self.tracklets, detections, "iou", 0.7)

        pass

    @staticmethod
    def associate(tracklets: List[Tracklet], detections: List[Detection],
                  metric: str,
                  metric_treshold: float):

        if metric.lower() == "iou":

            tracklets_xyxy = []
            for tracklet in tracklets:
                tracklets_xyxy.append(tracklet.state_xywh().xyxy())

            detections_xyxy = []
            for detection in detections:
                detections_xyxy.append(detection.xyxy())

            # Compute IoU
            a = np.array([[1,2,3,3]], dtype=np.float64)
            b = np.array([[1,2,3,6]], dtype=np.float64)
            r = cython_bbox.bbox_overlaps(a, b)

        else:
            raise NotImplementedError(f"The metric '{metric}' is not " +
                                      f"supported.")

    @property
    def labeled_bboxes(self) -> List[LabeledXYWH]:
        pass



# USAGE IDEA
config = {}
object_tracker = Tracker(config)

# At each step
detections = [Detection(2,3,1,1,"cat",0.9), Detection(1,2,1,2, "dog", 0.2)]
from utils.xywh import XYWH
print(detections[0])
print(detections[0].corner_coordinates())
XYWH.visualize(detections)
object_tracker.update(detections)

#... = object_tracker.get_labeled_bboxes()

