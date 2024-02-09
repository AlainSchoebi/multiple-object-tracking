# Typing
from __future__ import annotations
from typing import List, Any, NewType, Callable

# Numpy
import numpy as np
from numpy.typing import NDArray

# Object Tracking
from tracklet import Tracklet

# Utils
from utils.bbox import BBox, XYXYMode
from bbox_tracking import LabeledBBox, Detection

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
        Tracker.associate(self.tracklets, detections, Tracker.iou_metric, 0.7)

        pass


    @staticmethod
    def associate(tracklets: List[Tracklet], detections: List[Detection],
                  metric: Callable[[List[Tracklet], List[Detection]], NDArray],
                  metric_treshold: float) -> TODOType:

        # Compute metric
        values = metric(tracklets, detections)

        # Threshold ????

        # Associate

        if metric.lower() == "iou":

            tracklets_xyxy = []
            for tracklet in tracklets:
                xyxy = tracklet.state_bbox().xyxy_array(mode=XYXYMode.PIXEL)
                tracklets_xyxy.append(xyxy)

            detections_xyxy = []
            for detection in detections:
                xyxy = detection.xyxy_array(mode=XYXYMode.PIXEL)
                detections_xyxy.append(xyxy)

            tracklets_xyxy = np.array(tracklets_xyxy)
            detections_xyxy = np.array(detections_xyxy)

            # Compute IoU
            ious = cython_bbox.bbox_overlaps(tracklets_xyxy, detections_xyxy)
            print(f"IoUs: {ious}")

            r = cython_bbox.bbox_overlaps(
                a.xyxy_array(mode=XYXYMode.PIXEL)[None, :],
                b.xyxy_array(mode=XYXYMode.PIXEL)[None, :]
            )

            print(f"Iou: {r}")
            BBox.visualize([a,b])
            print(f"Iou: {r}")
            print(f"Iou: {r}")
            print(f"Iou: {r}")

        else:
            raise NotImplementedError(f"The metric '{metric}' is not " +
                                      f"supported.")

    @property
    def labeled_bboxes(self) -> List[LabeledBBox]:
        pass


    @staticmethod
    def iou_metric(tracklets: List[Tracklet], detections: List[Detection]) \
          -> NDArray:

        # Turn tracklets into XYXY array (with PIXEL model)
        tracklets_xyxy = []
        for tracklet in tracklets:
            xyxy = tracklet.state_bbox().xyxy_array(mode=XYXYMode.PIXEL)
            tracklets_xyxy.append(xyxy)

        # Turn detections into XYXY array (with PIXEL model)
        detections_xyxy = []
        for detection in detections:
            xyxy = detection.xyxy_array(mode=XYXYMode.PIXEL)
            detections_xyxy.append(xyxy)

        tracklets_xyxy = np.array(tracklets_xyxy)
        detections_xyxy = np.array(detections_xyxy)

        # Compute IoU
        return cython_bbox.bbox_overlaps(tracklets_xyxy, detections_xyxy)


# USAGE IDEA
config = {}
object_tracker = Tracker(config)

# At each step
detections = [Detection(2,3,1,1,"cat",0.9), Detection(1,2,1,2, "dog", 0.2)]
print(detections[0])
print(detections[0].corner_coordinates())
#BBox.visualize(detections)
object_tracker.update(detections)

#... = object_tracker.get_labeled_bboxes()

