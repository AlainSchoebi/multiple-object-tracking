# Typing
from __future__ import annotations
from typing import List, Any, NewType, Callable

# Numpy
import numpy as np
from numpy.typing import NDArray

# Librairies
import lap

# Matplotlib
import matplotlib.pyplot as plt

# Utils
from utils.bbox import BBox, XYXYMode
from bbox_tracking import LabeledBBox, Detection

# Tracking
from tracklet import Tracklet
import metrics
from interactive_tracking import InteractiveTracker

TODOType = NewType("TODOType", Any)

class Tracker:

    default_config = {
    }

    def __init__(self, config = None):

        self.config = config
        if self.config is None:
            self.config = Tracker.default_config.copy()

        self.tracklets = []

    def update(self, detections: List[Detection]) -> TODOType:

        tracklet: Tracklet
        detection: Detection

        # Classify or threshold detections

        # Predict the tracklets using a Kalman Filter
        for tracklet in self.tracklets:
            tracklet.predict()

        # Associate
        matches, unmatched_tracklets, unmatched_detections = \
              Tracker.associate(self.tracklets, detections, metrics.iou, 0.7)

        # TODO comments
        # UPDATE THE TRAKCLETS (I.E. MEASUREMENT UPDATED)
        for tracklet, detection in matches:
            tracklet.update(detection=detection)

        # UPDATE the not matched tracklets -< i.e. invisble
        for tracklet in unmatched_tracklets:
            tracklet.update(detection=None)

        # NEW TRACKLETS FOR UNMATCHED
        for detection in unmatched_detections:
            tracklet = Tracklet.initiate_from_detection(
                detection, self.config["tracklet_config"]
            )
            self.tracklets.append(tracklet)

        # DELETE OLD TRACKLETS
        for tracklet in self.tracklets:
            if tracklet.history == None: # TODO
                pass

        # TODO something about recent tracklets, i.e. no active yet???

        pass


    @staticmethod
    def associate(tracklets: List[Tracklet], detections: List[Detection],
                  metric: Callable[[List[Tracklet], List[Detection]], NDArray],
                  metric_threshold: float) -> TODOType:
        """"
        TODO

        metric is maximized

        Inputs
        - tracklets:        list of `Tracklet`
        - detections:       list of `Detection`
        - metric:           function computing a specific metric for a list of tracklets and a list of detections
        - metric_threshold: `float` threshold below which matches are rejected,
                            i.e. matches with a score up to and including the
                            threshold are considered

        Returns
        """

        # Assert no empty lists
        if len(tracklets) == 0 or len(detections) == 0:
           return [], tracklets, detections

        # Compute the negative of the metric, to act as a cost.
        costs = -metric(tracklets, detections)

        # Use LAPJP to solve the linear assignment problem that minimizes the
        # cost. (`extend_cost=True` allows using non-square cost matrices,
        # and `cost_limit` sets an upper bound for a pair to be matched.)
        cost_limit = -metric_threshold + 1e-10
        total_cost, tra_corr, det_corr = lap.lapjv(
            costs, extend_cost=True, cost_limit=cost_limit
        )

        # Build matches from the correspondences
        matches = []
        for idx_tracklet, idx_detection in enumerate(tra_corr):
            matches.append(tracklets[idx_tracklet], detections[idx_detection])

        # Build unmatched lists (`-1` indicates no match)
        unmatched_tracklets = [tracklets[i] for i in np.where(tra_corr == -1)]
        unmatched_detections = [detections[i] for i in np.where(det_corr == -1)]

        return matches, unmatched_tracklets, unmatched_detections


    def labeled_bboxes(self) -> List[LabeledBBox]:
        pass



InteractiveTracker()


detections = [Detection(2,3,50,50,"cat",0.9), Detection(1,2,1,2, "dog", 0.2)]
# Tracklet
t = Tracklet.initiate_from_detection(detections[0])

# USAGE IDEA
config = {}
object_tracker = Tracker(config)

# At each step
print(detections[0])
print(detections[0].corners())
#BBox.visualize(detections)
object_tracker.update(detections)

#... = object_tracker.get_labeled_bboxes()

# TODO less than 1 pixel issue?
a = BBox(3,4,0.5,0.6)
print(a.xyxy_array())

a = BBox()
b = 0.5 * a

iou = BBox.iou(a,b)
print(iou)


#        costs = np.array([[8,4,7],
#                         [5,2,3],
#                         [9,6,7],
#                         [9,4,8]])
#
#        ious = np.array([[8,4,7],
#                         [5,9,3],
#                         [9,6,7],
#                         [9,4,8]])/10
#        print(ious)
#        costs = -ious


#q = deque(maxlen=3)
#q.append(2)
#q.append(5)
#q.append(3)