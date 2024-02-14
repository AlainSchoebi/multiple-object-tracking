# Typing
from __future__ import annotations
from typing import List, NewType, Callable, Tuple

# Numpy
import numpy as np
from numpy.typing import NDArray

# Librairies
import lap

# Utils
from bbox_tracking import LabeledBBox, Detection

# Tracking
from tracklet import Tracklet
import metrics

Matches = NewType("Matches", List[Tuple[Tracklet, Detection]])

class Tracker:

    default_config = {
        "iou_threshold": {
            "association_1": 0.4,
            "association_2": 0.3
        }
    }

    def __init__(self, config = None):

        self.config = config
        if self.config is None:
            self.config = Tracker.default_config.copy()

        self.tracklets: List[Tracklet] = []
        self.ids_count = 0


    def predict(self) -> None:
        """
        Predict the tracklets through the prior update of the Kalman Filter.
        """
        for tracklet in self.tracklets:
            tracklet.predict()


    def associate_and_measurement_update(self, detections: List[Detection]):
        """
        Associate the tracklets with the detections and update the tracklets
        through the measurement update of the Kalman Filter.
        """

        # Active and inactive tracklets
        active_tracklets = \
           [tracklet for tracklet in self.tracklets if tracklet.is_active()]
        inactive_tracklets = \
           [tracklet for tracklet in self.tracklets if not tracklet.is_active()]

        # Association 1 (active tracklets)
        matches, unmatched_active_tracklets, unmatched_detections = \
            Tracker.associate(active_tracklets, detections, metrics.iou,
                              self.config["iou_threshold"]["association_1"])

        # Update the matched tracklets with detections (i.e. measurement update)
        for tracklet, detection in matches:
            tracklet.update(detection=detection)

        # Update the not-matched tracklets
        for tracklet in unmatched_active_tracklets:
            tracklet.update(detection=None)

        # Association 2 (inactive tracklets)
        matches, unmatched_inactive_tracklets, unmatched_detections = \
            Tracker.associate(inactive_tracklets, unmatched_detections,
                              metrics.iou,
                              self.config["iou_threshold"]["association_2"])

        # Update the matched tracklets with detections (i.e. measurement update)
        for tracklet, detection in matches:
            tracklet.update(detection=detection)

        # Remove inactive unmatched tracklets
        for tracklet in unmatched_inactive_tracklets:
            self.tracklets.remove(tracklet)

        # Remove lost tracklets
        for i in range(len(self.tracklets)):
            if self.tracklets[i].is_long_lost():
               self.tracklets[i] = None

        # Remove None tracklets
        self.tracklets = [t for t in self.tracklets if t is not None]

        # Initiate new tracklets from unmatched detections
        for detection in unmatched_detections:
            tracklet = Tracklet.initiate_from_detection(detection,
                                                        label=self.ids_count)
            self.ids_count += 1
            self.tracklets.append(tracklet)


    def update(self, detections: List[Detection]):
        # Predict step
        self.predict()

        # Association and measurement update steps
        self.associate_and_measurement_update(detections)


    @staticmethod
    def associate(tracklets: List[Tracklet], detections: List[Detection],
                  metric: Callable[[List[Tracklet], List[Detection]], NDArray],
                  metric_threshold: float) -> \
                      Tuple[Matches, List[Tracklet], List[Detection]]:
        """
        Associate tracklets and detections using a specific metric and a
        threshold below which matches are rejected. The metric is being
        maximized.

        Inputs
        - tracklets:        list of `Tracklet`
        - detections:       list of `Detection`
        - metric:           function computing a similarity matrix for a list of
                            tracklets and a list of detections. The (i, j)-th
                            entry of the matrix should contain the similarity
                            between the i-th tracklet and the j-th detection.
        - metric_threshold: `float` threshold below which matches are rejected,
                            i.e. matches with a score up to and including the
                            threshold are considered

        Returns
        - matches:          list of tuples `(Tracklet, Detection)` representing
                            the matches
        - unmatched_tracklets:  list of `Tracklet` that were not matched
        - unmatched_detections: list of `Detection` that were not matched
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
        _total_cost, tra_corr, det_corr = lap.lapjv(
            costs, extend_cost=True, cost_limit=cost_limit
        )

        # Build matches from the correspondences
        matches = []
        for idx_tracklet, idx_detection in enumerate(tra_corr):
            if idx_detection == -1: continue
            match = (tracklets[idx_tracklet], detections[idx_detection])
            matches.append(match)

        # Build unmatched lists (`-1` indicates no match)
        unmatched_tracklets = \
            [tracklets[i] for i in np.where(tra_corr == -1)[0]]
        unmatched_detections = \
            [detections[i] for i in np.where(det_corr == -1)[0]]

        return matches, unmatched_tracklets, unmatched_detections


    def labeled_bboxes(self) -> List[LabeledBBox]:
        """
        Return the labeled bounding boxes of the current tracklets.
        Note: these bounding boxes are different from the detections.
        """
        return [tracklet.labeled_bbox() for tracklet in self.tracklets]