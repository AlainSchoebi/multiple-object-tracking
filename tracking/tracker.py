# Typing
from __future__ import annotations
from typing import List, NewType, Callable, Tuple, Dict, Optional

# Numpy
import numpy as np
from numpy.typing import NDArray

# Python
import copy

# Librairies
import lap

# Matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.legend_handler import HandlerPatch
    from matplotlib.patches import FancyArrow, Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Utils
from tracking.utils.config import update_config_dict

# Tracking
from tracking.bbox_tracking import LabeledBBox, Detection
from tracking.tracklet import Tracklet
import tracking.metrics as metrics

Matches = NewType("Matches", List[Tuple[Tracklet, Detection]])

class Tracker:
    """
    Tracker class used for tracking BBoxes using a Kalman Filter.
    """

    default_config = {
        "iou_threshold": {
            "association_1": float(0.4),
            "association_2": float(0.3)
        },
        "image_size": {
            "width": 500,
            "height": 100
        },
        "tracklet_config": Tracklet.default_config
    }


    def __init__(self, config: Optional[Dict] = {}):
        """
        Default constructor of the `Tracker`.
        """
        self.tracklets: List[Tracklet] = []
        self.ids_count = 0

        self.config = copy.deepcopy(Tracker.default_config)
        self.set_partial_config(config)


    def set_partial_config(self, config: Dict):
        """
        Set partial configuration of the `Tracker`.

        Inputs
        - config: `Dict` partial configuration
        """
        update_config_dict(self.config, config, Tracker.default_config)
        self._propagate_config_to_tracklets()


    def _propagate_config_to_tracklets(self):
        """
        Propagate the configuration to the tracklets.
        """
        for tracklet in self.tracklets:
            tracklet.set_partial_config(self.config["tracklet_config"])


    def predict(self) -> None:
        """
        Predict the tracklets through the prior update of the Kalman Filter.
        """
        for tracklet in self.tracklets:
            tracklet.predict()


    def associate_and_measurement_update(self, detections: List[Detection]) \
                                         -> None:
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
            tracklet = Tracklet.initiate_from_detection(
                           detection, config=self.config["tracklet_config"],
                           label=self.ids_count
                       )
            self.ids_count += 1
            self.tracklets.append(tracklet)


    def update(self, detections: List[Detection]) -> None:
        """
        Update the tracker with the new detections. This process includes the
        prediction, association and measurement update steps.
        """
        # Prediction step
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


    if MATPLOTLIB_AVAILABLE:
        def show(self, axes: Optional[Axes] = None,
                       savefig: Optional[str] = None,
                       show: Optional[bool] = True) -> Axes:
            """
            Show the current state of the tracker.
            """

            # Save figure
            if savefig is not None:
                axes = None
                show = False

            # Interactive mode
            if show is False:
                plt.ioff()

            # No axes provided
            if axes is None:
                # Create figure
                fig = plt.figure()
                ax: Axes = fig.add_subplot()

                # Title
                ax.set_title(f"Tracker visualization with " +
                             f"{len(self.tracklets)} tracklet" +
                             f"{'s' if len(self.tracklets) > 1 else ''}")

                # Axis labels
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                ax.set_xlim(0, self.config["image_size"]["width"])
                ax.set_ylim(0, self.config["image_size"]["height"])
                ax.set_aspect('equal')
                ax.invert_yaxis()

                # Legend
                def make_legend_arrow(legend, orig_handle, xdescent, ydescent,
                                      width, height, fontsize) -> FancyArrow:
                    return FancyArrow(0, 0.5 * height, width, 0,
                                      length_includes_head=True,
                                      head_width=0.75*height)

                legend_handles = [
                    Rectangle((0, 0), 1, 1, fc=np.full(3, 0.5),
                              ec=0.7 * np.full(3, 0.5), lw=1),
                    plt.scatter([], [], marker='+', color='k'),
                    FancyArrow(0,0,1,1, color='r'),
                    FancyArrow(0,0,1,1, color='k'),
                ]
                labels = ['Mean State', 'Center', 'Center Velocity',
                          'Size Velocity']
                handler_map = {
                    FancyArrow: HandlerPatch(patch_func=make_legend_arrow),
                }

                ax.legend(legend_handles, labels, loc='lower right',
                          handler_map=handler_map)

            # Axes provided
            else:
                ax = axes

            # Show the tracklets
            for tracklet in self.tracklets:
                tracklet.show(axes=ax)

            # Show
            if show:
                plt.ion()
                plt.show()

            if savefig:
                fig.savefig(savefig)

            return ax