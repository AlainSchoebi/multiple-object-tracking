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
    import matplotlib.colors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Utils
from tracking.utils.bbox import BBox
from tracking.utils.config import update_config_dict

# Tracking
from tracking.bbox_tracking import LabeledBBox, Detection
from tracking.tracklet import Tracklet
import tracking.metrics as metrics

Matches = NewType("Matches", List[Tuple[Tracklet, Detection]])

class Tracker:
    """
    Tracker class used for tracking BBoxes with a Kalman Filter.

    Configuration dictionnary for the `Tracker`class:
    - matching:
        - low_confidence:    `float` the confidence threshold defining whether a
                             `Detection` has a low or mid confidence.
        - high_confidence:   `float` the confidence threshold defining whether a
                             `Detection` has a mid or high confidence.
        - association_1_iou: `float` the Intersection-over-Union (IoU) threshold
                             used during the first association process, that is
                             the association between all the high confidence
                             detections and the active tracklets.
        - association_2_iou: `float` the Intersection-over-Union (IoU) threshold
                             used during the second association process, that is
                             the association the remaining high confidence
                             detections and the inactive tracklets.
        - association_3_iou: `float` the Intersection-over-Union (IoU) threshold
                             used during the third association process, that is
                             the association the mid confidence detections and
                             the remaining active tracklets.

    - tracklet_config: `Dict` configuration dictionnary of all the `Tracklet`s
                       belonging to this `Tracker`.
    """

    # Configuration
    default_config = {
        "matching": {
            "low_confidence": float(0.3),
            "high_confidence": float(0.7),
            "association_1_iou": float(0.3),
            "association_2_iou": float(0.3),
            "association_3_iou": float(0.3)
        },
        "image_size": {
               "width": 800,
               "height": 500
        },
        "tracklet_config": Tracklet.default_config,
        "visualization": {
            "detection_color": {
                "low_confidence": str("gray"),
                "mid_confidence": str("orangered"),
                "high_confidence": str("orange")
            },
            "show_last_detections": bool(True)
        },
    }


    def __init__(self, config: Optional[Dict] = {}):
        """
        Default constructor of the `Tracker`.
        """
        self.tracklets: List[Tracklet] = []
        self._last_detections = []
        self.id_count = 0

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


    def _classify_detections_by_confidence(self, detections: List[Detection]) \
        -> Tuple[List[Detection], List[Detection], List[Detection]]:
        """
        Split the detections based on their confidence into three lists using
        the thresholds defined in the configuration.

        Inputs
        - detections: list of `Detection`

        Returns
        - low_detections:  list of `Detection` with low confidence
        - mid_detections:  list of `Detection` with mid confidence
        - high_detections: list of `Detection` with high confidence
        """

        low_detections, mid_detections, high_detections = [], [], []
        for detection in detections:
            if detection.confidence < self.config["matching"] \
                                                 ["low_confidence"]:
                low_detections.append(detection)
            elif detection.confidence < self.config["matching"] \
                                                   ["high_confidence"]:
                mid_detections.append(detection)
            else:
                high_detections.append(detection)

        return low_detections, mid_detections, high_detections


    def _save_last_detections(self, last_detections: List[Detection]) -> None:
        """
        Save last detections for visualization.

        Inputs
        - last_detections: list of `Detection` to save
        """

        self._last_detections = [d.copy() for d in last_detections]

    def associate_and_measurement_update(self, detections: List[Detection]) \
        -> None:
        """
        Associate the tracklets with the detections and update the tracklets
        through the measurement update of the Kalman Filter.

        Inputs
        - detections: list of `Detection`
        """

        self._save_last_detections(detections)

        # Split detections based on their confidence
        low_detections, mid_detections, high_detections = \
            self._classify_detections_by_confidence(detections)

        # Split tracklets into active and inactive tracklets
        active_tracklets = \
           [tracklet for tracklet in self.tracklets if tracklet.is_active()]
        inactive_tracklets = \
           [tracklet for tracklet in self.tracklets if not tracklet.is_active()]

        # Association 1
        # active tracklets <-> high confidence detections
        matches, unmatched_active_tracklets, unmatched_high_detections = \
            Tracker.associate(active_tracklets, high_detections, metrics.iou,
                              self.config["matching"]["association_1_iou"])

        # Update the matched tracklets with detections (i.e. measurement update)
        for tracklet, detection in matches:
            tracklet.update(detection=detection)

        # Association 2
        # inactive tracklets <-> remaining high confidence detections
        matches, unmatched_inactive_tracklets, unmatched_high_detections = \
            Tracker.associate(inactive_tracklets, unmatched_high_detections,
                              metrics.iou,
                              self.config["matching"]["association_2_iou"])

        # Update the matched tracklets with detections (i.e. measurement update)
        for tracklet, detection in matches:
            tracklet.update(detection=detection)

        # Remove inactive unmatched tracklets
        for tracklet in unmatched_inactive_tracklets:
            self.tracklets.remove(tracklet)

        # Initiate new tracklets from unmatched detections
        for detection in unmatched_high_detections:
            tracklet = Tracklet.initiate_from_detection(
                           detection, config=self.config["tracklet_config"],
                           id=self.id_count
                       )
            self.id_count += 1
            self.tracklets.append(tracklet)

        # Association 3
        # remaining active tracklets <-> mid confidence detections
        matches, unmatched_active_tracklets, _unmatched_mid_detections = \
            Tracker.associate(unmatched_active_tracklets, mid_detections,
                              metrics.iou,
                              self.config["matching"]["association_3_iou"])

        # Update the matched tracklets with detections (i.e. measurement update)
        for tracklet, detection in matches:
            tracklet.update(detection=detection)

        # Update the unmatched active tracklets
        for tracklet in unmatched_active_tracklets:
            tracklet.update(detection=None)

        # Remove long lost tracklets
        for tracklet in self.tracklets.copy():
            if tracklet.is_long_lost():
                self.tracklets.remove(tracklet)


    def update(self, detections: List[Detection]) -> None:
        """
        Update the tracker with the new detections. This process includes the
        prediction, association and measurement update steps.

        Inputs
        - detections: list of `Detection`
        """
        # Prediction step
        self.predict()

        # Association and measurement update steps
        self.associate_and_measurement_update(detections)


    def measurement_update_for_given_association(
        self, detections: List[Detection],
        corresponding_tracklets: List[int] | List[Tracklet]
        ) -> None:
        """
        Update the tracker with detection-tracklet pairs. That is, the
        association is provided as input and only the measurement update is
        performed. Note that the prediction step is not performed.

        Note: not all the current tracklets of the tracker necessarily need to
              be updated with a provided detection. The remaining tracklets will
              be updated with a `None` detection.

        Inputs
        - detections:               list of `Detection`
        - correspondings_tracklets: list of `int` representing the ID of each
                                    corresponding tracklet, or list of
                                    `Tracklet` for a reference to each
                                    corresponding tracklet
        """

        self._save_last_detections(detections)

        # Assert valid inputs
        if not len(detections) == len(corresponding_tracklets):
            raise ValueError(f"The number of detections must be equal to the " +
                             f"number of corresponding tracklets.")

        if len(detections) == 0:
            return

        if len(corresponding_tracklets) != len(set(corresponding_tracklets)):
            raise ValueError(f"The corresponding tracklets list cannot " +
                             f"the same tracklet multiple times.")

        def do_tracklets_correspond(tracklet: Tracklet,
                                    tracklet_or_id: Tracklet | int):
            return tracklet is tracklet_or_id or tracklet.id == tracklet_or_id

        # Reorder the detection-tracklet pairs with the tracklets of the tracker
        pairs: Matches = []
        count = 0
        for tracklet in self.tracklets:

            # Find corresponding detection
            detection = None
            for tracklet_i, detection_i, in zip(corresponding_tracklets,
                                                detections):
                if do_tracklets_correspond(tracklet, tracklet_i):
                    detection = detection_i
                    count += 1
                    break

            # Append the tracklet-detection pair
            pairs.append((tracklet, detection))

        if not count == len(corresponding_tracklets):
            raise ValueError(f"Not all the corresponding_tracklets seem to " +
                             f"exist in the tracker.")

        # Measurement update for all the tracklets
        # Note: some tracklest may have a `None` detection
        for tracklet, detection in pairs:
            tracklet.update(detection=detection)

        # Remove long lost tracklets
        for tracklet in self.tracklets.copy():
            if tracklet.is_long_lost():
                self.tracklets.remove(tracklet)

        # Remove inactive unmatched tracklets
        for tracklet in self.tracklets.copy():
            if not tracklet.is_active() and not tracklet.is_tracked():
                self.tracklets.remove(tracklet)


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


    def get_bboxes(self, only_active_tracklets: Optional[bool] = True,
                         crop_to_screen_size: Optional[bool] = True,
                         return_tracklets_copy: Optional[bool] = False) \
          -> List[LabeledBBox] | Tuple[List[LabeledBBox], List[Tracklet]]:
        """
        Return the labeled bounding boxes of the current tracklets (and a copy
        of the tracklets).

        Optional inputs
        - only_active_tracklets: `bool` if `True`, only the active tracklets
                                 are returned. Default is `True`.
        - crop_to_screen_size:   `bool` if `True`, the bounding boxes are
                                 cropped to the screen size. Default is `True`.
        - return_tracklets_copy: `bool` if `True`, the method additionally
                                 returns a copy of each `Tracklet` corresponding
                                 to each bounding box. Default is `False`.

        Note: these bounding boxes are different from the last detections.
        """
        bboxes = [tracklet.labeled_bbox() for tracklet in self.tracklets
                if not only_active_tracklets or tracklet.is_active()]
        tracklets = [tracklet.copy() for tracklet in self.tracklets
                if not only_active_tracklets or tracklet.is_active()]

        if not crop_to_screen_size:
            return (bboxes, tracklets) if return_tracklets_copy else bboxes

        screen_box = BBox(0, 0, self.config["image_size"]["width"],
                                self.config["image_size"]["height"])
        cropped_bboxes = []
        for bbox in bboxes:
            if BBox.intersect(bbox, screen_box):
                cropped_bbox = LabeledBBox.from_bbox(
                    BBox.intersection(bbox, screen_box, intersect_check=False),
                    label=bbox.label
                )
                cropped_bboxes.append(cropped_bbox)

        if return_tracklets_copy:
            return (cropped_bboxes, tracklets)
        else:
            return cropped_bboxes


    if MATPLOTLIB_AVAILABLE:
        def show(self, axes: Optional[Axes] = None,
                       savefig: Optional[str] = None,
                       dpi: Optional[int] = 100,
                       show: Optional[bool] = True,
                       image_overlay: Optional[NDArray] = None,
                       title: Optional[str] = None,
                       title_fontsize: Optional[float] = 14) -> Axes:
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
                if title is None:
                    title = f"Tracker visualization with " + \
                            f"{len(self.tracklets)} tracklet" + \
                            f"{'s' if len(self.tracklets) > 1 else ''}"
                ax.set_title(title, fontsize=title_fontsize)

                # Axis labels
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                image_size = self.config["image_size"]
                ax.set_xlim(0, image_size["width"])
                ax.set_ylim(0, image_size["height"])
                ax.set_aspect('equal')
                ax.invert_yaxis()

                # Legend
                def make_legend_arrow(legend, orig_handle, xdescent, ydescent,
                                      width, height, fontsize) -> FancyArrow:
                    return FancyArrow(0, 0.5 * height, width, 0,
                                      length_includes_head=True,
                                      head_width=0.75*height)

                tracklet_state_color = np.array(matplotlib.colors.to_rgb(
                    self.config["tracklet_config"]["visualization"] \
                                                  ["tracked_color"]
                ))
                detection_color = np.array(matplotlib.colors.to_rgb(
                    self.config["visualization"]["detection_color"]
                                                ["high_confidence"]
                ))

                legend_handles = [
                    Rectangle((0, 0), 1, 1, fc="none",
                              ec=detection_color, lw=1.5, linestyle="dashed"),
                    Rectangle((0, 0), 1, 1, fc=tracklet_state_color,
                              ec=0.7 * tracklet_state_color, lw=1),
                    plt.scatter([], [], marker='+', color='k'),
                    FancyArrow(0,0,1,1, color=self.config["tracklet_config"]
                               ["visualization"]["velocity_color"]),
                    FancyArrow(0,0,1,1, color=self.config["tracklet_config"]
                               ["visualization"]["size_velocity_color"]),
                ]
                labels = ['Detection', 'State', 'Center',
                          'Center Velocity', 'Size Velocity']
                handler_map = {
                    FancyArrow: HandlerPatch(patch_func=make_legend_arrow),
                }

                ax.legend(legend_handles, labels, loc='lower right',
                          handler_map=handler_map)

            # Axes provided
            else:
                ax = axes

            # Image overlay
            if image_overlay is not None:
                ax.imshow(image_overlay, alpha=0.5)

            # Show the tracklets
            for tracklet in self.tracklets:
                tracklet.show(axes=ax)

            # Show the last detections
            if self.config["visualization"]["show_last_detections"]:


                low_detections, mid_detections, high_detections = \
                    self._classify_detections_by_confidence(
                        self._last_detections
                    )
                for detection in sorted(self._last_detections,
                                        key=lambda d: d.confidence):
                    if detection in low_detections:
                        color = self.config["visualization"] \
                            ["detection_color"]["low_confidence"]
                    elif detection in mid_detections:
                        color = self.config["visualization"] \
                            ["detection_color"]["mid_confidence"]
                    elif detection in high_detections:
                        color = self.config["visualization"] \
                            ["detection_color"]["high_confidence"]
                    else:
                        raise ValueError("Error when classifying detections.")

                    color = np.array(matplotlib.colors.to_rgb(color))

                    detection.show(axes=ax, color=color, show_text=False,
                                   only_borders=True, linewidth=1.5,
                                   linestyle="dashed", alpha=1)

                    ax.text(*detection.corners()[3],
                            f"({detection.confidence:.2f})", fontsize=8,
                            color="white", ha='right', va='bottom',
                            bbox=dict(facecolor=color, linewidth=0,
                                      boxstyle="round, pad=-0.05")
                            )
            # Show
            if show:
                plt.ion()
                plt.show()

            # Save figure
            if savefig:
                fig.savefig(savefig, dpi=dpi)
                plt.close(fig)

            return ax