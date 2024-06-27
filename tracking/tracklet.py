# Typing
from __future__ import annotations
from typing import Dict, Optional

# Numpy
import numpy as np
from numpy.typing import NDArray

# Python
from collections import deque
import copy

# Matplotlib
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrow
import matplotlib.colors

# Utils
import tracking.utils.kalman_filter as kf
from tracking.utils.bbox import BBox
from tracking.utils.config import update_config_dict

# Tracking
from tracking.bbox_tracking import Detection
from tracking.bbox_tracking import LabeledBBox

# Logging
from tracking.utils.loggers import get_logger
logger = get_logger(__name__)

class Tracklet:
    """
    Tracklet with state (x, y, w, h, vx, vy, vw, vh).

    Attributes
    - state:          `NDArray(8,)` the state of the tracklet.
    - covariance:     `NDArray(8, 8)` the covariance matrix of the tracklet.
    - id:             `int` optional id of the tracklet.
    - history:        `deque` the history of the tracklet. It contains `True` if
                      the tracklet was associated to a detection at the
                      corresponding time step, and `False` otherwise.
    - last_detection: `Detection` last detection with which the tracklet was
                      matched. Equals `None` if the tracklet was not matched.
    - config:         `Dict` the configuration of the tracklet. See below.

    Configuration dictionary for the `Tracklet` class:
    - history_maxlen:       `int` the maximum length of the history.
    - kf_position_noise:    `float` used for computing the position noise for
                            the prediction step.
                            The actual variances used are:
                            (kf_position_noise*width)^2, and
                            (kf_position_noise*height)^2.
    - kf_velocity_noise:    `float` used for computing the velocity noise for
                            the prediction step.
                            The actual variances used are:
                            (kf_velocity_noise*width)^2, and
                            (kf_velocity_noise*height)^2.
    - kf_measurement_noise: `float` used for computing the measurement noise for
                            the measurement update.
                            The actual variances used are:
                            (kf_measurement_noise*width)^2, and
                            (kf_measurement_noise*height)^2.
    - kf_init_position_noise_factor: `float` factor used for computing the
                            initial position covariance when a tracklet is
                            initiated from a detection.
                            The actual variances used are:
                            (kf_init_position_noise_factor*
                               kf_position_noise*width)^2,
                            (kf_init_position_noise_factor*
                               kf_position_noise*height)^2.
    - kf_init_velocity_noise_factor: `float` factor used for computing the
                            initial velocity covariance when a tracklet is
                            initiated from a detection.
                            The actual variances used are:
                            (kf_init_velocity_noise_factor*
                               kf_velocity_noise*width)^2,
                            (kf_init_velocity_noise_factor*
                               kf_velocity_noise*height)^2.
    - confidence_reference: `float` in `[0, 1]` the reference confidence value
                            used for adjusting the noise in the Kalman Filter
                            depending on the confidence of the detection.
    - noise_reduction_factor_at_full_confidence: `float` in `[0, 1]` the factor
                            used for reducing the noise when the confidence of
                            the detection is equal to 1. A factor of 1 means no
                            noise reduction at all, so the confidence has no
                            effect.
    - epsilon_size:         `float` the smallest value used for the BBoxes size
                            to avoid numerical issues.
    - active_steps:         `int` the number of steps during which a tracklet
                            must have existed to be considered as active.
    - long_lost_steps:      `int`. If a tracklet has not been matched to any
                            detection for the last `long_lost_steps` steps, it
                            is considered as long lost.
    """

    # Indices
    X, Y, W, H, VX, VY, VW, VH = range(8)


    # Configuration
    default_config = {
        "history_maxlen": int(10),
        "kf_position_noise": float(0.05),
        "kf_velocity_noise": float(0.00625),
        "kf_measurement_noise": float(0.05),
        "kf_init_position_noise_factor": 2,
        "kf_init_velocity_noise_factor": 10,
        "confidence_reference": float(0.9), # in [0,1]
        "noise_reduction_factor_at_full_confidence": float(0.5), # in [0,1]
        "epsilon_size": float(1.0),
        "active_steps": int(3),
        "long_lost_steps": int(5),
        "visualization":  {
            "n_frames_for_velocity": int(5),
            "alpha": float(0.5),
            "only_borders": bool(False),
            "inactive_color": str("black"),
            "lost_color": str("darkred"),
            "tracked_color": str("darkgreen"),
            "velocity_color": str("red"),
            "size_velocity_color": str("blue")
        }
    }


    def __init__(self, config: Optional[Dict] = {},
                 id: Optional[int] = None,
                 last_detection: Optional[Detection] = None):
        """
        Default constructor of the `Tracklet` class.

        Inputs
        - config: `Dict` partial configuration of the tracklet.
        - id:     `int` optional id of the tracklet.
        """

        self.config = copy.deepcopy(Tracklet.default_config)
        self.set_partial_config(config)

        self.history = deque(maxlen=self.config["history_maxlen"])
        self.state = np.array([])
        self.covariance = np.array([])
        self.id = id
        self.last_detection = last_detection


    @staticmethod
    def initiate_from_detection(detection: Detection,
                                config: Optional[Dict] = {},
                                id: Optional[int] = None) -> Tracklet:
        """
        Initiate a new `Tracklet` from a `Detection`.
        The state is set to the center, width, height of the detection, and 0
        velocity. The covariance is set to the default initial covariance
        specified in the configuration.

        Inputs
        - detection: `Detection` the detection to initiate the tracklet from.
        - config:    `Dict` partial configuration of the tracklet.
        - id:        `int` optional id of the tracklet.

        Returns
        - tracklet: `Tracklet` the initiated tracklet.
        """

        # New tracklet
        tracklet = Tracklet(config=config, id=id)

        # State: center, w, h from detection, and 0 velocity
        tracklet.state = np.array([
            *detection.center_wh_tuple(), 0, 0, 0, 0
        ])

        # Covariance
        tracklet.covariance = \
            tracklet._initial_covariance_from_detection(detection)

        tracklet._check_state()

        # History
        tracklet.history.append(True)

        # Last detection
        tracklet.last_detection = detection.copy()

        return tracklet


    def _initial_covariance_from_detection(self, detection: Detection) \
          -> NDArray:
        w, h = detection.w, detection.h
        init_pos_factor = self.config["kf_init_position_noise_factor"]
        init_vel_factor = self.config["kf_init_velocity_noise_factor"]

        std = [
            init_pos_factor * self.config["kf_position_noise"] * w,
            init_pos_factor * self.config["kf_position_noise"] * h,
            init_pos_factor * self.config["kf_position_noise"] * w,
            init_pos_factor * self.config["kf_position_noise"] * h,
            init_vel_factor * self.config["kf_velocity_noise"] * w,
            init_vel_factor * self.config["kf_velocity_noise"] * h,
            init_vel_factor * self.config["kf_velocity_noise"] * w,
            init_vel_factor * self.config["kf_velocity_noise"] * h
        ]

        cov = np.diag(np.square(std))
        cov = self._ajust_covariance_to_confidence(cov, detection.confidence)
        return cov


    def _ajust_covariance_to_confidence(self, sigma: NDArray,
                                        confidence: float) -> NDArray:

          confidence_ref = self.config["confidence_reference"]
          k = self.config["noise_reduction_factor_at_full_confidence"]
          r = (confidence - confidence_ref) / (1 - confidence_ref) * (1 - k)
          return sigma - r * sigma


    def set_partial_config(self, config: Dict):
        """
        Set partial configuration of the `Tracklet`.

        Inputs
        - config: `Dict` partial configuration
        """
        update_config_dict(self.config, config, Tracklet.default_config)


    def copy(self) -> Tracklet:
        """
        Return a deep copy of this `Tracklet`. The config is also copied.
        """
        tracklet = Tracklet()
        tracklet.history = deque(list(self.history),
                                 maxlen=self.config["history_maxlen"])
        tracklet.state = np.copy(self.state)
        tracklet.covariance = np.copy(self.covariance)
        tracklet.id = copy.deepcopy(self.id)
        tracklet.config = copy.deepcopy(self.config)
        return tracklet


    def bbox(self) -> BBox:
        """
        Return a `BBox` reprensenting the state of the tracklet.
        """
        return BBox.from_center_wh(*self.state[:4])


    def labeled_bbox(self) -> LabeledBBox:
        """
        Return a `LabeledBBox` reprensenting the state of the tracklet.
        """
        return LabeledBBox.from_bbox(self.bbox(), self.id)


    def _check_state(self):
        """
        Check the state of the tracklet. If the width or height is smaller than
        `epsilon_size`, it is set to `epsilon_size`.
        """
        if self.state[Tracklet.W] < self.config["epsilon_size"]:
            self.state[Tracklet.W] = self.config["epsilon_size"]
            logger.warn("KF: Small width detected. Set to `epsilon_size`.")
        if self.state[Tracklet.H] < self.config["epsilon_size"]:
            self.state[Tracklet.H] = self.config["epsilon_size"]
            logger.warn("KF: Small height detected. Set to `epsilon_size`.")


    def predict(self):
        """
        Predict the state and covariance of the tracklet at the next time step
        using the Kalman Filter.
        """
        # Access values
        _, _, w, h = self.bbox().xywh_tuple()

        A = np.kron(np.array([[1, 1],
                              [0, 1]]), np.eye(4))
        b = np.zeros(8)

        # Noise standard deviations
        position_x_noise_std = self.config["kf_position_noise"] * w
        position_y_noise_std = self.config["kf_position_noise"] * h
        velocity_x_noise_std = self.config["kf_velocity_noise"] * w
        velocity_y_noise_std = self.config["kf_velocity_noise"] * h

        # Covariance matrix
        Q = np.diag([position_x_noise_std**2, position_y_noise_std**2,  # x, y
                     position_x_noise_std**2, position_y_noise_std**2,  # w, h
                     velocity_x_noise_std**2, velocity_y_noise_std**2,  # vx, vy
                     velocity_x_noise_std**2, velocity_y_noise_std**2]) # vw, vh

        # Prediction step
        x_p, P_p = kf.prior_update(
            self.state, self.covariance, A, b, Q
        )

        # Update tracklet
        self.state, self.covariance = x_p, P_p
        self._check_state()


    def update(self, detection: Detection | None):
        """
        Update the state and covariance of the tracklet with a detection. It
        also updates the history and the last detection of the tracklet.

        Note: the detection can also be `None`, in which case only the history
              and the last detection of the tracklet are being updated.

        Inputs
        - detection: `Detection` the detection to update the tracklet with.
        """

        if detection is not None:
            w, h = self.bbox().w, self.bbox().h
            k_m = self.config["kf_measurement_noise"]

            z = detection.center_wh_array()
            H = np.c_[np.eye(4), np.zeros((4,4))]
            R = np.diag([(k_m * w)**2, (k_m * h)**2,  # x, y
                         (k_m * w)**2, (k_m * h)**2]) # w, h
            R = self._ajust_covariance_to_confidence(R, detection.confidence)

            # Assert that covariance is positive definite
            if np.linalg.det(self.covariance) < 1e-9:
                U, S, V = np.linalg.svd(self.covariance)
                S = np.maximum(S, 1e-1)
                self.covariance = U @ np.diag(S) @ V
                logger.warn("KF: Prior covariance matrix has too small " +
                            "eigenvalues to be properly invertible. Corrected.")

            # Measurement step
            x_m, P_m = kf.measurement_update(
                self.state, self.covariance, z, H, R
            )

            # Update tracklet
            self.state, self.covariance = x_m, P_m
            self._check_state()

        # History and last detection
        if detection is None:
            self.history.append(False)
            self.last_detection = None
        else:
            self.history.append(True)
            self.last_detection = detection.copy()


    # Status
    def is_active(self) -> bool:
        """
        Determine whether the tracklet is active or not. Active means that the
        tracklet existed for at least `active_steps` steps already.
        """
        return len(self.history) >= self.config["active_steps"]

    def is_tracked(self) -> bool:
        """
        Determine whether the tracklet is tracked or not. Tracked means that the
        tracklet has been associated to a detection in the last tracking step.
        """
        return self.history[-1]

    def is_lost(self) -> bool:
        """
        Determine whether the tracklet is lost or not. Lost is the opposite of
        tracked.
        """
        return not self.is_tracked()

    def is_long_lost(self) -> bool:
        """
        Determine whether the tracklet is long lost or not. Long lost means that
        the tracklet has been lost for the last `long_lost_steps` steps.
        """
        if len(self.history) < self.config["long_lost_steps"]:
            return False
        return np.all(~np.array(self.history)[-self.config["long_lost_steps"]:])


    # Visualization
    def show(self, num: Optional[int] = 50, **args) -> Axes:

        if not self.is_active():
            color = self.config["visualization"]["inactive_color"]
        elif self.is_lost():
            color = self.config["visualization"]["lost_color"]
        else:
            color = self.config["visualization"]["tracked_color"]
        color = np.array(matplotlib.colors.to_rgb(color))

        args["show_text"] = False
        if not hasattr(args, "colors"):
            args["color"] = color
        axes = self._show_distribution(num, **args)

        args["axes"] = axes
        args["alpha"] = self.config["visualization"]["alpha"]
        axes = self._show_mean_state(**args)
        return axes

    def _show_mean_state(self, alpha: float, color: NDArray, **args) -> Axes:
        alpha = self.config["visualization"]["alpha"]
        bbox = self.bbox()
        ax = bbox.show(alpha=alpha, only_borders=True, color=color*0.8, **args)
        ax.text(bbox.x + 2/ax.figure.dpi, bbox.y + 2/ax.figure.dpi, self.id,
                ha='left', va='top', alpha=alpha, color="k", fontsize=15)
        ax.scatter(self.state[Tracklet.X], self.state[Tracklet.Y],
                   color="k", marker="+", alpha=alpha)
        self._plot_velocity(ax)
        return ax

    def _show_distribution(self, num: Optional[int] = 10, **args) -> Axes:
        bboxes = []
        discarded_trials = 0
        while len(bboxes) < num:
            sample = np.random.multivariate_normal(mean=self.state,
                                                   cov=self.covariance)
            try:
                bbox = BBox.from_center_wh(*sample[:4])
                bboxes.append(bbox)
            except ValueError:
                discarded_trials += 1

        if discarded_trials/num > 0.5:
            logger.debug(f"Tracklet samples: considerable number of samples " +
                         f"discarded. Generate {discarded_trials + num} " +
                         f"samples to obtain {num} valid samples.")


        alpha = self.config["visualization"]["alpha"]
        ax = BBox.visualize(bboxes, only_borders=self.config["visualization"]
                            ["only_borders"], alpha=alpha/num*1.5, **args)
        return ax


    def _plot_velocity(self, axes: Axes) -> None:
        x, y = self.state[Tracklet.X], self.state[Tracklet.Y]
        vx, vy = self.state[Tracklet.VX], self.state[Tracklet.VY]
        vw, vh = self.state[Tracklet.VW], self.state[Tracklet.VH]
        n_frames = self.config["visualization"]["n_frames_for_velocity"]

        # Position velocity arrow
        color = self.config["visualization"]["velocity_color"]
        alpha = self.config["visualization"]["alpha"]
        arrow = FancyArrow(x, y, vx * n_frames, vy * n_frames,
                           width=0.6, color=color, head_width=8, head_length=8,
                           alpha=alpha, length_includes_head=True)
        axes.add_patch(arrow)

        # Size velocity arrows
        color = self.config["visualization"]["size_velocity_color"]
        corners = self.bbox().corners()
        d = (-1, -1)
        for corner in corners:
            arrow = FancyArrow(
                corner[0], corner[1],
                vw * n_frames / 2 * d[0], vh * n_frames / 2 * d[1],
                width=0.6, color=color, length_includes_head=True,
                head_width=8, head_length=8, alpha=alpha)
            axes.add_patch(arrow)
            d = (d[1], -d[0])


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Return the string representation of the `Tracklet`.
        """
        history_txt = str(list(self.history))
        if len(self.history) > 3:
            history_txt = str(["..."] + list(self.history)[-3:]).replace("'","")

        return f"Tracklet(" + \
               f"id={self.id}, " + \
               f"x={self.state[Tracklet.X]:.2f}, " + \
               f"y={self.state[Tracklet.Y]:.2f}, " + \
               f"w={self.state[Tracklet.W]:.2f}, " + \
               f"h={self.state[Tracklet.H]:.2f}, " + \
               f"vx={self.state[Tracklet.VX]:.2f}, " + \
               f"vy={self.state[Tracklet.VY]:.2f}, " + \
               f"vw={self.state[Tracklet.VW]:.2f}, " + \
               f"vh={self.state[Tracklet.VH]:.2f}, " + \
               f"covariance=..., " + \
               f"history={history_txt})"