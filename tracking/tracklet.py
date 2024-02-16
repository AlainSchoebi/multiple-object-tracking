# Typing
from __future__ import annotations
from typing import Dict, Optional, Any

# Numpy
import numpy as np
from numpy.typing import NDArray

# Python
from collections import deque
import copy

# Matplotlib
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrow

# Utils
import utils.kalman_filter as kf
from utils.bbox import BBox, XYXYMode
from utils.config import update_config_dict

# Tracking
from bbox_tracking import Detection
from bbox_tracking import LabeledBBox

# Logging
from utils.loggers import get_logger
logger = get_logger(__name__)

class Tracklet:
    """
    Tracklet with state (x, y, w, h, vx, vy, vw, vh).

    Attributes
    - state:      `NDArray(8,)` the state of the tracklet.
    - covariance: `NDArray(8, 8)` the covariance matrix of the tracklet.
    - label:      `Any` optional label of the tracklet.
    - history:    `deque` the history of the tracklet. It contains `True` if the
                  tracklet was associated to a detection at the corresponding
                  time step, and `False` otherwise.
    - config:     `Dict` the configuration of the tracklet. See below.

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
    }


    def __init__(self, config: Optional[Dict] = {},
                 label: Optional[Any] = None):
        """
        Default constructor of the `Tracklet` class.

        Inputs
        - config: `Dict` partial configuration of the tracklet.
        - label:  `Any` optional label of the tracklet.
        """

        self.config = copy.deepcopy(Tracklet.default_config)
        self.set_partial_config(config)

        self.history = deque(maxlen=self.config["history_maxlen"])
        self.state = np.array([])
        self.covariance = np.array([])
        self.label = label


    @staticmethod
    def initiate_from_detection(detection: Detection,
                                config: Optional[Dict] = {},
                                label: Optional[Any] = None) -> Tracklet:
        """
        Initiate a new `Tracklet` from a `Detection`.
        The state is set to the center, width, height of the detection, and 0
        velocity. The covariance is set to the default initial covariance
        specified in the configuration.

        Inputs
        - detection: `Detection` the detection to initiate the tracklet from.
        - config:    `Dict` partial configuration of the tracklet.
        - label:     `Any` optional label of the tracklet.

        Returns
        - tracklet: `Tracklet` the initiated tracklet.
        """

        # New tracklet
        tracklet = Tracklet(config=config, label=label)

        # State: center, w, h from detection, and 0 velocity
        tracklet.state = np.array([
            *detection.center_wh_tuple(mode=XYXYMode.NORMAL), 0, 0, 0, 0
        ])

        # Covariance
        tracklet.covariance = \
            tracklet._initial_covariance_from_detection(detection)

        tracklet._check_state()

        # History
        tracklet.history.append(True)

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
        tracklet.label = copy.deepcopy(self.label)
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
        return LabeledBBox.from_bbox(self.bbox(), self.label)


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


    def update(self, detection: Detection):
        """
        Update the state and covariance of the tracklet with a detection. It
        also updates the history of the tracklet.

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

        # History
        if detection is None:
            self.history.append(False)
        else:
            self.history.append(True)


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
    def show(self, num: Optional[int] = 50,
             mean_state_color: Optional[NDArray] = np.array([0.5, 0.5, 0.5]),
             **args) -> Axes:

        args["alpha"] = 0.2 if self.is_active() else 0.01
        args["show_text"] = False
        axes = self.show_distribution(num, **args)

        args["axes"] = axes
        args["alpha"] = 0.8 if self.is_active() else 0.1
        args["color"] = mean_state_color
        args["show_text"] = True
        axes = self.show_mean_state(**args)
        return axes

    def show_mean_state(self, alpha, **args) -> Axes:
        bbox = self.bbox()
        ax = bbox.show(alpha=alpha, **args)
        ax.text(bbox.x + 0.5, bbox.y + 0.5, self.label, ha='left', va='top',
                alpha=alpha, color="k", fontsize=15)
        ax.scatter(self.state[Tracklet.X], self.state[Tracklet.Y],
                   c='k', marker="+", alpha=alpha)
        self._plot_velocity(ax)
        return ax

    def show_distribution(self, num: Optional[int] = 10, **args) -> Axes:
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

        ax = BBox.visualize(bboxes, **args)
        return ax


    def _plot_velocity(self, axes: Axes) -> None:
        x, y = self.state[Tracklet.X], self.state[Tracklet.Y]
        vx, vy = self.state[Tracklet.VX], self.state[Tracklet.VY]
        vw, vh = self.state[Tracklet.VW], self.state[Tracklet.VH]
        dt = 1

        # Position velocity arrow
        arrow = FancyArrow(x, y, vx * dt, vy * dt, width=0.1, color='red',
                           head_width=2, head_length=2, alpha=0.5,
                           length_includes_head=True)
        axes.add_patch(arrow)

        # Size velocity arrows
        corners = self.bbox().corners()
        d = (-1, -1)
        for corner in corners:
            arrow = FancyArrow(
                corner[0], corner[1], vw * dt / 2 * d[0], vh * dt / 2 * d[1],
                width=0.1, color='k', length_includes_head=True,
                head_width=2, head_length=2, alpha=0.5)
            axes.add_patch(arrow)
            d = (d[1], -d[0])


    def __str__(self):
        """
        Return the string representation of the `Tracklet`.
        """
        return f"Tracklet(" + \
               f"x={self.state[Tracklet.X]:.2f}, " + \
               f"y={self.state[Tracklet.Y]:.2f}, " + \
               f"w={self.state[Tracklet.W]:.2f}, " + \
               f"h={self.state[Tracklet.H]:.2f}, " + \
               f"vx={self.state[Tracklet.VX]:.2f}, " + \
               f"vy={self.state[Tracklet.VY]:.2f}, " + \
               f"vw={self.state[Tracklet.VW]:.2f}, " + \
               f"vh={self.state[Tracklet.VH]:.2f}, " + \
               f"covariance=..., " + \
               f"history={list(self.history)})"