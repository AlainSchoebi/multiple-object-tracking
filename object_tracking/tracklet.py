# Typing
from __future__ import annotations
from typing import Dict, Optional, Any

# Numpy
import numpy as np
from numpy.typing import NDArray

# Python
from collections import deque

# Matplotlib
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrow

# Utils
import utils.kalman_filter as kf
from utils.bbox import BBox, XYXYMode

# Tracking
from bbox_tracking import Detection
from bbox_tracking import LabeledBBox

# Logging
from utils.loggers import get_logger
logger = get_logger(__name__)

def _initial_covariance() -> NDArray:
    # Covariance matrix
    cov = np.diag([1,1,1,1,1,1,1,1])

    X, Y, W, H, VX, VY, VW, VH = range(8)

    # Positive correlation
    cov[X, VX] = 0.4
    cov[VX, X] = 0.4

    cov[Y, VY] = 0.4
    cov[VY, Y] = 0.4

    cov[W, VW] = 0.4
    cov[VW, W] = 0.4

    cov[H, VH] = 0.4
    cov[VH, H] = 0.4
    return cov.astype(float)

class Tracklet:
    """
    Tracklet with state (x, y, w, h, vx, vy, vw, vh).

    Configuration dictionary for the `Tracklet` class:
    - history_maxlen:       `int` the maximum length of the history.
    - kf_position_noise:    `float` used for computing the position noise.
                            The actual variances used are:
                            (kf_position_noise*width)^2, and
                            (kf_position_noise*height)^2.
    - kf_velocity_noise:    `float` used for computing the velocity noise.
                            The actual variances used are:
                            (kf_velocity_noise*width)^2, and
                            (kf_velocity_noise*height)^2.
    - kf_measurement_noise: `float` used for computing the measurement noise.
                            The actual variances used are:
                            (kf_measurement_noise*width)^2, and
                            (kf_measurement_noise*height)^2.
    - default_covariance:   `NDArray(8, 8)` default covariance matrix when
                            initiating a new tracklet from a detection.
    - epsilon_size:         `float` the smallest value used for the BBoxes size
                            to avoid numerical issues.
    TODO
    """

    # Indices
    X, Y, W, H, VX, VY, VW, VH = range(8)


    # Configuration
    config = {
        "history_maxlen": 10,
        "kf_position_noise": 0.05,
        "kf_velocity_noise": 0.00625,
        "kf_measurement_noise": 0.05,
        "initial_covariance": _initial_covariance(),
        "epsilon_size": 1.0,
        "active_steps": 2,
        "long_lost_steps": 5,
    }


    @staticmethod
    def set_config(config: Dict):
        """
        Set the configuration for the `Tracklet` class.

        Inputs
        - config: `Dict` a configuration dictionary.
        """
        Tracklet.config = config


    @staticmethod
    def set_config_arg(arg: str, value):
        """
        Set a specific configuration argument for the `Tracklet` class.

        Inputs
        - arg:   `str` the name of the argument to set.
        - value: the value to set.
        """
        if not arg in Tracklet.config:
            logger.error(f"Argument '{arg}' not found in the tracklet " +
                         f"configuration.")
            raise ValueError(f"Argument '{arg}' not found in the tracklet " +
                             f"configuration.")

        if type(Tracklet.config[arg]) == float and type(value) == int:
            value = float(value)

        if not type(Tracklet.config[arg]) == type(value):
            logger.error(f"Argument '{arg}' has type " +
                         f"{type(Tracklet.config[arg])} but the value has " +
                         f"type {type(value)}.")
            raise ValueError(f"Argument '{arg}' has type " +
                             f"{type(Tracklet.config[arg])} but the value " +
                             f"has type {type(value)}.")

        Tracklet.config[arg] = value


    def __init__(self, label: Optional[Any] = None):
        """
        Default constructor of the `Tracklet` class.

        Inputs
        - label: `Any` optional label of the tracklet.
        """
        self.history = deque(maxlen=Tracklet.config["history_maxlen"])
        self.state = np.array([])
        self.covariance = np.array([])
        self.label = label


    def copy(self) -> Tracklet:
        """
        Return a deep copy of this `Tracklet`.
        """
        tracklet = Tracklet()
        tracklet.history = deque(list(self.history),
                                 maxlen=Tracklet.config["history_maxlen"])
        tracklet.state = np.copy(self.state)
        tracklet.covariance = np.copy(self.covariance)
        return tracklet


    @staticmethod
    def initiate_from_detection(detection: Detection,
                                label: Optional[Any] = None) -> Tracklet:
        """
        Initiate a new `Tracklet` from a `Detection`.
        The state is set to the center, width, height of the detection, and 0
        velocity. The covariance is set to the default initial covariance
        specified in the configuration. # TODO depending on detection score

        Inputs
        - detection: `Detection` the detection to initiate the tracklet from.
        - label: `Any` optional label of the tracklet.

        Returns
        - tracklet: `Tracklet` the initiated tracklet.
        """

        # New tracklet
        tracklet = Tracklet(label=label)

        # State: center, w, h from detection, and 0 velocity
        tracklet.state = np.array([
            *detection.center_wh_tuple(mode=XYXYMode.NORMAL), 0, 0, 0, 0
        ])

        # Covariance: default covariance for detection # TODO depending on detection score
        tracklet.covariance = Tracklet.config["initial_covariance"]

        # History
        tracklet.history.append(True)

        return tracklet

    def bbox(self) -> BBox:
        try:
            return BBox.from_center_wh(*self.state[:4])
        except:
            logger.critical("This should not happen TODO")
            raise ValueError("This sould not happenn!!!! TODO")

    def labeled_bbox(self) -> LabeledBBox:
        return LabeledBBox.from_bbox(self.bbox(), self.label)

    def _check_state(self):
        if self.state[Tracklet.W] < Tracklet.config["epsilon_size"]:
            self.state[Tracklet.W] = Tracklet.config["epsilon_size"]
            logger.warn("KF: Small width detected. Set to `epsilon_size`.")
        if self.state[Tracklet.H] < Tracklet.config["epsilon_size"]:
            self.state[Tracklet.H] = Tracklet.config["epsilon_size"]
            logger.warn("KF: Small height detected. Set to `epsilon_size`.")


    def predict(self):
        """
        Predicts the state and covariance of the tracklet at the next time step
        using the Kalman Filter.
        """

        # Access values
        _, _, w, h = self.bbox().xywh_tuple()

        A = np.kron(np.array([[1, 1],
                              [0, 1]]), np.eye(4))
        b = np.zeros(8)

        # Noise standard deviations
        position_x_noise_std = Tracklet.config["kf_position_noise"] * w
        position_y_noise_std = Tracklet.config["kf_position_noise"] * h
        velocity_x_noise_std = Tracklet.config["kf_velocity_noise"] * w
        velocity_y_noise_std = Tracklet.config["kf_velocity_noise"] * h

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
        Updates the state and covariance of the tracklet with a detection. It
        also updates the history of the tracklet.

        Inputs
        - detection: `Detection` the detection to update the tracklet with.
        """

        # TODO capture detection uncertainty in H !!
        if detection is not None:
            w, h = self.bbox().w, self.bbox().h
            k_m = Tracklet.config["kf_measurement_noise"]

            z = detection.center_wh_array()
            H = np.c_[np.eye(4), np.zeros((4,4))]
            R = np.diag([(k_m * w)**2, (k_m * h)**2,  # x, y
                         (k_m * w)**2, (k_m * h)**2]) # w, h

            # Measurement step
            x_m, P_m = kf.measurement_update(
                self.state, self.covariance, z, H, R
            )

            # Update tracklet
            self.state, self.covariance = x_m, P_m

        # History
        if detection is None:
            self.history.append(False)
        else:
            self.history.append(True)


    # Properties
    def is_active(self) -> bool:
        """
        Determine whether the tracklet is active or not. Active means that the
        tracklet existed for at least `active_steps` steps already.
        """
        return len(self.history) >= Tracklet.config["active_steps"]

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
        if len(self.history) < Tracklet.config["long_lost_steps"]:
            return False
        return np.all(~np.array(self.history)[-Tracklet.config["long_lost_steps"]:])

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
        ax.scatter(self.state[Tracklet.X], self.state[Tracklet.Y], c='k', marker="+", alpha=alpha)
        Tracklet._plot_velocity(self.state, ax)
        return ax

    def show_distribution(self, num: Optional[int] = 10, **args) -> Axes:
        bboxes = []
        discarded_trials = 0
        while len(bboxes) < num:

            # Draw sample
            sample = np.random.multivariate_normal(mean=self.state,
                                                   cov=self.covariance)
            try:
                bbox = BBox.from_center_wh(*sample[:4])
            except ValueError:
                discarded_trials += 1
                continue

            bboxes.append(bbox)

        if discarded_trials/num > 0.5:
            logger.warn(f"Tracklet samples: considerable number of samples " +
                        f"iscarded. Generate {discarded_trials + num} " +
                        f"samples to obtain {num} valid samples.")

        ax = BBox.visualize(bboxes, **args)
        return ax


    @staticmethod
    def _plot_velocity(state: NDArray, axes: Axes):
        x, y = state[Tracklet.X], state[Tracklet.Y]
        vx, vy = state[Tracklet.VX], state[Tracklet.VY]
        vw, vh = state[Tracklet.VW], state[Tracklet.VH]
        dt = 1

        # Velocity arrow
        arrow = FancyArrow(x, y, vx * dt, vy * dt, width=0.1, color='red',
                           length_includes_head=True, head_width=2, head_length=2, alpha=0.5)
        axes.add_patch(arrow)

        # Size velocity arrows
        corners = BBox.from_center_wh(*state[:4]).corners()
        d = (-1, -1 )
        for corner in corners:
            arrow = FancyArrow(
                corner[0], corner[1], vw * dt / 2 * d[0], vh * dt / 2 * d[1],
                width=0.1, color='k', length_includes_head=True,
                head_width=2, head_length=2, alpha=0.5)
            axes.add_patch(arrow)
            d = (d[1], -d[0])


    def __str__(self):
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