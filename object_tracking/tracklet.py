# Typing
from __future__ import annotations
from typing import Dict, Optional

# Numpy
import numpy as np

# Utils
from utils.kalman_filter import KalmanFilter
from utils.bbox import BBox, XYXYMode

# Tracking
from bbox_tracking import Detection

# Python
from collections import deque
from enum import Enum
from math import sqrt

# Matplotlib
from matplotlib.axes import Axes

#class BaseTracklet???:
#    pass

class TrackletStatus(Enum):
    Tracked = 1
    GREEN = 2
    BLUE = 3

    #    self.status = None -> getstatus BUT FROM the Tracker!

from_config = 0

class Tracklet:
    """
    Tracklet with state (x, y, w, h, vx, vy, vw, vh)
    """

    # Default config
    default_config = {
        "history_maxlen": 10,
        "dt": 1, # [s] # TODO
        "kf_bbox_normalized_position_noise": 1, # ... TODO (p_w)
        "kf_bbox_normalized_velocity_noise": 1, # ... TODO (\dot p_w)
        "kf_normalized_measurement_noise": 0.5, # TODO
    }

    def __init__(self, config: Dict = None):
        """
        defualt init here"""

        self.config = config
        if self.config is None:
            self.config = Tracklet.default_config

        self.history = deque(maxlen=self.config["history_maxlen"])

        self.state = np.array([])
        self.covariance = np.array([])
#        self.id = ???

    pass

    def copy(self) -> Tracklet:
        """
        Return a deep copy of this `Tracklet`.
        """
        tracklet = Tracklet(self.config.copy())
        tracklet.history = deque(list(self.history),
                                 maxlen=self.config["history_maxlen"])
        tracklet.state = np.copy(self.state)
        tracklet.covariance = np.copy(self.covariance)
        return tracklet

    @staticmethod
    def initiate_from_detection(detection: Detection, config: Dict = None) \
          -> Tracklet:
        """
        TODO
        """

        # New tracklet
        tracklet = Tracklet(config)

        # State: center, w, h from detection, and 0 velocity
        tracklet.state = np.array([
            *detection.center_wh_tuple(mode=XYXYMode.NORMAL), 0, 0, 0, 0
        ])

        # Covariance: TODO
        tracklet.covariance = np.diag([1,1,0.01,0.01,8,8,8,8]) # TODO

        # History
        tracklet.history.append(True)

        return tracklet

    def mean_state_bbox(self):
        return BBox.from_center_wh(*self.state[:4])

    def predict(self):

        # Access values
        _, _, w, h = self.mean_state_bbox().xywh_tuple()
        k_p = self.config["kf_bbox_normalized_position_noise"]
        k_v = self.config["kf_bbox_normalized_velocity_noise"]
        dt = self.config["dt"]

        A = np.kron(np.array([[1, dt],
                              [0, 1]]), np.eye(4))
        b = np.zeros(8)

        # Noise standard deviations
        position_x_noise_std = k_p * w * sqrt(dt)
        position_y_noise_std = k_p * h * sqrt(dt)
        velocity_x_noise_std = k_v * w * sqrt(dt)
        velocity_y_noise_std = k_v * h * sqrt(dt)

        # Covariance matrix
        Q = np.diag([position_x_noise_std**2, position_y_noise_std**2,  # x, y
                     position_x_noise_std**2, position_y_noise_std**2,  # w, h
                     velocity_x_noise_std**2, velocity_y_noise_std**2,  # vx, vy
                     velocity_x_noise_std**2, velocity_y_noise_std**2]) # vw, vh

        # Prediction step
        x_p, P_p = KalmanFilter.prior_update(
            self.state, self.covariance, A, b, Q
        )

        # Update tracklet
        self.state, self.covariance = x_p, P_p

    def update(self, detection: Detection):

        # TODO capture detection uncertainty in H !!
        if detection is not None:
            w, h = self.mean_state_bbox().w, self.mean_state_bbox().h
            k_m = self.config["kf_normalized_measurement_noise"]

            z = detection.center_wh_array()
            H = np.c_[np.eye(4), np.zeros((4,4))]
            R = np.diag([(k_m * w)**2, (k_m * h)**2,  # x, y
                         (k_m * w)**2, (k_m * h)**2]) # w, h

            # Measurement step
            x_m, P_m = KalmanFilter.measurement_update(
                self.state, self.covariance, z, H, R
            )

            # Update tracklet
            self.state, self.covariance = x_m, P_m

        else:
            # No measurement update, only prediction like alwys :) #TODO
#            KalmanFilter.measurement_update() # with no detection but to add noise?????
            pass

        # History
        if detection is None:
            self.history.append(False)
        else:
            self.history.append(True)


    def sample(self):
        """
        TODO
        """
        np.random.multivariate_normal(mean=self.state, cov=self.covariance)
        pass

    def show(self, num: Optional[int] = 100, **args) -> Axes:

        bboxes = []
        while len(bboxes) < num:

            # Draw sample
            sample = np.random.multivariate_normal(mean=self.state,
                                                   cov=self.covariance)
            try:
                bbox = BBox.from_center_wh(*sample[:4])
            except ValueError:
                print("bad luck invalid bbox")
                continue

            bboxes.append(bbox)

        ax = BBox.visualize(bboxes, **args)
        return ax