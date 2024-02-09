import numpy as np

from enum import Enum

from kalman_filter import KalmanFilter

from utils.xywh import XYWH

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

    # Static attributes
    dt = from_config
    A = np.kron(np.array([[1, dt],
                          [0, 1]]),
                np.eye(4))
    b = np.zeros(4)

    def __init__(self):
        self.history = []
        self.state = np.array([])
        self.covariance = np.array([])
#        self.id = ???

    pass

    def state_xywh(self):
        return XYWH(*self.state[:4])

    def predict(self):
        Q = ...
        x_p, P_p = KalmanFilter.predict(self.state, self.covariance,
                                        Tracklet.A, Tracklet.b, Q)
        self.state, self.covariance = x_p, P_p


