# Typing
from typing import NewType, List, Tuple

# Numpy
import numpy as np
from numpy.typing import NDArray

# Types
State = NewType("State", NDArray)
Covariance = NewType("Covariance", NDArray)

class KalmanFilter:
   """
   General class for standard Kalman Filter.
   """

   @staticmethod
   def prior_update(x_m: State, P_m: Covariance,
                     A: NDArray, b: NDArray, Q: Covariance) \
                    -> Tuple[State, Covariance]:
       """
       Prior update of the Kalman Filter, also referred to as the prediction
       step.

       The dynamics are described by:
         x_k = A * x_{k-1} + b + v,
         where v is a zero-mean Gaussian RV with covariance Q.

       Inputs
       - x_m: `(n, 1)` or `(n,)` a posteriori estimate of the state,
              i.e. x_{k-1|k-1}.
       - P_m: `(n, n)` a posteriori covariance matrix of the state,
               i.e. P_{k-1|k-1}.
       - A:   `(n, n)` state transition matrix
       - b:   `(n, 1)` or `(n,)` bias vector
       - Q:   `(n, n)` process noise covariance matrix

       Outputs
       - x_p: `(n, 1)` or `(n,)` predicted state estimate at the next time step
              when applying the dynamics, i.e. x_{k|k-1}.
       - P_p: `(n, n)` covariance matrix of the state at the next time step, when
              applying the dynamics, i.e. P_{k|k-1}.
       """

       n = x_m.shape[0]
       assert A.ndim == 2 and x_m.ndim <= 2
       assert (x_m.shape == (n, 1) or x_m.shape == (n,)) and \
              P_m.shape == (n, n) and \
              A.shape == (n, n) and \
              b.shape == x_m.shape and \
              Q.shape == (n, n) and \
              "Incorrect input shapes."

       # Prior update
       x_p = A @ x_m + b
       P_p = A @ P_m @ A.T + Q

       return x_p, P_p

   @staticmethod
   def measurement_update(x_p: State, P_p: Covariance, z: NDArray,
                          H: NDArray, R: Covariance) \
                   -> Tuple[State, Covariance]:
       """
       Measurement update of the Kalman Filter, also referred to as the a
       posteriori update.

       The measurement equation is given by:
         z_k = H * x_k + w,
         where w is a zero-mean Gaussian RV with covariance R.

       Inputs
       - x_p: `(n, 1)` or `(n,)` prior estimate of the state, i.e. x_{k|k-1}.
       - P_p: `(n, n)` prior covariance matrix of the state, i.e. P_{k|k-1}.
       - z:   `(m, 1)` or `(m,)` observed measurement, i.e. z_k
       - H:   `(m, n)` observation matrix
       - R:   `(m, m)` measurement noise covariance matrix, symmetric definite
              postive

       Outputs
       - x_m: `(n, 1)` or `(n,)` a posteriori state estimate after employing
              the measurement z_k, i.e. x_{k|k}.
       - P_m: `(n, n)` a posteriori covariance matrix of the state after
              employing the measurement z_k, i.e. x_{k|k}.
       """

       n, m = x_p.shape[0], z.shape[0]
       assert A.ndim == 2 and x_p.ndim <= 2 and z.ndim <= 2
       assert (x_p.shape == (n, 1) or x_p.shape == (n,)) and \
              P_p.shape == (n, n) and \
              (z.shape == (m, 1) or z.shape == (m,)) and \
              z.ndim == x_p.ndim and \
              H.shape == (m, n) and \
              R.shape == (m, m) and \
              "Incorrect input shapes."

       if np.linalg.det(P_p) < 1e-10:
           raise ValueError("The provided prior covariance matrix P_p is not " +
                            "is not invertible.")

       try:
           R_inv = np.linalg.inv(R)
       except:
           raise ValueError("Provided measurement noise covariance matrix R " +
                            "is not invertible.")

       P_m_inv = H.T @ R_inv @ H + np.linalg.inv(P_p)

       try:
           P_m = np.linalg.inv(P_m_inv)
       except:
           raise ValueError("The computed a posteriori covariance matrix is " +
                            "not invertible.")

       x_m = x_p + P_m @ H.T @ R_inv @ (z - H @ x_p)

       return x_m, P_m


#TEST
x = np.array([2,3,4])
P = np.eye(3)
#A = np.array([[3,2,1],[2,5,2],[7,8,1]])
A = np.eye(3)
b = np.array([20,6,3])
Q = np.eye(3) * 2
x_p, P_p = KalmanFilter.prior_update(x, P, A, b, Q)

z = np.array([5,2])
H = np.array([[1,0,0], [0,1,0]
])
R = np.eye(2) * 0.001
x_m, P_m = KalmanFilter.measurement_update(x_p, P_p, z, H, R)

print(f"x: {x}, x_p: {x_p}, x_m: {x_m}")
print("ey")