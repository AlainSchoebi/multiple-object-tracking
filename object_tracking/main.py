from interactive_sot import InteractiveSOT
from interactive_mot import InteractiveMOT



#InteractiveSOT()
#InteractiveMOT()

from utils.pose import Pose
import numpy as np


p = Pose(np.array([[0,1,0],[0, 0, 1], [1, 0, 0]]), np.array([8, 8, 8]))

print(p)
p2 = Pose()