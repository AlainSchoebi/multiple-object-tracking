# Typing
from typing import List

# Numpy
import numpy as np
from numpy.typing import NDArray

# Librairies
import cython_bbox

# Object Tracking
from tracklet import Tracklet

# Utils
from utils.bbox import XYXYMode
from bbox_tracking import Detection

def iou(tracklets: List[Tracklet], detections: List[Detection]) \
    -> NDArray:
    """
    Compute the Intersection-over-Union between a list of tracklets and a list
    of detections.

    Inputs
    - tracklets: list of N `Tracklet`
    - detections: list of M `Detection`

    Returns
    - iou: `NDArray(N, M)` containing IoU(tracklet_i, detection_j)
    """

    # Assert not empty lists
    if len(tracklets) == 0 or len(detections) == 0: return None

    # Turn tracklets into XYXY array (with PIXEL model)
    tracklets_xyxy = []
    for tracklet in tracklets:
        xyxy = tracklet.bbox().xyxy_array(mode=XYXYMode.PIXELNOERROR)
        tracklets_xyxy.append(xyxy)
    tracklets_xyxy = np.array(tracklets_xyxy)

    # Turn detections into XYXY array (with PIXEL model)
    detections_xyxy = []
    for detection in detections:
        xyxy = detection.xyxy_array(mode=XYXYMode.PIXELNOERROR)
        detections_xyxy.append(xyxy)
    detections_xyxy = np.array(detections_xyxy)

    # Compute IoU
    return cython_bbox.bbox_overlaps(tracklets_xyxy, detections_xyxy)
