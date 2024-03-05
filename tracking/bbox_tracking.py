# Typing
from __future__ import annotations
from typing import List, Any

# Numpy
from numpy.typing import NDArray

# Utils
from tracking.utils.bbox import BBox

class LabeledBBox(BBox):
    """
    LabeledBBox class that extends BBox with a label.
    """

    def __init__(self, x:int, y: int, w: int, h: int, label: Any):
        super().__init__(x, y, w, h)
        self.label = label

    def copy(self) -> LabeledBBox:
        """
        Return a deep copy of this `LabeledBBox`.
        """
        return LabeledBBox(self.x, self.y, self.w, self.h, self.label)

    @staticmethod
    def from_bbox(bbox: BBox, label: Any) -> LabeledBBox:
        return LabeledBBox(*bbox, label)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.label, str):
            label_str = '"' + self.label + '"'
        else:
            label_str = str(self.label)

        return f"LabeledBBOX(x={self.x:.1f}, y={self.y:.1f}, " + \
               f"w={self.w}, h={self.h:.1f}, label={label_str})"

class Detection(BBox):
    """
    Detection class that extends BBox with a label and a confidence score.
    """

    def __init__(self, x:int, y: int, w: int, h: int,
                label: str, confidence: float):
        super().__init__(x, y, w, h)
        self.label = label
        self.confidence = confidence
        if confidence < 0 or confidence > 1:
            raise ValueError(f"Confidence must be in interval [0, 1] " +
                             f"for creating a Detection.")

    def copy(self) -> Detection:
        """
        Return a deep copy of this `Detection`.
        """
        return Detection(self.x, self.y, self.w, self.h,
                         self.label, self.confidence)

    @staticmethod
    def from_bbox(bbox: BBox, label: str, confidence: float) -> Detection:
        return Detection(*bbox, label, confidence)

    @staticmethod
    def from_yolo_format(bboxes: NDArray, scores: NDArray) -> List[Detection]:
        """
        Create a list of `Detection` from bounding boxes in the YOLO format.

        Inputs
        - bboxes: `NDArray(n, 4)` bounding boxes array in xyxy format
        - scores: `NDArray(n)` confidence score array for each bounding box

        Returns:
        - detections: list of `Detection`
        """
        assert len(bboxes) == len(scores)

        detections = []
        for bbox_data, score in zip(bboxes, scores):

            # Build detection
            bbox = BBox.from_xyxy(*bbox_data)
            detection = Detection.from_bbox(bbox, label=None, confidence=score)
            detections.append(detection)

        return detections

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.label, str):
            label_str = '"' + self.label + '"'
        else:
            label_str = str(self.label)

        return f"Detection(x={self.x:.1f}, y={self.y:.1f}, " + \
               f"w={self.w:.1f}, h={self.h:.1f}, label={label_str}, " + \
               f"confidence={self.confidence:.2f})"