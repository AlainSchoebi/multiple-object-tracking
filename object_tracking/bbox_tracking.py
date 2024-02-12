# Typing
from __future__ import annotations
from typing import Any

# Utils
from utils.bbox import BBox

class LabeledBBox(BBox):

    def __init__(self, x:int, y: int, w: int, h: int, label: Any):
        super().__init__(x, y, w, h)
        self.label = label

    @staticmethod
    def from_bbox(bbox: BBox, label: Any) -> LabeledBBox:
        return LabeledBBox(*bbox, label)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"LabeledBBOX(x={self.x}, y={self.y}, " + \
               f"w={self.w}, h={self.h}, label='{self.label}')"

class Detection(BBox):

    def __init__(self, x:int, y: int, w: int, h: int,
                label: str, confidence: float):
        super().__init__(x, y, w, h)
        self.label = label
        self.confidence = confidence
        if confidence < 0 or confidence > 1:
            raise ValueError(f"Confidence must be in interval [0, 1] " +
                             f"for creating a Detection.")

    @staticmethod
    def from_bbox(bbox: BBox, label: str, confidence: float) -> Detection:
        return Detection(*bbox, label, confidence)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Detection(x={self.x}, y={self.y}, " + \
               f"w={self.w}, h={self.h}, label='{self.label}', " + \
               f"confidence={self.confidence:.2f})"