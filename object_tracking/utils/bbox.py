# Typing
from __future__ import annotations
from typing import Optional, Union, List, Tuple

# Numpy
import numpy as np
from numpy.typing import NDArray

# Python
from enum import Enum

# Matplotlib
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class XYXYMode(Enum):
    NORMAL = 0
    PIXEL = 1

class BBox:
    """
    Bounding box class for 2D space.

    Note: TODO pixel!!
    """

    # Constructors
    def __init__(self, x: float = 0, y: float = 0, w: float = 1, h: float = 1):
        self.__x = float(x)
        self.__y = float(y)
        self.__w = float(w)
        self.__h = float(h)
        self._check()

    @staticmethod
    def from_xyxy(x1: float, y1: float, x2: float, y2: float,
                  mode: XYXYMode = XYXYMode.NORMAL) -> BBox:
        if not mode == XYXYMode.NORMAL:
            raise NotImplementedError("Only mode 'XYXYMode.NORMAL' is " +
                                      "supported here.")
        return BBox(x1, y1, x2 - x1, y2 - y1)


    @staticmethod
    def from_center_wh(x_center: float, y_center: float, w: float, h: float,
                       mode: XYXYMode = XYXYMode.NORMAL) -> BBox:
        if not mode == XYXYMode.NORMAL:
            raise NotImplementedError("Only mode 'XYXYMode.NORMAL' is " +
                                      "supported here.")
        return BBox(x_center - 0.5 * w, y_center - 0.5 * h, w, h)

    @staticmethod
    def random(position_max: float = 100,
               size_mean: float = 10, size_std: float = 6) -> BBox:

        w, h = -1, -1
        while w < 0: w = np.random.normal(size_mean, size_std)
        while h < 0: h = np.random.normal(size_mean, size_std)

        x_center = np.random.uniform(0, position_max)
        y_center = np.random.uniform(0, position_max)

        return BBox.from_center_wh(x_center, y_center, w, h)


    def copy(self) -> BBox:
        """
        Return a deep copy of this bounding box.
        """
        return BBox(self.x, self.y, self.w, self.h)

    def _check(self):
        if self.w < 0 or self.h < 0:
            raise ValueError("The width and height of a BBox can't be " +
                             "negative.")

    # Properties
    @property
    def x(self) -> float:
       """
       `float`: Left x coordinate.
       """
       return self.__x

    @property
    def y(self) -> float:
       """
       `float`: Top y coordinate.
       """
       return self.__y

    @property
    def w(self) -> float:
       """
       `float`: Width of the bounding box (>= 0).
       """
       return self.__w

    @property
    def h(self) -> float:
       """
       `float`: Height of the bounding box (>= 0).
       """
       return self.__h

    # Bottom coordinates depending on mode
    def x2(self, mode: XYXYMode = XYXYMode.NORMAL) -> float:
        """
        `float`: Right x coordinate. Depends on mode.
        """
        if mode == XYXYMode.NORMAL:
            return self.x + self.w
        elif mode == XYXYMode.PIXEL:
            return self.x + self.w - 1
        else:
            raise NotImplementedError(f"The mode '{mode}' is not supported.")

    def y2(self, mode: XYXYMode = XYXYMode.NORMAL) -> float:
        """
        `float`: Bottom y coordinate. Depends on mode.
        """
        if mode == XYXYMode.NORMAL:
            return self.y + self.h
        elif mode == XYXYMode.PIXEL:
            return self.y + self.h - 1
        else:
            raise NotImplementedError(f"The mode '{mode}' is not supported.")


    # Equality
    def __eq__(self, x):
        if not isinstance(x, BBox):
            return False

        return self.x == x.x and self.y == x.y and \
               self.w == x.w and self.h == x.h


    # Unpack
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.w
        yield self.h


    # Methods
    def corner_coordinates(self, mode: XYXYMode = XYXYMode.NORMAL) -> NDArray:
        """
        Returns the four corner coordinates:
          - in a CCW manner when the x-axis points right and the y-axis down
          - in a CW manner when the x-axis points right and the y-axis up

        Returns
        - corners: `NDArray(4, 2)` array containing the four corners
        """
        return np.array([[            self.x,             self.y],
                         [            self.x, self.y2(mode=mode)],
                         [self.x2(mode=mode), self.y2(mode=mode)],
                         [self.x2(mode=mode),             self.y]])


    def xywh_tuple(self) -> Tuple[float, float, float, float]:
        """
        Returns a `Tuple` (x, y, w, h).
        """
        return self.x, self.y, self.w, self.h

    def xywh_array(self) -> NDArray:
        """
        Returns a `NDArray(4,)` [x, y, w, h].
        """
        return np.array([*self.xywh_tuple()])


    def xyxy_tuple(self, mode: XYXYMode = XYXYMode.NORMAL) \
          -> Tuple[float, float, float, float]:
        """
        Returns a `Tuple` (x1, y1, x2, y1).
        """
        return self.x, self.y, self.x2(mode=mode), self.y2(mode=mode)

    def xyxy_array(self, mode: XYXYMode = XYXYMode.NORMAL) -> NDArray:
        """
        Returns a `NDArray(4,)` (x1, y1, x2, y1).
        """
        return np.array([*self.xyxy_tuple(mode=mode)])

    def xyxy_matrix(self, mode: XYXYMode = XYXYMode.NORMAL) -> NDArray:
        """
        Returns the top left and bottom right corner coordinates.

        Returns
        - xyxy: `NDArray(2, 2)` array containing the two corners
        """
        return self.xyxy_array(mode=mode).reshape((2,2))


    def center(self, mode: XYXYMode = XYXYMode.NORMAL) -> NDArray:
        """
        Returns the center coordinates of the bounding box.

        Returns
        - center: `NDArray(2)` array containing the center coordinates
        """
        return np.mean(self.xyxy_matrix(mode=mode), axis = 0)

    def center_wh_tuple(self, mode: XYXYMode = XYXYMode.NORMAL) \
          -> Tuple[float, float, float, float]:
        """
        Returns a `Tuple` (x_center, y_center, w, h).
        """
        return *self.center(mode=mode), self.w, self.h

    def center_wh_array(self, mode: XYXYMode = XYXYMode.NORMAL) -> NDArray:
        """
        Returns a `NDArray(4,)` (x_center, y_center, w, h).
        """
        return np.array([*self.center_wh_tuple(mode=mode)])


    # Representations
    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return f"BBox(x={self.x}, y={self.y}, w={self.w}, h={self.h})"


    # Operators
    def __mul__(self, scale: float) -> BBox:
        """
        Scale the bounding box by a scalar.

        Inputs
        - scale: `float` or `int` scale value
        """
        if not isinstance(scale, (int, float)):
            raise NotImplementedError(f"BBox multiplication with type " +
                                      f"{type(scale)} is not supported.")

        return BBox.from_center_wh(*self.center(),
                                   scale * self.w, scale * self.h)

    def __rmul__(self, scale: float) -> BBox:
        return self.__mul__(scale)


    # Visualization functions
    if MATPLOTLIB_AVAILABLE :
        def show(self, axes: Optional[Axes] = None) -> Axes:
            """
            Visualize the BBox in a matloptlib plot.
            """
            return BBox.visualize(self, axes)

        @staticmethod
        def visualize(bboxes: Union[BBox, List[BBox]],
                      axes: Optional[Axes] = None) -> Axes:
            """
            Visualize a list of BBoxes in a matloptlib plot.

            Inputs:
            - bboxes: list of BBoxs to plot
            """

            if type(bboxes) == BBox:
                bboxes = [bboxes]

            # No axes provided
            if axes is None:
                # Create figure
                fig = plt.figure()
                ax: Axes = fig.add_subplot()

                # Title
                ax.set_title(f"XYWH{'s' if len(bboxes) > 1 else ''} " +
                             f"visualization")

                # Axis labels
                ax.set_xlabel('x')
                ax.set_ylabel('y')

            # Axes provided
            else:
                ax = axes

            # Default color cycle
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colors = [matplotlib.colors.to_rgb(color) for color in colors]

            alpha = max(0.2, 1/len(bboxes))

            # Plot poses
            for i, (bbox, color) in enumerate(zip(bboxes, colors)):

                # Rectangle
                rectangle = matplotlib.patches.Rectangle(
                    (bbox.x, bbox.y), bbox.w, bbox.h, alpha=alpha,
                    edgecolor=np.array(color) * 0.7, facecolor=color,
                   )
                ax.add_patch(rectangle)

                # Labels
                if len(bboxes) > 1 or \
                   hasattr(bbox, "label") or hasattr(bbox, "name"):
                    label = i
                    if hasattr(bbox, "name"):
                        label = bbox.name
                    elif hasattr(bbox, "label"):
                        label = bbox.label
                    ax.text(*bbox.center(), label, ha='center', va='center',
                            alpha=alpha, color=np.array(color) * 0.7,
                            fontsize=12)

            if axes is None:
                # Axis parameters
                ax.axis('equal')
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                ax.invert_yaxis()

                # Show
                plt.show()

            return ax