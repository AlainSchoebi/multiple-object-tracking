import sys
#sys.path.insert(0, "/home/bmw/alain/inference_api/") # TODO
#sys.path.insert(0, "C:/Users/Q637136/src/inference_api") # TODO
sys.path.insert(0, "C:/Users/alain/source/repos/object-tracking") # TODO

from tracking.interactive_mot import InteractiveMOT

class InteractiveSOT(InteractiveMOT):

    def __init__(self):

        config = {
        "iou_threshold": {
            "association_1": float(0.0),
            "association_2": float(0.0)
        },
        "tracklet_config": {
            "kf_measurement_noise": float(0.1),
        }
        }
        super().__init__(config)


    def init_plot(self):

        ax = self.tracker.show(show=False, title="Interactive SOT")

        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event',
                                                      self.on_click)
        self.fig = ax.get_figure()
        self.ax = ax


if __name__ == "__main__":
    InteractiveSOT()