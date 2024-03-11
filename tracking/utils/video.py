# Typing
from typing import List, Any

# Python
from tqdm import tqdm
from pathlib import Path
import re

# OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from tracking.utils.loggers import get_logger
logger = get_logger(__name__)

def sorted_nicely(l: List[Any]) -> List[Any]:
    """
    Sort the given list in the way that humans expect.

    Input
    - l: `List[Any]` the list to be sorted.

    Return
    - `List[Any]` the sorted list.
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', str(s))]

    return sorted(l, key=alphanum_key)


if OPENCV_AVAILABLE:
    def generate_video(images: List[Path] | List[str],
                       video_filename: Path | str,
                       frame_rate: int = 5) -> None:

        if len(images) == 0:
            logger.warning("The images list contains zero images. Not generating any video.")
            return

        # Get the image dimensions
        frame = cv2.imread(str(images[0]))
        height, width, layers = frame.shape

        # Initialize the VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(video_filename), fourcc, frame_rate, (width, height))

        # Loop through the images and add them to the video
        for image in tqdm(images, total=len(images), desc="Generating video"):

            # Read image and add the frame to the video
            frame = cv2.imread(str(image))
            video.write(frame)

        # Close the video file
        video.release()
        logger.info(f"Successfully generated video with {len(images)} frames to '{video_filename}'.")


    def generate_video_from_folder(folder: Path | str,
                                   video_filename: Path | str,
                                   image_extension: str = ".png",
                                   frame_rate: int = 5) -> None:
        images = sorted_nicely(list(Path(folder).glob(f"*{image_extension}")))
        generate_video(images, video_filename, frame_rate=frame_rate)