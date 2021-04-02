import logging
import os
import re
from glob import glob
from typing import Optional

import cv2
import numpy as np

from overtrack_cv.core.tflite import TFLiteSequenceModel
from overtrack_cv.frame import Frame
from overtrack_cv.games.apex.processors.coordinates.models import Coordinates
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


def _draw_coordinates(debug_image: Optional[np.ndarray], coords: Coordinates) -> None:
    if debug_image is None:
        return

    line = repr(coords)
    cv2.putText(
        debug_image,
        line,
        (300, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        debug_image,
        line,
        (300, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
    )


class CoordinateProcessor(Processor):
    SINGLE_COORD = r"(-?[0-9]+ [0-9]{2})"
    COORD_PATTERN = re.compile(" ".join([SINGLE_COORD] * 3))
    OCR = TFLiteSequenceModel(os.path.join(os.path.dirname(__file__), "data", "coordinate_ocr"))
    COORDINATES_TOP = 60
    SPACE_WIDTH = 14

    def process(self, frame: Frame) -> bool:
        image = getattr(frame, "image_unscaled", frame.image)
        im = image[self.COORDINATES_TOP : self.COORDINATES_TOP + 13, 45 : 45 + 220]

        # The OCR only reads "-[0-9]" - spaces and decimals are detected by the gap between digits
        # e.g. -123 45 678 90 -123 45 -> -123.45 678.90 -123.45 -> (-123.45, 678.90, -123.45)
        position = self.OCR.predict(np.expand_dims(im, 0).astype(np.float32))[0]
        result = " ".join(position.words(self.SPACE_WIDTH))
        match = self.COORD_PATTERN.fullmatch(result)
        if match:
            frame.apex.coordinates = Coordinates(*[float(c.replace(" ", ".")) for c in match.groups()])
            _draw_coordinates(frame.debug_image, frame.apex.coordinates)
            return True
        # else:
        #     logger.warning(f"Got invalid coordinates parse: {position.string}")

        return False


def main():
    from overtrack_cv.util.test_processor import test_processor

    proc = CoordinateProcessor()

    test_processor(proc, "coordinates", test_all=False, warmup=False)

    paths = sorted(
        [
            f
            for f in glob("S:/Downloads/apexframesdump/*.png")
            if float(os.path.basename(f).split("_")[0]) > 1612499100
        ]
    )
    test_processor(proc, "coordinates", images=paths)


if __name__ == "__main__":
    main()
