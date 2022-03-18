import logging
import os

import cv2
import numpy as np

from overtrack_cv.core import imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.core.uploadable_image import lazy_upload
from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch.ocr import big_noodle
from overtrack_cv.games.overwatch.processors.endgame_sr import EndgameSR
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


class EndgameSRProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))

    COMPETITIVE_POINTS_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "competitive_points.png"), 0
    )
    COMPETITIVE_POINTS_THRESH = 0.8

    def process(self, frame: Frame) -> bool:
        y = frame.image_yuv[:, :, 0]
        im = self.REGIONS["competitive_points"].extract_one(y)
        _, thresh = cv2.threshold(im, 50, 255, cv2.THRESH_BINARY)
        match = np.max(cv2.matchTemplate(thresh, self.COMPETITIVE_POINTS_TEMPLATE, cv2.TM_CCORR_NORMED))

        frame.overwatch.endgame_sr_match = round(float(match), 5)

        if match > self.COMPETITIVE_POINTS_THRESH:
            sr_image = self.REGIONS["sr"].extract_one(y)
            sr = big_noodle.ocr_int(sr_image)
            if sr is None:
                logger.warning(f"Unable to parse SR")
            else:
                frame.overwatch.endgame_sr = EndgameSR(
                    sr, image=lazy_upload("end_sr", self.REGIONS.blank_out(frame.image), frame.timestamp)
                )
                return True

        return False


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(EndgameSRProcessor(), "endgame_sr", "endgame_sr_match")
