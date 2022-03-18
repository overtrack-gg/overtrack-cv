import logging
import os

import cv2
import numpy as np

from overtrack_cv.core import imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch.ocr import big_noodle
from overtrack_cv.games.overwatch.processors.hero_select import AssembleYourTeam
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


class HeroSelectProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))

    ASSEMBLE_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "assemble_your_team.png"), 0
    )
    ASSEMBLE_THRESH = 0.8

    ASSEMBLE_HSV_RANGE = [(0, 0, 200), (255, 15, 255)]

    def process(self, frame: Frame) -> bool:
        im = self.REGIONS["assemble_your_team"].extract_one(frame.image)
        thresh = cv2.inRange(
            cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL), self.ASSEMBLE_HSV_RANGE[0], self.ASSEMBLE_HSV_RANGE[1]
        )
        match = np.max(cv2.matchTemplate(thresh, self.ASSEMBLE_TEMPLATE, cv2.TM_CCORR_NORMED))

        frame.overwatch.assemble_your_team_match = round(float(match), 5)
        if match > self.ASSEMBLE_THRESH:
            map_image = self.REGIONS["map"].extract_one(frame.image)
            map_thresh = imageops.otsu_thresh_lb_fraction(np.min(map_image, axis=2), 1.1)
            map_text = big_noodle.ocr(map_thresh, channel=None, threshold=None)

            mode_image = self.REGIONS["mode"].extract_one(frame.image_yuv[:, :, 0])
            mode_thresh = cv2.adaptiveThreshold(
                mode_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, -10
            )
            mode_text = big_noodle.ocr(mode_thresh, channel=None, threshold=None)

            name_images = self.REGIONS["blue_names"].extract(frame.image)
            blue_thresh = [cv2.inRange(i, (200, 145, 0), (255, 255, 130)) for i in name_images]
            green_thresh = [cv2.inRange(i, (0, 220, 200), (100, 255, 255)) for i in name_images]
            name_thresh = [cv2.bitwise_or(i1, i2) for i1, i2 in zip(blue_thresh, green_thresh)]
            names = [
                big_noodle.ocr(
                    i[:, :, 1],
                    channel=None,
                    threshold=t,
                    # debug=True
                )
                for i, t in zip(name_images, name_thresh)
            ]

            frame.overwatch.assemble_your_team = AssembleYourTeam(
                map=map_text,
                mode=mode_text,
                blue_names=names,
                is_in_queue=self.detect_in_queue(frame),
            )
            return True

        return False

    TIME_ELAPSED_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "time_elapsed.png"), 0)
    TIME_ELAPSED_MATCH_THRESHOLD = 0.75

    def detect_in_queue(self, frame: Frame) -> bool:
        region = self.REGIONS["time_elapsed"].extract_one(frame.image_yuv[:, :, 0])

        _, thresh = cv2.threshold(region, 130, 255, cv2.THRESH_BINARY)
        match = np.max(cv2.matchTemplate(thresh, self.TIME_ELAPSED_TEMPLATE, cv2.TM_CCORR_NORMED))
        logger.debug(f"Time elapsed match={match:.1f}")

        return match > self.TIME_ELAPSED_MATCH_THRESHOLD


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(HeroSelectProcessor(), "assemble_your_team", "assemble_your_team_match")
