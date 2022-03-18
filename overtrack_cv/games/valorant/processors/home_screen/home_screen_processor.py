import logging
import os
from typing import Optional

import cv2
import Levenshtein as levenshtein
import numpy as np

from overtrack_cv.core import imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.processor import Processor
from overtrack_cv.games.valorant.processors.home_screen.models import HomeScreen

logger = logging.getLogger("HomeScreenProcessor")


def draw_home_screen(debug_image: Optional[np.ndarray], home_screen: HomeScreen) -> None:
    if debug_image is None:
        return

    for c, t in ((0, 0, 0), 4), ((0, 255, 128), 2):
        cv2.putText(
            debug_image,
            str(home_screen),
            (1450, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            c,
            t,
        )


class HomeScreenProcessor(Processor):

    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))
    PLAY_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "play.png"), 0)
    SEARCH_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "search.png"), 0)

    def process(self, frame: Frame) -> bool:
        if frame.valorant.home_screen:
            return True

        # self.REGIONS.draw(frame.debug_image)
        if not imageops.match_thresh_template(
            self.REGIONS["play"].extract_one(frame.image_yuv[:, :, 0]),
            self.PLAY_TEMPLATE,
            130,
            0.8,
        ):
            return False

        if not imageops.match_thresh_template(
            self.REGIONS["search"].extract_one(frame.image_yuv[:, :, 0]),
            self.SEARCH_TEMPLATE,
            100,
            0.8,
        ):
            return False

        play_text = imageops.ocr_region(
            frame,
            self.REGIONS,
            "play",
        )
        if levenshtein.distance(play_text.upper(), "PLAY") > 1:
            return False

        frame.valorant.home_screen = HomeScreen()
        draw_home_screen(frame.debug_image, frame.valorant.home_screen)
        return True

    def ocr_match(self, frame: Frame, region: str, target: str, requirement: float) -> bool:
        text = self.ocr_region(frame, region)
        match = levenshtein.ratio(text.upper(), target.upper())
        logger.debug(
            f"OCR match {text.upper()!r} ~ {target.upper()!r} => {match:.2f} > {requirement:.2f} => {match > requirement}"
        )
        return match > requirement

    def ocr_region(self, frame: Frame, target_region: str):
        region = self.REGIONS[target_region].extract_one(frame.image)
        gray = 255 - imageops.normalise(np.min(region, axis=2))
        text = imageops.tesser_ocr(
            gray,
            engine=imageops.tesseract_lstm,
        )
        return text


def main():
    from overtrack_cv.util.logging_config import config_logger
    from overtrack_cv.util.test_processor import test_processor

    config_logger(os.path.basename(__file__), level=logging.DEBUG, write_to_file=False)

    test_processor(HomeScreenProcessor(), "valorant.home_screen")


if __name__ == "__main__":
    main()
