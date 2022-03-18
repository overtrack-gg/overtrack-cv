import logging
import os
import re
from typing import Optional

import cv2
import numpy as np

from overtrack_cv.core import arrayops, imageops, textops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch.ocr import big_noodle
from overtrack_cv.games.overwatch.processors.score import FinalScore, ScoreScreen
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


class ScoreProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))

    RESULTS = ["VICTORY", "DEFEAT", "DRAW"]

    ROUND_N_COMPLETE = re.compile(r"R[DO]UN[DO]([0-9O]{1,2})COMPLETE")

    def process(self, frame: Frame) -> bool:
        if self.detect_score_screen(frame):
            logger.debug(f"Matched score screen with match={frame.overwatch.score_screen_match}")

            score = self.parse_score_screen(frame)
            if score:
                frame.overwatch.score_screen = score

            return True
        elif self.detect_final_score(frame):
            logger.debug(f"Matched final score with match={frame.overwatch.final_score_match}")

            final = self.parse_final_score(frame)
            if final:
                frame.overwatch.final_score = final

            return True

        return False

    def parse_score_screen(self, frame: Frame) -> Optional[ScoreScreen]:
        score_ims = self.REGIONS["score"].extract(frame.image)
        try:
            blue_score, red_score = big_noodle.ocr_all_int(score_ims, height=212)
        except ValueError as ve:
            logger.warning(f"Failed to parse scores: {ve}")
            return None

        logger.debug(f"Got score {blue_score} / {red_score}")

        # manual thresholding
        im = self.REGIONS["round_text"].extract_one(frame.image)
        # debugops.manual_thresh_otsu(im)
        # im = np.min(im, axis=2)
        # _, thresh = cv2.threshold(im, imageops.otsu_thresh(im, 200, 255), 255, cv2.THRESH_BINARY)
        # round_text = big_noodle.ocr(thresh, threshold=None, height=70, debug=True)
        round_text = big_noodle.ocr(im, channel="min", threshold="otsu_above_mean", height=72, debug=False)

        round_number = None
        match = self.ROUND_N_COMPLETE.match(round_text)
        if match:
            round_number = int(match.group(1).replace("O", "0"))
            logger.debug(f"Got round {round_number} from round string {round_text!r}")
        else:
            logger.warning(f"Could not parse round from round string {round_text!r}")

        return ScoreScreen(blue_score, red_score, round_number)

    def parse_final_score(self, frame: Frame) -> Optional[FinalScore]:
        score_ims = self.REGIONS["final_score"].extract(frame.image)
        score_ims = [imageops.otsu_thresh_lb_fraction(im, 1.4) for im in score_ims]
        try:
            blue_score, red_score = big_noodle.ocr_all_int(score_ims, channel=None, threshold=None, height=127)
        except ValueError as ve:
            logger.warning(f"Failed to parse final score: {ve}")
            return None

        logger.debug(f"Got final score {blue_score} / {red_score}")

        im = cv2.cvtColor(self.REGIONS["final_result_text"].extract_one(frame.image), cv2.COLOR_BGR2HSV_FULL)
        thresh = cv2.inRange(im, (0, 0, 240), (255, 255, 255))
        result_text = big_noodle.ocr(thresh, channel=None, threshold=None, height=120, debug=False)
        matches = textops.matches(result_text, self.RESULTS)
        result: Optional[str]
        if np.min(matches) > 2:
            if blue_score is not None and red_score is not None:
                if blue_score > red_score:
                    result = "VICTORY"
                elif red_score > blue_score:
                    result = "DEFEAT"
                else:
                    result = "DRAW"
                logger.warning(
                    f"Could not identify result from {result_text!r} (match={np.min(matches)}) - "
                    f"using score {blue_score}, {red_score} to infer result={result}"
                )
            else:
                logger.warning(
                    f"Could not identify result from {result_text!r} (match={np.min(matches)}) and did not parse scores"
                )
                result = None
        else:
            result = self.RESULTS[arrayops.argmin(matches)]
            logger.debug(f"Got result {result} from {result_text!r}")

        return FinalScore(blue_score, red_score, result)

    COMPLETE_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "complete_template.png"), 0
    )
    COMPLETE_TEMPLATE_THRESH = 0.6

    def detect_score_screen(self, frame: Frame) -> bool:
        text_region = self.REGIONS["complete_text"].extract_one(frame.image)
        text_region = cv2.resize(text_region, (0, 0), fx=0.5, fy=0.5)
        _, thresh = cv2.threshold(np.min(text_region, 2), 200, 255, cv2.THRESH_BINARY)
        frame.overwatch.score_screen_match = round(
            1 - float(np.min(cv2.matchTemplate(thresh, self.COMPLETE_TEMPLATE, cv2.TM_SQDIFF_NORMED))), 5
        )
        return frame.overwatch.score_screen_match > self.COMPLETE_TEMPLATE_THRESH

    FINAL_SCORE_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "final_score_template.png"), 0
    )
    FINAL_SCORE_TEMPLATE_THRESH = 0.6

    def detect_final_score(self, frame: Frame) -> bool:
        text_region = self.REGIONS["final_score_text"].extract_one(frame.image)
        text_region = cv2.resize(text_region, (0, 0), fx=0.75, fy=0.75)
        thresh = imageops.otsu_thresh_lb_fraction(text_region, 0.8)
        frame.overwatch.final_score_match = round(
            1 - float(np.min(cv2.matchTemplate(thresh, self.FINAL_SCORE_TEMPLATE, cv2.TM_SQDIFF_NORMED))), 5
        )
        return frame.overwatch.final_score_match > self.FINAL_SCORE_TEMPLATE_THRESH


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(ScoreProcessor(), "score_screen", "final_score", "score_screen_match", "final_score_match")
