import logging
import os
import string

import cv2
import numpy as np

from overtrack_cv.core import arrayops, imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.core.uploadable_image import lazy_upload
from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch.ocr import big_noodle
from overtrack_cv.games.overwatch.processors.role_select import RoleSelect
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


def _draw_role_select(debug_image: np.ndarray, role_select: RoleSelect) -> None:
    if debug_image is None:
        return

    cv2.putText(
        debug_image,
        str(role_select).split(", image=", 1)[0] + ")",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )


class RoleSelectProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))
    TANK_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "tank.png"), 0)
    TANK_LARGE_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "tank_large.png"), 0)
    LOCK_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "lock.png"), 0)

    REQUIRED_MATCH = 0.95

    def process(self, frame: Frame) -> bool:
        y = frame.image_yuv[:, :, 0]
        tank_region = np.max(self.REGIONS["tank_region"].extract_one(frame.image), axis=2)

        _, thresh = cv2.threshold(tank_region, 100, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh1', thresh)

        tank_match_sm = cv2.matchTemplate(thresh, self.TANK_TEMPLATE, cv2.TM_CCORR_NORMED)
        _, match_sm, _, mxloc_sm = cv2.minMaxLoc(tank_match_sm)

        tank_match_lg = cv2.matchTemplate(thresh, self.TANK_LARGE_TEMPLATE, cv2.TM_CCORR_NORMED)
        _, match_lg, _, mxloc_lg = cv2.minMaxLoc(tank_match_lg)

        lock_match = cv2.matchTemplate(thresh, self.LOCK_TEMPLATE, cv2.TM_CCORR_NORMED)
        _, match_lock, _, mxloc_lock = cv2.minMaxLoc(lock_match)

        matched_i = arrayops.argmax([match_sm, match_lg, match_lock])
        # print([match_sm, match_lg, match_lock])
        match = [match_sm, match_lg, match_lock][matched_i]
        matched = ["tank", "tank_lg", "lock"][matched_i]
        best_match_pos = [mxloc_sm, mxloc_lg, mxloc_lock][matched_i]
        match_x = best_match_pos[0]
        # print(matched, match_x)

        frame.overwatch.role_select_match = round(match, 2)

        if match > self.REQUIRED_MATCH:
            grouped = match_x < 150

            logger.debug(
                f"Found match for {matched!r} with match={match:0.3f} ({match_sm:.2f}, {match_lg:.2f}, {match_lock:.2f}), x={match_x} => grouped={grouped}"
            )

            suffix = "_group" if grouped else "_solo"
            frame.overwatch.role_select = RoleSelect(
                placement_text=imageops.tesser_ocr_all(
                    self.REGIONS["placements" + suffix].extract(y), whitelist=string.digits + "/-"
                ),
                sr_text=big_noodle.ocr_all(self.REGIONS["srs" + suffix].extract(y), height=23, invert=True),
                account_name=imageops.tesser_ocr(
                    self.REGIONS["account_name"].extract_one(y), engine=imageops.tesseract_lstm
                ),
                grouped=grouped,
                image=lazy_upload(
                    "role_select", self.REGIONS.blank_out(frame.image), frame.timestamp, selection="last"
                ),
            )
            if frame.debug_image is not None:
                self.REGIONS.draw(frame.debug_image)
            _draw_role_select(frame.debug_image, frame.overwatch.role_select)
            return True

        return False


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(RoleSelectProcessor(), "role_select", "role_select_match")
