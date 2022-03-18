import logging
import os
import string

import cv2
import numpy as np

from overtrack_cv.core import imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch.processors.menu import MainMenu, PlayMenu
from overtrack_cv.games.processor import Processor

logger = logging.getLogger("MenuProcessor")


def _draw_main_menu(debug_image: np.ndarray, main_menu: MainMenu) -> None:
    if debug_image is None:
        return

    cv2.putText(debug_image, "Main Menu", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 4)
    cv2.putText(debug_image, main_menu.version, (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 2)


def _draw_play_menu(debug_image: np.ndarray, play_menu: PlayMenu) -> None:
    if debug_image is None:
        return

    if play_menu.placements:
        text = "placement"
    else:
        text = f"SR: {play_menu.sr}"

    cv2.putText(debug_image, text, (980, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 128, 255), 2)


class MenuProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))

    PLACEMENT_MATCHES_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "placement_matches.png"), 0
    )
    PLACEMENT_MATCHES_TEMPLATE_THRESHOLD = 0.6

    def process(self, frame: Frame) -> bool:
        if frame.overwatch.main_menu or frame.overwatch.play_menu:
            return True

        self.REGIONS.draw(frame.debug_image)
        if self.detect_main_menu(frame):
            version_region = self.REGIONS["version"].extract_one(frame.image)
            thresh = imageops.otsu_thresh_lb_fraction(version_region, 0.75)
            version = imageops.tesser_ocr(thresh, whitelist=string.digits + ".-", invert=True, scale=4, blur=2)

            frame.overwatch.main_menu = MainMenu(version=version)

            _draw_main_menu(frame.debug_image, frame.overwatch.main_menu)

            return True

        elif self.detect_play_menu(frame):
            # placement_region = self.REGIONS['placement_matches'].extract_one(frame.image)
            # placement_region = cv2.cvtColor(placement_region, cv2.COLOR_BGR2GRAY)
            # _, thresh = cv2.threshold(placement_region, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # match = 1 - float(np.min(cv2.matchTemplate(thresh, self.PLACEMENT_MATCHES_TEMPLATE, cv2.TM_SQDIFF_NORMED)))
            # is_placements = match > self.PLACEMENT_MATCHES_TEMPLATE_THRESHOLD
            #
            # if not is_placements:
            #     group_sr_region = self.REGIONS['group_sr'].extract_one(frame.image)
            #     color_variance = np.mean(np.var(group_sr_region, axis=(0, 1)))
            #     if color_variance < 100:
            #         # only one color - maybe placement
            #         logger.warning(f'Got low color variance ({color_variance:.2f}) - ignoring parsed SR')
            #         sr = None
            #     else:
            #         sr = self.read_sr(frame)
            # else:
            #     sr = None
            #
            # frame.overwatch.play_menu = PlayMenu(
            #     placements=is_placements,
            #     sr=sr,
            #     image=lazy_upload(
            #         'sr_full',
            #         self.REGIONS['sr_full'].extract_one(frame.image),
            #         frame.timestamp
            #     )
            # )
            #
            # _draw_play_menu(frame.debug_image, frame.overwatch.play_menu)

            return True

        return False

    # def read_sr(self, frame):
    #     group_sr_region = self.REGIONS['group_sr'].extract_one(frame.image)
    #     personal_sr_region = self.REGIONS['personal_sr'].extract_one(frame.image)
    #
    #     # try read personal SR using hue - this can read behind text on raw images
    #     personal_sr_hue = 255 - cv2.cvtColor(personal_sr_region, cv2.COLOR_BGR2HSV_FULL)[:, :, 0]
    #     sr = self._parse_sr(personal_sr_hue, 'personal_hue')
    #     if sr:
    #         return sr
    #
    #     # try read group using grayscale - requires group leader
    #     sr = self._parse_sr(np.min(personal_sr_region, axis=2), 'personal')
    #     if sr:
    #         return sr
    #
    #     # try read "group" SR region - this will reject SR if it is actual group SR
    #     sr = self._parse_sr(np.min(group_sr_region, axis=2), 'group')
    #     if sr:
    #         return sr
    #
    #     return sr
    #
    # def _parse_sr(self, im: np.ndarray, typ: str) -> Optional[int]:
    #     _, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #     thresh = cv2.dilate(thresh, np.ones((5, 5)))
    #
    #     labels, components = imageops.connected_components(thresh)
    #     components = sorted(components, key=lambda c: c.x)
    #     components = [
    #         c for c in components
    #         if c.y and c.x + c.w != im.shape[1] and c.y + c.h != im.shape[0] and
    #         (abs(c.h - 17) > 3 or c.w < 60) and
    #         c.x < 200
    #     ]
    #     if not len(components):
    #         logger.debug(f'{typ}: Got 0 components for sr')
    #         return None
    #
    #     leftmost = components[0].x + components[0].w
    #     if leftmost > 150:
    #         logger.warning(f'{typ}: Rank icon at {leftmost} - rejecting group SR')
    #         return None
    #
    #     logger.info(f'{typ}: Found rank icon at {leftmost}')
    #
    #     im = im[:, leftmost:leftmost + 150]
    #     result = imageops.tesser_ocr(im, int, engine=imageops.tesseract_only)
    #     logger.debug(f'{typ}: Parsed SR as {result}')
    #     if not result:
    #         result = imageops.tesser_ocr(im, int, engine=imageops.tesseract_lstm)
    #         logger.debug(f'{typ}: Parsed SR as {result}')
    #
    #     if result and 500 <= result <= 5000:
    #         return result
    #     else:
    #         logger.warning(f'{typ}: Got invalid SR: {result}')
    #         return None

    OVERWATCH_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "overwatch_template.png"), 0
    )
    OVERWATCH_TEMPLATE_THRESH = 0.6

    def detect_main_menu(self, frame: Frame) -> bool:
        text_region = self.REGIONS["overwatch_text"].extract_one(frame.image)
        text_region = cv2.resize(text_region, (0, 0), fx=0.5, fy=0.5)

        thresh = imageops.otsu_thresh_lb_fraction(text_region, 1.1)
        frame.overwatch.main_menu_match = round(
            1 - float(np.min(cv2.matchTemplate(thresh, self.OVERWATCH_TEMPLATE, cv2.TM_SQDIFF_NORMED))), 5
        )
        return frame.overwatch.main_menu_match > self.OVERWATCH_TEMPLATE_THRESH

    COMPETITIVE_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "competitive_play.png"), 0
    )
    COMPETITIVE_TEMPLATE_LARGE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "competitive_play_large.png"), 0
    )
    COMPETITIVE_TEMPLATE_THRESH = 0.6

    def detect_play_menu(self, frame: Frame) -> bool:
        competitive_region = self.REGIONS["competitive_play"].extract_one(frame.image)
        competitive_region = cv2.resize(competitive_region, (0, 0), fx=0.5, fy=0.5)

        gray = np.min(competitive_region, axis=2)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        match = 0.0
        for t in self.COMPETITIVE_TEMPLATE, self.COMPETITIVE_TEMPLATE_LARGE:
            match = max(match, round(1 - float(np.min(cv2.matchTemplate(thresh, t, cv2.TM_SQDIFF_NORMED))), 5))
        frame.overwatch.play_menu_match = match
        return frame.overwatch.play_menu_match > self.COMPETITIVE_TEMPLATE_THRESH


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(MenuProcessor(), "play_menu", "play_menu_match", "main_menu", "main_menu_match")
