import glob
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from overtrack_cv.core import arrayops, imageops, textops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch import data
from overtrack_cv.games.overwatch.ocr import big_noodle
from overtrack_cv.games.overwatch.processors.hero import Hero
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


def _load_template(im: np.ndarray) -> np.ndarray:
    return cv2.resize(im, (0, 0), fx=0.5, fy=0.5)


def _circle(r: float, n: int = 100) -> np.ndarray:
    return np.array(
        [(np.cos(2 * np.pi / n * x) * r + r, np.sin(2 * np.pi / n * x) * r + r) for x in range(0, n + 1)],
        dtype=np.int32,
    )


def _draw_hero(debug_image: np.ndarray, hero: Hero) -> None:
    if debug_image is None:
        return

    for t, c in (5, (0, 0, 0)), (2, (255, 0, 255)):
        cv2.putText(debug_image, str(hero), (50, 1080 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, c, t)


class HeroProcessor(Processor):

    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))
    # STATE_TEMPLATES = {
    #     k: imageops.imread(os.path.join(os.path.dirname(__file__), 'data', 'state_templates', k + '.png'), 0)
    #     for k in ['eliminated', 'spectating', 'potg']
    # }
    STATES = [
        "ELIMINATED BY",
        "YOU ARE NOW DEATH SPECTATING",
        "YOU ARE NOW SPECTATING",
        "PLAY OF THE GAME BY",
        "PLAY OF THE MATCH BY",
    ]
    WEAPON_TEMPLATES: Optional[List[Tuple[str, np.ndarray]]] = None
    ULT_CONTOUR_TEMPLATE = np.expand_dims(_circle(21, 40), 0)

    # TODO: add dva pilot, ashe
    def __init__(self) -> None:
        if self.WEAPON_TEMPLATES is None:
            self.WEAPON_TEMPLATES = [
                (str(os.path.basename(p)).split(".")[0], _load_template(imageops.imread(p, 0)))
                for p in glob.glob(os.path.join(os.path.dirname(__file__), "data", "weapon_templates", "*.png"))
            ]
            for hero in sorted(data.heroes):
                if hero not in [n[0] for n in self.WEAPON_TEMPLATES]:
                    logger.warning(f"Did not get weapon template for {hero}")
                else:
                    weapon_count = len([t for t in self.WEAPON_TEMPLATES if t[0].startswith(hero)])
                    logger.info(f"Got {weapon_count} weapon templates for {hero}")
        logger.info(
            f"Loaded {len(self.WEAPON_TEMPLATES)} weapon templates",
        )

    def process(self, frame: Frame) -> bool:
        hero_name = self._parse_hero_from_weapon(frame)
        state = self._parse_state(frame)

        ult_status = None
        if hero_name:
            ult_status = self._parse_ult_status(frame)

        frame.overwatch.hero = Hero(
            hero=hero_name,
            ult=ult_status,
            potg="PLAY OF THE" in state,
            spectating="SPECTATING" in state,
            killcam="ELIMINATED BY" == state,
        )

        _draw_hero(frame.debug_image, frame.overwatch.hero)

        return bool(frame.overwatch.hero.hero or state)

    def _parse_hero_from_weapon(self, frame: Frame) -> Optional[str]:
        image = self.REGIONS["weapon"].extract_one(frame.image)
        gray = np.min(image, axis=2)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 205, -51)
        if thresh.shape != (81, 231) and thresh.shape != (100, 220):
            logger.warning("Ignoring weapon image with dimensions %s", thresh.shape)
            return None
        thresh = cv2.resize(
            cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0), (0, 0), fx=0.5, fy=0.5
        )
        assert self.WEAPON_TEMPLATES is not None
        matches = [np.min(cv2.matchTemplate(thresh, t, cv2.TM_SQDIFF_NORMED)) for h, t in self.WEAPON_TEMPLATES]
        match: int = arrayops.argmin(matches)
        if matches[match] < 0.3:
            h = self.WEAPON_TEMPLATES[match][0]
            logger.debug(f"Got hero {h} with match {matches[match]:.2f}")
            return h
        else:
            return None

    def _parse_state(self, frame: Frame) -> str:
        map_info_image = self.REGIONS["potg_eliminted_deathspec"].extract_one(frame.image)
        yellow_text = cv2.inRange(
            cv2.cvtColor(map_info_image, cv2.COLOR_BGR2HSV_FULL),
            ((35 / 360) * 255, 0.5 * 255, 0.8 * 255),
            ((55 / 360) * 255, 1.0 * 255, 1.0 * 255),
        )
        p = np.sum(yellow_text > 0) / np.prod(yellow_text.shape)
        state = ""
        if 0.05 < p < 0.4:
            state_text = big_noodle.ocr(yellow_text, channel=None)
            if state_text and len(state_text) > 5:
                state_text_matches = textops.matches(state_text, self.STATES)
                match_i: int = arrayops.argmin(state_text_matches)
                match = state_text_matches[match_i]
                if match < 7:
                    state = self.STATES[match_i]
                    logger.info(
                        f"Got state={state_text!r} (text fill: {p*100:.0f}%) -> best match: {state!r} (match={match})"
                    )
                else:
                    logger.warning(
                        f'Got state={state_text!r}, but this was not recognized as a valid state (closest was "{self.STATES[match_i]}", match={match})'
                    )
        return state

    def _parse_ult_status(self, frame: Frame) -> Optional[int]:
        ult_image = self.REGIONS["ult"].extract_one(frame.image)
        thresh = imageops.unsharp_mask(ult_image, unsharp=2, weight=4, threshold=240)

        ult_circle = cv2.resize(thresh, (0, 0), fx=0.5, fy=0.5)
        contours, _ = imageops.findContours(ult_circle, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_match = 1.0
        for cnt in contours:
            if 1500 < cv2.contourArea(cnt) < 2500:
                contour_match = min(contour_match, cv2.matchShapes(cnt, self.ULT_CONTOUR_TEMPLATE, 1, 0))

        if contour_match < 0.01:
            logger.debug(f"Got ult contour match {contour_match:1.5f} - ult=100%")
            return 100
        else:
            ult = big_noodle.ocr_int(thresh, channel=None, threshold=None, height=33)
            logger.debug(f"Parsed ult as {ult}%")
            if ult is not None and 0 <= ult <= 99:
                return ult
            else:
                return None


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(HeroProcessor(), "hero")
