import logging
import os
import string
from typing import List, Optional, Tuple

import cv2
import numpy as np

from overtrack_cv.core import arrayops, imageops, textops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch import data
from overtrack_cv.games.overwatch.ocr import big_noodle, digit
from overtrack_cv.games.overwatch.processors.endgame import Endgame, Stats
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)

ALL_HEROES = "all heroes"


def load_hero_templates(hero: str) -> List[np.ndarray]:
    im = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "hero", hero + ".png"),
    )

    h = np.min(im, axis=2)
    t = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -10)
    t = t[10:-35, 20:-20]
    templates = [t]

    # find right edge of text
    b = cv2.GaussianBlur(t, (5, 5), 0)
    row = cv2.reduce(b[6:15], 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0] > 200
    right = t.shape[1] - np.argmax(row[::-1])
    right += 10
    if right < 360 and hero != ALL_HEROES:
        # if the text is not all the way to the edge, then the text will *sometimes* have
        # "< summary" if the player played more than 1 hero. Add a template without this text as well
        t2 = t.copy()
        t2[:, right:] = 0
        templates.append(t2)

    return templates


class EndgameProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))

    RESULTS = ["VICTORY", "DEFEAT", "DRAW"]

    VALID_STATS_NAMES = list(data.heroes.keys()) + [ALL_HEROES]

    def process(self, frame: Frame) -> bool:
        if self.detect_endgame(frame):

            result, map_text = self.parse_result_and_map(frame)
            if not result or not map_text:
                return True

            stats = self.parse_stats(frame)

            frame.overwatch.endgame = Endgame(result=result, map=map_text, stats=stats)

            return True

        return False

    def parse_stats(self, frame: Frame) -> Optional[Stats]:
        hero = self.parse_stats_hero(frame)
        if not hero:
            # ALL HEREOS now shows as no text - just parse empty as ALL HEROES and detect failed parses
            hero = ALL_HEROES
        if hero in self.VALID_STATS_NAMES:
            stats = dict(
                zip(
                    [s.name for s in data.generic_stats],
                    big_noodle.ocr_all_int(self.REGIONS["stats"].extract(frame.image), channel="max", height=56),
                )
            )
            logger.debug(f"Parsed stats: {stats}")

            if hero == ALL_HEROES and sum(v is not None for v in stats.values()) <= 2:
                # because we parse unknowns as ALL HEREOS, if the stats failed to parse this is probably not a stats screen
                logger.info(f"Did not get valid stats for potential ALL HEROES stats - ignoring")
                return None

            if stats["objective time"] is not None:
                stats["objective time"] = textops.mmss_to_seconds(stats["objective time"])
                logger.debug(f'Transformed MMSS objective time to {stats["objective time"]}')

            if hero == ALL_HEROES:
                hero_specific_stats = None
            else:
                stat_names_row_1 = [s.name for s in data.heroes[hero].stats[0]]
                stat_names_row_2 = [s.name for s in data.heroes[hero].stats[1]]
                stat_names = stat_names_row_1 + stat_names_row_2
                logger.debug(f"Hero: {hero} has {len(stat_names)} hero specific stats: {stat_names}")

                images = self.REGIONS[f"hero_stats_{len(stat_names)}"].extract(frame.image)
                normed = [
                    ((image.astype(np.float) / np.percentile(image, 98)) * 255).clip(0, 255).astype(np.uint8)
                    for image in images
                ]
                # cv2.imshow('ims', np.vstack((np.hstack(images[:len(stat_names)]), np.hstack(normed[:len(stat_names)]))))
                # cv2.waitKey(0)
                stat_values = digit.ocr_images(normed[: len(stat_names)], scale=0.73)
                hero_specific_stats = dict(zip(stat_names, stat_values))
                logger.info(f"Parsed {hero} stats: {hero_specific_stats}")

            return Stats(
                hero,
                eliminations=stats["eliminations"],
                objective_kills=stats["objective kills"],
                objective_time=stats["objective time"],
                hero_damage_done=stats["hero damage done"],
                healing_done=stats["healing done"],
                deaths=stats["deaths"],
                hero_specific_stats=hero_specific_stats,
            )
        elif hero:
            logging.error(f"Parsed hero name as {hero!r} but was not in list of valid names")
            return None
        else:
            return None

    HERO_STAT_TEMPLATES = []
    for h in VALID_STATS_NAMES:
        for t in load_hero_templates(h):
            HERO_STAT_TEMPLATES.append((h, t))
    HERO_NAME_TEMPLATE_MATCH_THRESHOLD = 0.3

    def parse_stats_hero(self, frame: Frame) -> Optional[str]:
        hero_image = np.min(self.REGIONS["hero_stat_name"].extract_one(frame.image), axis=2)
        hero_image_thresh = cv2.adaptiveThreshold(
            hero_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -10
        )
        matches = sorted(
            [
                (np.min(cv2.matchTemplate(hero_image_thresh, t, cv2.TM_SQDIFF_NORMED)), n)
                for (n, t) in self.HERO_STAT_TEMPLATES
            ]
        )
        logger.debug("Found hero stat matches: " + ", ".join(f"({n}: {m:1.2f})" for (m, n) in matches[:5]) + "...")
        if matches[0][0] < self.HERO_NAME_TEMPLATE_MATCH_THRESHOLD:
            hero = matches[0][1]
            logger.info(f"Classifying stats hero as {hero}")
            return hero
        else:
            logger.debug("Could not identify hero")
            return None

    def parse_result_and_map(self, frame: Frame) -> Tuple[Optional[str], Optional[str]]:
        result_im = self.REGIONS["result"].extract_one(frame.image)
        gray = np.max(result_im, axis=2)
        # mask out white/gray text (this is map and match time info)
        white_text = ((gray > 100) & (np.ptp(result_im, axis=2) < 20)).astype(np.uint8) * 255
        white_text = cv2.erode(white_text, None)
        white_text = np.sum(white_text, axis=0) / 255
        right = np.argmax(white_text > 2)
        if right > 150:
            right -= 10
            logger.info(f"Trimming width of result image {gray.shape[1]} -> {right} to cut white text")
            gray = gray[:, :right]
        else:
            right = gray.shape[1]
        result_text = imageops.tesser_ocr(gray, whitelist="".join(set("".join(self.RESULTS))), invert=True)
        result = textops.matches(result_text, self.RESULTS)
        if np.min(result) > 2:
            logger.warning(f"Could not identify result from {result_text!r} (match={np.min(result)})")
            return None, None

        result = self.RESULTS[arrayops.argmin(result)]
        logger.debug(f"Got result {result} from {result_text!r}")
        # TODO: test this with "draw" result
        map_image = self.REGIONS["map_name"].extract_one(frame.image)[:, right:]
        gray = np.min(map_image, axis=2)
        map_text = textops.strip_string(
            imageops.tesser_ocr(gray, whitelist=string.ascii_uppercase + " :'", invert=True, scale=2),
            string.ascii_uppercase + " ",
        )
        logger.debug(f"Parsed map as {map_text}")

        return result, map_text

    LEAVE_GAME_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "leave_game_template.png"), 0
    )
    LEAVE_GAME_TEMPLATE_THRESH = 0.6

    def detect_endgame(self, frame: Frame) -> bool:
        leave_game_button = self.REGIONS["leave_game_button"].extract_one(frame.image)
        # leave_game_button = cv2.resize(leave_game_button, (0, 0), fx=0.5, fy=0.5)

        gray = np.min(leave_game_button, axis=2)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        frame.overwatch.endgame_match = round(
            1 - float(np.min(cv2.matchTemplate(thresh, self.LEAVE_GAME_TEMPLATE, cv2.TM_SQDIFF_NORMED))), 5
        )
        return frame.overwatch.endgame_match > self.LEAVE_GAME_TEMPLATE_THRESH


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(EndgameProcessor(), "endgame")
