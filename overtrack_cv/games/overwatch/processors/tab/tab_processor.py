import glob
import itertools
import logging
import os
import string
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from overtrack_cv.core import imageops, textops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch import data
from overtrack_cv.games.overwatch.ocr import big_noodle, digit
from overtrack_cv.games.overwatch.processors.endgame import Stats
from overtrack_cv.games.overwatch.processors.tab import TabScreen
from overtrack_cv.games.overwatch.processors.tab.models import NameImages
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


def _draw_tab_screen(debug_image: np.ndarray, tab_screen: TabScreen) -> None:
    if debug_image is None:
        return

    lines: List[Tuple[Tuple[int, int], Optional[str]]] = [
        ((320, 850), tab_screen.player_name),
        ((1250, 850), tab_screen.player_hero),
    ]
    x = 387
    y1 = 180
    y2 = 780
    y3 = 810
    w = 192
    for i in range(6):
        lines.append(((x, y1), tab_screen.red_team[i]))
        lines.append(((x, y1 + 150), tab_screen.red_team_hero[i]))

        lines.append(((x, y2), tab_screen.blue_team[i]))
        lines.append(((x, y2 - 150), tab_screen.blue_team_hero[i]))
        lines.append(((x, y3), f"{tab_screen.blue_team_ults[i]}%"))
        x += w
    for pos, line in lines:
        if line:
            cv2.putText(debug_image, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)


def _load_template(im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    im = im[:, 1:-1]
    im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
    return im[:, :, :3], cv2.cvtColor(im[:, :, 3], cv2.COLOR_GRAY2BGR)


class TabProcessor(Processor):
    # ExtractionRegionsCollection(regions={
    #   'vs': ExtractionRegions(name="vs", 1 regions),
    #   'blue_names': ExtractionRegions(name="blue_names", 6 regions),
    #   'red_names': ExtractionRegions(name="red_names", 6 regions),
    #   'player_hero': ExtractionRegions(name="player_hero", 2 regions),
    #   'stats': ExtractionRegions(name="stats", 12 regions),
    #   'medals': ExtractionRegions(name="medals", 5 regions)
    # } regions)
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))
    TEMPLATES = {
        os.path.basename(p).split(".")[0]: _load_template(imageops.imread(p, -1))
        for p in glob.glob(os.path.join(os.path.dirname(__file__), "data", "hero_icons", "*.png"))
    }
    HERO_TEMPLATE_THRESH = 100

    def __init__(self, save_name_images: bool = False):
        self.save_name_images = save_name_images

        self._last_matches: List[Optional[str]] = [None for _ in range(12)]

    def process(self, frame: Frame) -> bool:
        if not self.detect_tab(frame):
            return False

        player_name_image, player_hero_image = self.REGIONS["player_hero"].extract(frame.image)
        images = NameImages(
            blue_team=self._mask_roles_out(self.REGIONS["blue_names"].extract(frame.image)),
            red_team=self.REGIONS["red_names"].extract(frame.image),
            ult_images=self.REGIONS["ults"].extract(frame.image),
            player_name_image=player_name_image,
            player_hero_image=player_hero_image,
            hero_icons_red=self.REGIONS["hero_icons_red"].extract(frame.image),
            hero_icons_blue=self.REGIONS["hero_icons_blue"].extract(frame.image),
        )

        player_hero_text = big_noodle.ocr(player_hero_image)
        hero = textops.best_match(
            player_hero_text.lower(),
            [h[1].key for h in data.heroes.items()],
            list(data.heroes.keys()),
            threshold=3,
        )

        heroes_played = self.parse_heroes(images)
        map_text, mode_text = self.parse_map_info(frame)

        stats: Optional[Stats]
        if hero:
            stat_values = []
            for i, im in enumerate(self.REGIONS["stats"].extract(frame.image)):
                masked = self._filter_digit_components(im)
                stat_values.append(digit.ocr(masked, 1.0))

            stat_names_row_1 = [s.name for s in data.generic_stats[:3]]
            stat_names_row_2 = [s.name for s in data.generic_stats[3:]]
            hero_stat_names_row_1 = [s.name for s in data.heroes[hero].stats[0]]
            hero_stat_names_row_2 = [s.name for s in data.heroes[hero].stats[1]]
            stat_names = stat_names_row_1 + hero_stat_names_row_1 + stat_names_row_2 + hero_stat_names_row_2

            stat_parsed: Dict[str, Optional[int]] = dict(zip(stat_names, stat_values))

            if stat_parsed["objective time"] is not None:
                stat_parsed["objective time"] = textops.mmss_to_seconds(stat_parsed["objective time"])
                logger.debug(f'Transformed MMSS objective time to {stat_parsed["objective time"]}')

            stats = Stats(
                hero,
                eliminations=stat_parsed["eliminations"],
                objective_kills=stat_parsed["objective kills"],
                objective_time=stat_parsed["objective time"],
                hero_damage_done=stat_parsed["hero damage done"],
                healing_done=stat_parsed["healing done"],
                deaths=stat_parsed["deaths"],
                hero_specific_stats={
                    s.name: stat_parsed[s.name] for s in itertools.chain.from_iterable(data.heroes[hero].stats)
                },
            )
            logger.info(f"Parsed stats as {stats}")

        else:
            logger.warning(f"Could not recognise {player_hero_text} as a hero")
            stats = None

        frame.overwatch.tab_screen = TabScreen(
            map=map_text,
            mode=mode_text,
            blue_team=big_noodle.ocr_all(images.blue_team, channel="max"),
            blue_team_hero=heroes_played[6:12],
            blue_team_ults=[0 for _ in range(6)],
            red_team=big_noodle.ocr_all(images.red_team, channel="r"),
            red_team_hero=heroes_played[:6],
            player_name=big_noodle.ocr(player_name_image),
            player_hero=hero,
            stats=stats,
        )
        _draw_tab_screen(frame.debug_image, frame.overwatch.tab_screen)

        return True

    def _mask_roles_out(self, ims: List[np.ndarray]) -> List[np.ndarray]:
        """
        Mask out the role icons that appear to the left of the blue team names
        """
        r = []
        for im in ims:
            _, rank_mask = cv2.threshold(np.max(im, axis=2), 250, 255, cv2.THRESH_BINARY)
            rank_mask = cv2.erode(rank_mask, None)
            rank_mask = cv2.dilate(rank_mask, np.ones((11, 7)))
            masked = cv2.bitwise_and(im, 255 - cv2.cvtColor(rank_mask, cv2.COLOR_GRAY2BGR))
            r.append(masked)

        return r

    def _filter_digit_components(self, im: np.ndarray) -> np.ndarray:
        im = np.min(im, axis=2)
        t = imageops.otsu_thresh(im, 100, 255)
        _, mask = cv2.threshold(im, t, 255, cv2.THRESH_BINARY)

        dmask = cv2.dilate(mask, np.ones((3, 3)))
        gray = cv2.bitwise_and(im, dmask)

        # nmask = np.full_like(mask, 255)
        # labels, components = imageops.connected_components(mask)
        # for c1 in components:
        #     for c2 in components:
        #         if c1 is c2:
        #             continue
        #         if c1.x < c2.x < c1.x + c1.w or c1.x < c2.x + c2.w < c1.x + c1.w:
        #             # c1 is above/below c2
        #             nmask[labels == c1.label] = 0
        #         #     nmask[labels == c2.label] = 0
        #
        #
        # cv2.imshow('mask', np.vstack([im, mask, dmask, nmask, gray]))
        # cv2.waitKey(0)

        return gray

    def parse_heroes(self, images: NameImages) -> List[Optional[str]]:
        hero_played: List[Optional[str]] = [None for _ in range(12)]
        for i, icon in enumerate(images.hero_icons_red + images.hero_icons_blue):
            icon = cv2.resize(icon, (0, 0), fx=0.5, fy=0.5)
            last = self._last_matches[i]
            dontcheck = None
            if last:
                # check the hero this player was playing last
                dontcheck = last
                t, mask = self.TEMPLATES[last]
                match = np.min(cv2.matchTemplate(icon, t, cv2.TM_SQDIFF, mask=mask))
                if match < self.HERO_TEMPLATE_THRESH:
                    hero_played[i] = last
                else:
                    # hero has changed
                    last = None
            if not last:
                for hero_name, (t, mask) in self.TEMPLATES.items():
                    if hero_name == dontcheck:
                        # already tested
                        continue
                    match = np.min(cv2.matchTemplate(icon, t, cv2.TM_SQDIFF, mask=mask))
                    if match < self.HERO_TEMPLATE_THRESH:
                        self._last_matches[i] = hero_name
                        hero_played[i] = hero_name
                        break
        return hero_played

    def parse_map_info(self, frame: Frame) -> Tuple[Optional[str], Optional[str]]:
        map_info_image = self.REGIONS["map_info"].extract_one(frame.image)
        yellow_text = cv2.inRange(
            cv2.cvtColor(map_info_image, cv2.COLOR_BGR2HSV_FULL),
            ((30 / 360) * 255, 0.5 * 255, 0.6 * 255),
            ((45 / 360) * 255, 1.0 * 255, 1.0 * 255),
        )
        yellow_text = cv2.filter2D(yellow_text, -1, np.ones((4, 2)) / (4 * 2))
        yellow_text_left = np.argmax(np.sum(yellow_text, axis=0) / 255 > 4)
        map_image, mode_image = (
            map_info_image[:, : yellow_text_left - 20],
            map_info_image[:, yellow_text_left - 5 :],
        )
        map_text = imageops.tesser_ocr(
            np.min(map_image, axis=2), whitelist=string.ascii_uppercase + " ", scale=2, invert=True
        )
        mode_text = imageops.tesser_ocr(
            np.max(mode_image, axis=2), whitelist=string.ascii_uppercase + " ", scale=2, invert=True
        )
        if len(map_text) < 4 or len(mode_text) < 4:
            logger.warning(f"Unexpected map/mode text: {map_text} | {mode_text}")
            return None, None
        else:
            logger.debug(f"Got map={map_text}, mode={mode_text}")
            return map_text, mode_text

    VS_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "vs_template.png"), 0)

    VS_MATCH_THRESHOLD = 0.6

    def detect_tab(self, frame: Frame) -> bool:
        region = self.REGIONS["vs"].extract_one(frame.image)
        region = cv2.resize(region, (50, 50), cv2.INTER_NEAREST)
        region_gray = np.min(region, axis=2)

        # threshold of around 200, allow for flux/lower brightness settings bringing the range down
        _, thresh = cv2.threshold(region_gray, np.percentile(region_gray.ravel(), 93), 255, cv2.THRESH_BINARY)

        match = 1 - float(np.min(cv2.matchTemplate(thresh, self.VS_TEMPLATE, cv2.TM_SQDIFF_NORMED)))

        frame.overwatch.tab_match = round(match, 5)
        return match > self.VS_MATCH_THRESHOLD


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(TabProcessor(), "tab_screen", "tab_match")
