import logging
import os
from typing import Optional

import cv2
import Levenshtein as levenshtein
import numpy as np

from overtrack_cv.core import imageops, textops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.core.uploadable_image import lazy_upload
from overtrack_cv.frame import Frame
from overtrack_cv.games.processor import Processor
from overtrack_cv.games.valorant.data import agents
from overtrack_cv.games.valorant.ocr import din_next_regular_digits
from overtrack_cv.games.valorant.processors.postgame.models import (
    PlayerStats,
    Postgame,
    Scoreboard,
)

logger = logging.getLogger("PostgameProcessor")


def draw_postgame(debug_image: Optional[np.ndarray], postgame: Postgame) -> None:
    if debug_image is None:
        return

    for c, t in ((0, 0, 0), 5), ((64, 0, 255), 2):
        cv2.putText(
            debug_image,
            str(postgame),
            (500, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            c,
            t,
        )


def draw_scoreboard(debug_image: Optional[np.ndarray], scoreboard: Scoreboard) -> None:
    if debug_image is None:
        return

    for i, stat in enumerate(scoreboard.player_stats):
        for c, t in ((0, 0, 0), 3), ((255, 255, 255), 1):
            cv2.putText(
                debug_image,
                str(stat),
                (335, 400 + i * 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                c,
                t,
            )


def load_agent_template(path):
    image = imageops.imread(path, -1)[5:-5, 5:-5]
    return image[:, :, :3], cv2.cvtColor(image[:, :, 3], cv2.COLOR_GRAY2BGR)


class PostgameProcessor(Processor):

    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))
    SCOREBOARD_REGIONS = ExtractionRegionsCollection(
        os.path.join(os.path.dirname(__file__), "data", "regions", "scoreboard", "16_9.zip")
    )
    RESULTS = {
        "victory": imageops.imread(os.path.join(os.path.dirname(__file__), "data", "victory.png"), 0),
        "defeat": imageops.imread(os.path.join(os.path.dirname(__file__), "data", "defeat.png"), 0),
    }
    RESULT_TEMPLATE_REQUIRED_MATCH = 0.8

    # TAB_SELECTED_TEMPLATE = np.array([0] * 50 + [3] * 73 + [5] * 24 + [3] * 73 + [0] * 50)
    # TABS = [
    #     ('summary', 240),
    #     ('scoreboard', 335)
    # ]
    SCOREBOARD_SORT_MODES = [
        textops.strip_string("Individually Sorted").upper(),
        textops.strip_string("Grouped By Team").upper(),
    ]

    AGENT_TEMPLATES = {
        name: load_agent_template(os.path.join(os.path.dirname(__file__), "data", "agents", name.lower() + ".png"))
        for name in agents
    }
    AGENT_TEMPLATE_REQUIRED_MATCH = 50

    def process(self, frame: Frame) -> bool:
        result_y = self.REGIONS["result"].extract_one(frame.image_yuv[:, :, 0])
        _, result_thresh = cv2.threshold(result_y, 220, 255, cv2.THRESH_BINARY)
        match, result = imageops.match_templates(
            result_thresh,
            self.RESULTS,
            cv2.TM_CCORR_NORMED,
            required_match=self.RESULT_TEMPLATE_REQUIRED_MATCH,
            previous_match_context=(self.__class__.__name__, "result"),
        )

        if match > self.RESULT_TEMPLATE_REQUIRED_MATCH:
            logger.debug(f"Round result is {result} with match={match}")

            score_ims = self.REGIONS["scores"].extract(frame.image)
            score_gray = [imageops.normalise(np.max(im, axis=2)) for im in score_ims]
            scores = imageops.tesser_ocr_all(
                score_gray,
                expected_type=int,
                engine=din_next_regular_digits,
                invert=True,
            )
            logger.debug(f"Round score is {scores}")

            frame.valorant.postgame = Postgame(
                victory=result == "victory",
                score=(scores[0], scores[1]),
                map=imageops.ocr_region(frame, self.REGIONS, "map"),
                game_mode=imageops.ocr_region(frame, self.REGIONS, "game_mode"),
                image=lazy_upload("postgame", self.REGIONS.blank_out(frame.image), frame.timestamp),
            )
            draw_postgame(frame.debug_image, frame.valorant.postgame)

            sort_mode_gray = np.min(
                self.SCOREBOARD_REGIONS["scoreboard_sort_mode"].extract_one(frame.image), axis=2
            )
            sort_mode_filt = 255 - imageops.normalise(sort_mode_gray, bottom=75)
            # cv2.imshow('sort_mode_gray', sort_mode_gray)
            sort_mode = imageops.tesser_ocr(sort_mode_filt, engine=imageops.tesseract_lstm)

            sort_mode_match = max(
                [
                    levenshtein.ratio(textops.strip_string(sort_mode).upper(), expected)
                    for expected in self.SCOREBOARD_SORT_MODES
                ]
            )
            logger.debug(f"Got scoreboard sort mode: {sort_mode!r} match={sort_mode_match:.2f}")

            if sort_mode_match > 0.75:
                frame.valorant.scoreboard = self._parse_scoreboard(frame)
                draw_scoreboard(frame.debug_image, frame.valorant.scoreboard)

            return True

        return False

    def _parse_scoreboard(self, frame: Frame) -> Scoreboard:
        agent_images = self.SCOREBOARD_REGIONS["agents"].extract(frame.image)

        name_images = self.SCOREBOARD_REGIONS["names"].extract(frame.image)

        stat_images = self.SCOREBOARD_REGIONS["stats"].extract(frame.image)
        stat_images_filt = [self._filter_statrow_image(im) for im in stat_images]
        stat_image_rows = [stat_images_filt[r * 8 : (r + 1) * 8] for r in range(10)]

        # cv2.imshow(
        #     'stats',
        #     np.vstack([
        #         np.hstack([self._filter_statrow_image(n)] + r)
        #         for n, r in zip(name_images, stat_image_rows)
        #     ])
        # )

        stats = []
        for i, (agent_im, name_im, stat_row) in enumerate(zip(agent_images, name_images, stat_image_rows)):
            agent_match, agent = imageops.match_templates(
                agent_im,
                self.AGENT_TEMPLATES,
                method=cv2.TM_SQDIFF,
                required_match=self.AGENT_TEMPLATE_REQUIRED_MATCH,
                use_masks=True,
                previous_match_context=(self.__class__.__name__, "scoreboard", "agent", i),
            )
            if agent_match > self.AGENT_TEMPLATE_REQUIRED_MATCH:
                agent = None

            row_bg = name_im[np.max(name_im, axis=2) < 200]
            row_color = np.median(row_bg, axis=0).astype(np.int)

            # cv2.imshow('name', self._filter_statrow_image(name_im))
            # cv2.waitKey(0)
            stat = PlayerStats(
                agent,
                imageops.tesser_ocr(
                    self._filter_statrow_image(name_im),
                    engine=imageops.tesseract_lstm,
                ),
                row_color[0] > row_color[2],
                *imageops.tesser_ocr_all(
                    stat_row,
                    expected_type=int,
                    engine=din_next_regular_digits,
                ),
            )
            stats.append(stat)
            logger.debug(
                f"Got player stats: {stat} - agent match={agent_match:.2f}, row colour={tuple(row_color)}"
            )

        return Scoreboard(
            stats,
            image=lazy_upload("scoreboard", self.SCOREBOARD_REGIONS.blank_out(frame.image), frame.timestamp),
        )

    def _filter_statrow_image(self, im):
        im_gray = np.min(im, axis=2).astype(np.float)
        bgcol = np.percentile(im_gray, 90)
        im_norm = im_gray - bgcol
        im_norm = im_norm / np.max(im_norm)
        im = 255 - np.clip(im_norm * 255, 0, 255).astype(np.uint8)
        return im


def main():
    from overtrack_cv.util.logging_config import config_logger
    from overtrack_cv.util.test_processor import test_processor

    config_logger(os.path.basename(__file__), level=logging.DEBUG, write_to_file=False)
    p = PostgameProcessor()
    test_processor(p, "valorant.postgame", "valorant.scoreboard", test_all=False)
    test_processor(p, "valorant.postgame", "valorant.scoreboard", test_all=False)


if __name__ == "__main__":
    main()
