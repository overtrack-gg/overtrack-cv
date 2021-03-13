import logging
import os
import string
from dataclasses import fields
from typing import Optional, Tuple

import cv2
import numpy as np

from overtrack_cv.core import imageops, textops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.core.textops import mmss_to_seconds
from overtrack_cv.core.uploadable_image import lazy_upload
from overtrack_cv.frame import Frame
from overtrack_cv.games.apex.processors.match_summary import MatchSummary, XPStats
from overtrack_cv.games.apex.processors.match_summary.models import ScoreReport
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


def _draw_match_summary(debug_image: Optional[np.ndarray], summary: MatchSummary) -> None:
    if debug_image is None:
        return
    cv2.putText(
        debug_image,
        f"MatchSummary(placed={summary.placed})",
        (650, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    if summary.xp_stats:
        for i, f in enumerate(fields(XPStats)):
            cv2.putText(
                debug_image,
                f"{f.name}: {getattr(summary.xp_stats, f.name)}",
                (1000, 150 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
    elif summary.score_report:
        for i, f in enumerate(fields(ScoreReport)):
            cv2.putText(
                debug_image,
                f"{f.name}: {getattr(summary.score_report, f.name)}",
                (1000, 150 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )


class MatchSummaryProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))

    MATCH_SUMMARY_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "match_summary.png"), 0
    )
    XP_BREAKDOWN_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "xp_breakdown.png"), 0)
    SCORE_REPORT_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "score_report.png"), 0)
    REQUIRED_MATCH = 0.75

    PLACED_COLOUR = (32, 61, 238)

    XP_STATS = [
        "Won Match",
        "Top 3 Finish",
        "Time Survived",
        "Kills",
        "Damage Done",
        "Revive Ally",
        "Respawn Ally",
    ]
    XP_STATS_NORMED = [s.replace(" ", "").upper() for s in XP_STATS]
    SUBS = ["[(", "{(", "])", "})"]

    def eager_load(self):
        self.REGIONS.eager_load()

    def process(self, frame: Frame) -> bool:
        y = frame.image_yuv[:, :, 0]
        your_squad_image = self.REGIONS["match_summary"].extract_one(y)
        t, thresh = cv2.threshold(your_squad_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        match = np.max(cv2.matchTemplate(thresh, self.MATCH_SUMMARY_TEMPLATE, cv2.TM_CCORR_NORMED))
        frame.apex.match_summary_match = round(float(match), 5)
        if match >= self.REQUIRED_MATCH:
            self.REGIONS.draw(frame.debug_image)
            placed = self._get_placed(frame)

            image_title = "match_summary"
            xp_stats, score_report = None, None

            xp_breakdown_title_image = self.REGIONS["xp_breakdown"].extract_one(y)
            _, xp_breakdown_title_thresh = cv2.threshold(xp_breakdown_title_image, 150, 255, cv2.THRESH_BINARY)
            xp_breakdown_title_match = np.max(
                cv2.matchTemplate(
                    xp_breakdown_title_thresh,
                    self.XP_BREAKDOWN_TEMPLATE,
                    cv2.TM_CCORR_NORMED,
                )
            )
            if xp_breakdown_title_match > self.REQUIRED_MATCH:
                xp_stats = self._parse_xp_breakdown(y)
                image_title += "_xp_breakdown"
            else:
                score_report_title_image = self.REGIONS["score_report"].extract_one(y)
                _, score_report_title_thresh = cv2.threshold(score_report_title_image, 150, 255, cv2.THRESH_BINARY)
                score_report_title_match = np.max(
                    cv2.matchTemplate(
                        score_report_title_thresh,
                        self.SCORE_REPORT_TEMPLATE,
                        cv2.TM_CCORR_NORMED,
                    )
                )
                if score_report_title_match > self.REQUIRED_MATCH:
                    score_report = self._parse_score_report(y)
                    image_title += "_score_report"

            if placed is not None:
                frame.apex.match_summary = MatchSummary(
                    placed=placed,
                    xp_stats=xp_stats,
                    score_report=score_report,
                    image=lazy_upload(
                        image_title,
                        self.REGIONS.blank_out(frame.image),
                        frame.timestamp,
                        selection="last",
                    ),
                )
                _draw_match_summary(frame.debug_image, frame.apex.match_summary)
                return True

        return False

    def _parse_xp_breakdown(self, y: np.ndarray) -> XPStats:
        xp_breakdown_image = self.REGIONS["xp_fields"].extract_one(y)
        xp_breakdown_image = cv2.adaptiveThreshold(
            xp_breakdown_image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            63,
            -30,
        )
        lines = imageops.tesser_ocr(
            xp_breakdown_image,
            whitelist=string.ascii_letters + string.digits + "() \n",
            engine=imageops.tesseract_lstm_multiline,
        )
        for s1, s2 in self.SUBS:
            lines = lines.replace(s1, s2)

        xp_stats = XPStats()
        for line in lines.splitlines():
            stat_name, stat_value = self._parse_stat(line)
            if stat_name == "Won Match":
                xp_stats.won = True
            elif stat_name == "Top 3 Finish":
                xp_stats.top3_finish = True
            elif stat_name and stat_value is not None:
                # require stat value parsed correctly
                if stat_name == "Time Survived":
                    xp_stats.time_survived = mmss_to_seconds(stat_value)
                elif stat_name == "Kills":
                    xp_stats.kills = stat_value
                elif stat_name == "Damage Done":
                    xp_stats.damage_done = stat_value
                elif stat_name == "Revive Ally":
                    xp_stats.revive_ally = stat_value
                elif stat_name == "Respawn Ally":
                    xp_stats.respawn_ally = stat_value
        return xp_stats

    def _parse_score_report(self, y: np.ndarray) -> ScoreReport:
        rp_report_image = self.REGIONS["rp_fields"].extract_one(y)

        lines = []
        for line in range(3):
            line_im = rp_report_image[line * 40 + 5 : (line + 1) * 40 - 7, 5:]
            lines.append(imageops.tesser_ocr(line_im, engine=imageops.tesseract_lstm, invert=True, scale=2))

        score_report = ScoreReport()
        for line in lines:
            valid = False
            if ":" in line:
                stat_name, stat_value = line.lower().replace(" ", "").split(":", 1)
                if stat_name == "entrycost":
                    score_report.entry_rank = stat_value.lower()
                    valid = True
                elif stat_name == "kills":
                    try:
                        score_report.kills = int(stat_value.replace("o", "0"))
                    except ValueError:
                        logger.warning(f'Could not parse Score Report > kills: {stat_value!r}" as int')
                    else:
                        valid = True
                elif stat_name == "matchplacement":
                    stat_value = stat_value.replace("#", "")
                    try:
                        score_report.placement = int(stat_value.replace("o", "0").split("/", 1)[0])
                    except ValueError:
                        logger.warning(f'Could not parse Score Report > placement: {stat_value!r}" as placement')
                    else:
                        valid = True
            if not valid:
                logger.warning(f"Unknown line in score report: {line!r}")

        score_adjustment_image = self.REGIONS["score_adjustment"].extract_one(y)
        score_adjustment_text = imageops.tesser_ocr(
            score_adjustment_image, engine=imageops.tesseract_lstm, invert=True, scale=1
        )
        score_adjustment_text_strip = (
            textops.strip_string(score_adjustment_text, alphabet=string.digits + "RP+-")
            .replace("RP", "")
            .replace("+", "")
            .replace("-", "")
        )
        try:
            score_report.rp_adjustment = int(score_adjustment_text_strip)
        except ValueError:
            logger.warning(
                f'Could not parse Score Report > score adjustment: {score_adjustment_text!r}" as valid adjustment'
            )

        current_rp_image = self.REGIONS["current_rp"].extract_one(y)
        current_rp_text = imageops.tesser_ocr(
            current_rp_image, engine=imageops.tesseract_lstm, invert=True, scale=1
        )
        current_rp_text_strip = textops.strip_string(current_rp_text, alphabet=string.digits + "RP").replace(
            "RP", ""
        )
        try:
            score_report.current_rp = int(current_rp_text_strip)
        except ValueError:
            logger.warning(f'Could not parse Score Report > current RP: {current_rp_text!r}" as valid RP')

        return score_report

    def _parse_stat(self, line: str) -> Tuple[Optional[str], Optional[int]]:
        if len(line) > 5:
            parts = line.split("(", 1)
            if len(parts) > 1:
                stat_name_s, stat_value_s = parts[:2]
            else:
                stat_name_s, stat_value_s = line, None
            match, stat_name_normed = textops.matches_ratio(
                stat_name_s.replace(" ", "").upper(), self.XP_STATS_NORMED
            )
            if match > 0.8:
                stat_name = self.XP_STATS[self.XP_STATS_NORMED.index(stat_name_normed)]
                if stat_value_s:
                    stat_value = self._parse_stat_number(stat_value_s)
                    if stat_value is not None:
                        logger.info(f"Parsed {stat_name}={stat_value} ({line!r} ~ {match:1.2f})")
                        return stat_name, stat_value
                    else:
                        logger.info(f"Unable to parse value for {stat_name} ({line!r} ~ {match:1.2f})")
                        return stat_name, None
                else:
                    return stat_name, None
            else:
                logger.warning(f"Don't know how to parse stat {line!r}")
                return None, None
        elif line:
            logger.warning(f"Ignoring stat {line!r} - too short")
            return None, None
        else:
            return None, None

    def _parse_stat_number(self, stat_value_s: str) -> Optional[int]:
        stat_value_s = stat_value_s.upper()

        # common errors in parsing digits
        for s1, s2 in "D0", "I1", "L1":
            stat_value_s = stat_value_s.replace(s1, s2)

        # remove brackets, spaces, X (e.g. in "Kills (x3)"), time separators, commas
        stat_value_s = "".join(c for c in stat_value_s if c not in "() X:.,;|")

        try:
            return int(stat_value_s)
        except ValueError:
            return None

    def _get_placed(self, frame: Frame) -> Optional[int]:
        placed_image = self.REGIONS["squad_placed"].extract_one(frame.image).copy()
        cv2.normalize(placed_image, placed_image, 0, 255, cv2.NORM_MINMAX)
        orange = cv2.inRange(
            placed_image,
            np.array(self.PLACED_COLOUR) - 40,
            np.array(self.PLACED_COLOUR) + 40,
        )
        text = imageops.tesser_ocr(orange, whitelist=string.digits + "#")
        if text and text[0] == "#":
            try:
                placed = int(text[1:])
            except ValueError:
                logger.warning(f"Could not parse {text!r} as number")
                return None
            else:
                logger.debug(f"Parsed {text!r} as {placed}")
                if 1 <= placed <= 30:
                    return placed
                else:
                    logger.warning(f"Rejected placed={placed}")
        else:
            logger.warning(f'Rejected placed text {text!r} - did not get "#"')
            return None


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(
        "match_summary",
        MatchSummaryProcessor(),
        "match_summary",
        "match_summary_match",
        game="apex",
    )
