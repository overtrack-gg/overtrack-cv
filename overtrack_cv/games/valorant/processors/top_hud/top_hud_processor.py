import logging
import os
from typing import Optional, Sequence, Tuple, TypeVar, cast

import cv2
import numpy as np

from overtrack_cv.core import imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.processor import Processor
from overtrack_cv.games.valorant.data import agents
from overtrack_cv.games.valorant.ocr import din_next_regular_digits
from overtrack_cv.games.valorant.processors.top_hud.models import (
    FiveOFloat,
    TeamComp,
    TopHud,
)

logger = logging.getLogger("TopHudProcessor")

T = TypeVar("T")
FiveT = Tuple[T, T, T, T, T]


def cast_teams(items: Sequence[Sequence[T]]) -> Tuple[FiveT, FiveT]:
    assert len(items) == 2
    assert len(items[0]) == 5
    assert len(items[1]) == 5
    return cast(FiveT, items[0]), cast(FiveT, items[1])


HAVE_ULT_THRESHOLD = 0.6
SPIKE_THRESHOLD = 0.75


def draw_top_hud(debug_image: Optional[np.ndarray], top_hud: TopHud) -> None:
    if debug_image is None:
        return

    def strfb(*bls):
        return ", ".join("(" + ", ".join(str(e)[0] for e in bl) + ")" for bl in bls)

    def playerfield_str(val: Optional[float], t: float):
        if val is None:
            return "?"
        elif val > t:
            c = "X"
        else:
            c = "-"
        return f"{c} {val:.2f}"

    top_hud_s = (
        f"TopHud("
        f"score={top_hud.score}, "
        # f'teams={top_hud.teams}, '
        # f'has_ult={strfb(*top_hud.has_ult)}, '
        # f'has_spike={strfb(top_hud.has_spike)}'
        f")"
    )
    texts = [
        (800, 90, top_hud_s),
    ]
    for t in range(2):
        for i in range(5):
            texts.append((440 + 730 * t + 70 * i, 95, top_hud.teams[t][i] or "?"))
            texts.append(
                (440 + 730 * t + 70 * i, 112, playerfield_str(top_hud.has_ult_match[t][i], HAVE_ULT_THRESHOLD))
            )
            if t == 0:
                texts.append(
                    (440 + 730 * t + 70 * i, 130, playerfield_str(top_hud.has_spike_match[i], SPIKE_THRESHOLD))
                )

    for x, y, t in texts:
        for c, th in ((0, 0, 0), 4), ((0, 128, 255), 1):
            cv2.putText(
                debug_image,
                t,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                c,
                th,
            )


def load_agent_template(path):
    image = imageops.imread(path, -1)[3:-3, 3:-3]
    return image[:, :, :3], cv2.cvtColor(image[:, :, 3], cv2.COLOR_GRAY2BGR)


class TopHudProcessor(Processor):

    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))

    AGENT_TEMPLATES = {
        name: load_agent_template(os.path.join(os.path.dirname(__file__), "data", "agents", name.lower() + ".png"))
        for name in agents
    }
    AGENT_TEMPLATES_FLIP = {
        name: (images[0][:, ::-1], images[1][:, ::-1]) for name, images in AGENT_TEMPLATES.items()
    }
    AGENT_TEMPLATE_REQUIRED_MATCH = 0.1

    HAVE_ULT_SIGNAL = np.array([1] * 5 + [0] * 44 + [1] * 5, dtype=np.float)

    SPIKE_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "spike.png"), 0)

    def process(self, frame: Frame) -> bool:
        teams = self._parse_teams(frame)

        frame.valorant.top_hud = TopHud(
            score=self.parse_score(frame),
            teams=teams,
            has_ult_match=self._parse_ults(frame, teams),
            has_spike_match=self._parse_spike(frame, teams),
        )
        draw_top_hud(frame.debug_image, frame.valorant.top_hud)

        # self.REGIONS.draw(frame.debug_image)

        return frame.valorant.top_hud.score[0] is not None or frame.valorant.top_hud.score[1] is not None

    def parse_score(self, frame: Frame) -> Tuple[Optional[int], Optional[int]]:
        score_ims = self.REGIONS["scores"].extract(frame.image)
        score_gray = [np.min(im, axis=2) for im in score_ims]
        score_norm = [imageops.normalise(im, bottom=80, top=100) for im in score_gray]

        # debugops.normalise(score_gray[0])
        # cv2.imshow('score_ims', np.hstack(score_ims))
        # cv2.imshow('score_gray', np.hstack(score_gray))
        # cv2.imshow('score_ys_norm', np.hstack(score_norm))

        score = imageops.tesser_ocr_all(score_norm, expected_type=int, invert=True, engine=din_next_regular_digits)
        logger.debug(f"Got score={score}")
        return score[0], score[1]

    def _parse_teams(self, frame: Frame) -> Tuple[TeamComp, TeamComp]:
        agents = []
        for i, agent_im in enumerate(self.REGIONS["teams"].extract(frame.image)):
            blurlevel = cv2.Laplacian(agent_im, cv2.CV_64F).var()
            if blurlevel < 100:
                agents.append(None)
                logger.debug(f"Got agent {i}=None (blurlevel={blurlevel:.2f})")
            else:
                templates = self.AGENT_TEMPLATES
                if i > 4:
                    templates = self.AGENT_TEMPLATES_FLIP
                # cv2.imshow('agent', self.AGENT_TEMPLATES_FLIP['Raze'][0])
                match, r_agent = imageops.match_templates(
                    agent_im,
                    templates,
                    method=cv2.TM_SQDIFF_NORMED,
                    required_match=self.AGENT_TEMPLATE_REQUIRED_MATCH,
                    use_masks=True,
                    previous_match_context=(self.__class__.__name__, "_parse_teams", i),
                    # verbose=True
                )
                agent = r_agent
                if match > self.AGENT_TEMPLATE_REQUIRED_MATCH:
                    agent = None

                logger.debug(
                    f"Got agent {i}={agent} (best={r_agent}, match={match:.3f}, blurlevel={blurlevel:.1f})"
                )
                agents.append(agent)
        return cast_teams((agents[:5], agents[5:]))

    def _parse_ults(self, frame: Frame, teams: Tuple[TeamComp, TeamComp]) -> Tuple[FiveOFloat, FiveOFloat]:
        ults = []
        for i, ult_im in enumerate(self.REGIONS["has_ult"].extract(frame.image)):
            if not teams[i // 5][i % 5]:
                ults.append(None)
                continue

            matches = [0.0]
            ult_hsv = cv2.cvtColor(ult_im, cv2.COLOR_BGR2HSV_FULL)

            ult_col = np.median(ult_hsv, axis=(0,))
            ult_col = ult_col.astype(np.float)

            # The median pixel value for each channel should be the value of the "yellow"
            # Compute the abs offset from this value
            ult_col = np.abs(ult_col - np.median(ult_col, axis=(0,)))

            for c in range(3):
                # Compute the maximum (filtered with a width 5 bloxfilter) value and normalize by this
                # Check both sides of the image, as they may be different and use the lower of the two then clip the higher so it matches
                ult_col_hi = np.convolve(ult_col[:, c], [1 / 5] * 5)
                avg_diff_at_edge = min(
                    np.max(ult_col_hi[: len(ult_col_hi) // 2]), np.max(ult_col_hi[len(ult_col_hi) // 2 :])
                )
                # print(i, c, avg_diff_at_edge)
                if avg_diff_at_edge < 15:
                    # Not significant difference
                    continue

                have_ult_thresh_1d = np.clip(ult_col[:, c] / avg_diff_at_edge, 0, 1)

                # This leaves have_ult_thresh_1d as a signal [0, 1] where 0 is match to the has ult colour,
                # and 1 is match to the outside

                # Correlate this (normalizing to [-1, 1] to make the correlation normalized) with the expected width for the has ult block
                have_ult_correlation = np.correlate(
                    have_ult_thresh_1d * 2 - 1, self.HAVE_ULT_SIGNAL * 2 - 1
                ) / len(self.HAVE_ULT_SIGNAL)
                have_ult_match = np.max(have_ult_correlation)
                #
                # if not frame.get('warmup') and i == 2:
                #     import matplotlib.pyplot as plt
                #     plt.figure()
                #     plt.imshow(ult_im)
                #     plt.figure()
                #     plt.plot(have_ult_thresh_1d)
                #     plt.plot(have_ult_correlation)
                #     plt.show()

                matches.append(have_ult_match)
            have_ult_match = np.max(matches)
            logger.debug(f"Got player {i} has ult match={have_ult_match:.3f}")
            ults.append(round(float(have_ult_match), 3))

        return cast_teams((ults[:5], ults[5:]))

    def _parse_spike(self, frame: Frame, teams: Tuple[TeamComp, TeamComp]) -> FiveOFloat:
        spikes = []
        for i, ult_im in enumerate(self.REGIONS["has_spike"].extract(frame.image_yuv[:, :, 0])[:5]):
            if not teams[0][i % 5]:
                spikes.append(None)
                continue

            _, thresh = cv2.threshold(ult_im, 240, 255, cv2.THRESH_BINARY)
            match = np.max(cv2.matchTemplate(thresh, self.SPIKE_TEMPLATE, cv2.TM_CCORR_NORMED))
            #     cv2.imshow('match', thresh)
            #     cv2.waitKey(0)
            spikes.append(round(float(match), 3))

        return cast(FiveOFloat, spikes)


def main():
    from overtrack_cv.util.logging_config import config_logger
    from overtrack_cv.util.test_processor import test_processor

    config_logger(os.path.basename(__file__), level=logging.DEBUG, write_to_file=False)

    proc = TopHudProcessor()

    agent_paths = [
        os.path.join(
            "C:/Users/simon/overtrack_2/valorant_images/top_hud_agents",
            agent_name.lower() + ".png",
        )
        for agent_name in agents
    ]
    agent_frames = [agent_path for agent_path in agent_paths if os.path.exists(agent_path)]
    # util.test_processor(agent_frames, proc, 'valorant.top_hud', test_all=False, wait=True)

    # util.test_processor([
    #     r'C:/Users/simon/overtrack_2/valorant_images/ingame\00-41-617.image.png'
    # ], TopHudProcessor(), 'valorant.top_hud', test_all=False, wait=True)
    #
    test_processor(proc, "valorant.top_hud", test_all=False, wait=True)

    # paths = glob.glob("D:/overtrack/valorant_stream_client/frames/*/*/*.png", recursive=True)
    # paths = [p for p in paths if 'debug' not in p]
    # paths.sort()
    # paths = paths[::500]
    # test_processor(paths, proc, 'valorant.top_hud', game='valorant')


if __name__ == "__main__":
    main()
