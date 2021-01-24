import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from overtrack_cv.core import imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.apex import data
from overtrack_cv.games.apex.squad import Squad
from overtrack_cv.games.processor import Processor


def _draw_squad(debug_image: Optional[np.ndarray], squad: Squad) -> None:
    if debug_image is None:
        return
    for x, y, s in [
        (
            330,
            790,
            f"{squad.squadmate_champions_names[0]} ({np.max(squad.squadmate_champions[0]):1.4f}) - {squad.squadmate_names[0]}",
        ),
        (
            330,
            860,
            f"{squad.squadmate_champions_names[1]} ({np.max(squad.squadmate_champions[1]):1.4f}) - {squad.squadmate_names[1]}",
        ),
        (
            440,
            970,
            f"{squad.champion_name}: ({np.max(squad.champion):1.4f}) - {squad.name}",
        ),
        (
            40,
            1050,
            f"bodyshield={squad.bodyshield_level}, helmet={squad.helmet_level}, health={squad.health*100:.0f}%, shields={squad.shields*100:.0f}%",
        ),
    ]:
        cv2.putText(debug_image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)


def _load_template(name: str, large: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if large:
        image = imageops.imread(
            os.path.join(os.path.dirname(__file__), "data", "champions", "large", f"{name}.png"),
            -1,
        )
    else:
        image = imageops.imread(
            os.path.join(os.path.dirname(__file__), "data", "champions", f"{name}.png"),
            -1,
        )
    # image = cv2.resize(
    #     image,
    #     (0, 0),
    #     cv2.INTER_NEAREST,
    #     fx=0.55,
    #     fy=0.55
    # )
    image = image[5:-5, 5:-5]
    return image[:, :, :3], cv2.cvtColor(image[:, :, 3], cv2.COLOR_GRAY2BGR)


class SquadProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(
        os.path.join(os.path.dirname(__file__), "..", "data", "regions", "16_9.zip")
    )
    CHAMPIONS = [
        (c, (_load_template(c, large=False), _load_template(c, large=True))) for c in data.champions.keys()
    ]
    SPEAKER_LARGE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "speaker_lg.png"), 0)
    SPEAKER_SMALL = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "speaker_sm.png"), 0)
    ESC = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "esc.png"), 0)

    BODYSHIELD = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "bodyshield.png"), 0) > 0
    HELMET = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "helmet.png"), 0) > 0

    def eager_load(self):
        self.REGIONS.eager_load()

    def process(self, frame: Frame):
        y = frame.image_yuv[:, :, 0]

        # if we see escape menu option then don't process as positions of squad or POV has changed
        check_region = np.hstack(
            (
                y[-60:, :180],
                y[-60:, -300:],
            )
        )
        _, thresh = cv2.threshold(check_region, 180, 255, cv2.THRESH_BINARY)
        match = np.max(cv2.matchTemplate(thresh, self.ESC, cv2.TM_CCORR_NORMED))
        if match > 0.9:
            return False

        squadmate_names = self.REGIONS["squadmate_names"].extract(y)

        bodyshield, helmet = self._get_armour(frame)

        # noinspection PyTypeChecker
        frame.squad = Squad(
            name=self._ocr_playername(self.REGIONS["player_name"].extract_one(y), large=True),
            champion=self._get_template_matches(self.REGIONS["champion"].extract_one(frame.image), large=True),
            squadmate_names=(
                self._ocr_playername(squadmate_names[0], large=False),
                self._ocr_playername(squadmate_names[1], large=False),
            ),
            squadmate_champions=tuple(
                self._get_template_matches(im) for im in self.REGIONS["squadmate_champions"].extract(frame.image)
            ),
            bodyshield_level=bodyshield,
            helmet_level=helmet,
            health=self._get_health(frame),
            shields=self._get_shields(frame),
        )
        frame.squad_match = round(float(np.max(frame.squad.champion)), 5)
        if frame.squad_match > 0.9:
            self.REGIONS.draw(frame.debug_image)

        _draw_squad(frame.debug_image, frame.squad)
        return np.max(frame.squad.champion) > 0.99

    def _get_template_matches(self, im: np.ndarray, large: bool = False) -> List[float]:
        matches: List[float] = []
        for name, templates in self.CHAMPIONS:
            (template, mask) = templates[large]
            match = imageops.matchTemplate(im, template, cv2.TM_CCORR_NORMED, mask=mask)
            matches.append(round(float(np.max(match)), 4))
        return matches

    def _ocr_playername(self, image: np.ndarray, large: bool) -> Optional[str]:
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        # cv2.imshow('image', image)
        # cv2.imshow('thresh', thresh)
        # cv2.imwrite(os.path.join(os.path.dirname(__file__), 'data', 'champions', 'speaker_sm.png'), thresh)

        template = [self.SPEAKER_SMALL, self.SPEAKER_LARGE][large]
        mnv, mxv, mnl, mxl = cv2.minMaxLoc(cv2.matchTemplate(thresh, template, cv2.TM_CCORR_NORMED))
        # print(mxv, mxl)
        if mxv > 0.85:
            image = image[:, : mxl[0]]
            thresh = thresh[:, : mxl[0]]

        image = 255 - cv2.bitwise_and(image, cv2.dilate(thresh, None))

        # cv2.imshow('image_crop', image)
        # debugops.test_tesser_engines(image)
        # cv2.waitKey(0)

        s = imageops.tesser_ocr(image, engine=imageops.tesseract_lstm)
        c = np.mean(imageops.tesseract_lstm.AllWordConfidences())
        if c < 20:
            return None
        else:
            return s

    ARMOUR_COLOURS = {
        1: (74, 74, 74),
        2: (115, 68, 41),
        3: (116, 48, 83),
        4: (0, 77, 103),
    }

    def _get_armour(self, frame: Frame) -> Tuple[int, int]:
        bodyshield_region, helmet_region = self.REGIONS["equipment"].extract(frame.image)

        have_bodyshield = np.mean(np.std(bodyshield_region, axis=(0, 1))) > 25
        have_helmet = np.mean(np.std(helmet_region, axis=(0, 1))) > 25

        def _get_match(colour: np.ndarray) -> int:
            best = None
            for clas, expected in self.ARMOUR_COLOURS.items():
                match = np.linalg.norm(colour - expected)
                if not best or match < best[0]:
                    best = match, clas
            if best[0] > 50:
                best = 0, 0
            return best[1]

        if have_bodyshield:
            bodyshield_colour = np.median(bodyshield_region[self.BODYSHIELD], axis=0)
            bodyshield = _get_match(bodyshield_colour)
        else:
            bodyshield = 0

        if have_helmet:
            helmet_colour = np.median(helmet_region[self.BODYSHIELD], axis=0)
            helmet = _get_match(helmet_colour)
        else:
            helmet = 0

        return bodyshield, helmet

    def _get_health(self, frame: Frame) -> float:
        health_region = self.REGIONS["health"].extract_one(frame.image_yuv[:, :, 0])
        health_thresh = health_region > 240
        return round(float(np.sum(health_thresh) / np.prod(health_thresh.shape)), 2)

    def _get_shields(self, frame: Frame) -> float:
        shield_region = np.hstack(self.REGIONS["shields"].extract(frame.image))
        shield_thresh = np.max(shield_region, axis=2) > 180
        # cv2.imshow('shield_region', shield_region)
        # cv2.imshow('shield_thresh', shield_thresh.astype(np.uint8) * 255)
        return round(float(np.sum(shield_thresh) / np.prod(shield_thresh.shape)), 2)


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(SquadProcessor(), "squad", "squad_match")
