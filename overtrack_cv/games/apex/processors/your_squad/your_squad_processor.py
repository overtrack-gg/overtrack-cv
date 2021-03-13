import logging
import os
from dataclasses import replace
from typing import Optional, Union

import cv2
import numpy as np

from overtrack_cv.core import imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.core.uploadable_image import lazy_upload
from overtrack_cv.frame import Frame
from overtrack_cv.games.apex.processors.your_squad import (
    ChampionSquad,
    YourSelection,
    YourSquad,
)
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


def _draw_squad(
    debug_image: Optional[np.ndarray],
    squad: Union[YourSquad, ChampionSquad, YourSelection],
) -> None:
    if debug_image is None:
        return
    cv2.putText(
        debug_image, f"{replace(squad, images='*')}", (400, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )


class YourSquadProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))
    TEMPLATES = {
        k: imageops.imread(os.path.join(os.path.dirname(__file__), "data", k + ".png"), 0)
        for k in ["your_squad", "your_selection", "champion_squad"]
    }

    REQUIRED_MATCH = 0.95

    def __init__(self):
        self.duos = False
        self.duos_last_seen = 0

    def eager_load(self):
        self.REGIONS.eager_load()

    def process(self, frame: Frame):
        y = frame.image_yuv[:, :, 0]

        your_squad_image = self.REGIONS["your_squad"].extract_one(y)
        t, thresh = cv2.threshold(your_squad_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        match, key = imageops.match_templates(thresh, self.TEMPLATES, cv2.TM_CCORR_NORMED, self.REQUIRED_MATCH)
        frame.apex.your_squad_match = round(match, 4)
        if match < self.REQUIRED_MATCH:
            return False

        name1_trios = np.min(self.REGIONS["names"].extract_one(frame.image), axis=2)
        name1_duos = np.min(self.REGIONS["names_duos"].extract_one(frame.image), axis=2)
        # name1_thresh_value = max(np.max(name1_duos), np.max(name1_trios)) * 0.95
        name1_thresh_value = 240
        # logger.debug(f"Name thresh: {name1_thresh_value}")

        name1_trios_score = int(np.sum(name1_trios > name1_thresh_value))
        name1_duos_score = int(np.sum(name1_duos > name1_thresh_value))
        logger.debug(f"Trios name score: {name1_trios_score} vs duos name score: {name1_duos_score}")

        # self.duos = name1_duos_score and name1_duos_score > name1_trios_score
        self.duos = name1_trios_score < 100
        logger.info(f"Using duos={self.duos}")

        if key == "your_squad":
            names_region_name = "names_duos" if self.duos else "names"
            names = imageops.tesser_ocr_all(
                self.REGIONS[names_region_name].extract(y),
                engine=imageops.tesseract_lstm,
                invert=True,
            )
            frame.apex.your_squad = YourSquad(
                tuple(self._to_name(n) for n in names),
                mode="duos" if self.duos else None,
                images=lazy_upload(
                    "your_squad",
                    np.hstack(self.REGIONS[names_region_name].extract(frame.image)),
                    frame.timestamp,
                ),
            )
            self.REGIONS.draw(frame.debug_image)
            _draw_squad(frame.debug_image, frame.apex.your_squad)
        elif key == "your_selection":
            frame.apex.your_selection = YourSelection(
                name=self._to_name(
                    imageops.tesser_ocr(
                        self.REGIONS["names"].extract(y)[1],
                        engine=imageops.tesseract_lstm,
                        invert=True,
                    )
                ),
                image=lazy_upload(
                    "your_selection",
                    self.REGIONS["names"].extract(frame.image)[1],
                    frame.timestamp,
                ),
            )
            self.REGIONS.draw(frame.debug_image)
            _draw_squad(frame.debug_image, frame.apex.your_selection)
        elif key == "champion_squad":
            names_region_name = "names_duos" if self.duos else "names"
            names = imageops.tesser_ocr_all(
                self.REGIONS[names_region_name].extract(y),
                engine=imageops.tesseract_lstm,
                invert=True,
            )
            frame.apex.champion_squad = ChampionSquad(
                tuple(self._to_name(n) for n in names),
                mode="duos" if self.duos else None,
                images=lazy_upload(
                    "champion_squad",
                    np.hstack(self.REGIONS[names_region_name].extract(frame.image)),
                    frame.timestamp,
                ),
            )
            self.REGIONS.draw(frame.debug_image)
            _draw_squad(frame.debug_image, frame.apex.champion_squad)

        return True

    def _to_name(self, name_text: str) -> Optional[str]:
        for s1, s2 in "[(", "{(", "])", "})":
            name_text = name_text.replace(s1, s2)
        if len(name_text) > 3 and name_text[0] == "(" and name_text[-1] == ")":
            return name_text[1:-1].replace(" ", "").replace("(", "").replace(")", "")
        else:
            logger.warning(f"Got name {name_text!r} for player: not correctly bracketed")
            return name_text.replace(" ", "").replace("(", "").replace(")", "")


def main() -> None:
    from overtrack_cv.util.test_processor import test_processor

    test_processor(
        YourSquadProcessor(),
        "your_squad",
        "your_selection",
        "champion_squad",
        "squad_match",
        "duos_match",
        game="apex",
    )


if __name__ == "__main__":
    main()
