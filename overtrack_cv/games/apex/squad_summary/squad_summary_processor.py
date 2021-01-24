import dataclasses
import logging
import os
import pprint

import cv2
import numpy as np

from overtrack_cv.core import imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.core.textops import mmss_to_seconds
from overtrack_cv.core.uploadable_image import lazy_upload
from overtrack_cv.frame import Frame
from overtrack_cv.games.apex import ocr
from overtrack_cv.games.apex.squad_summary.models import *
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


def _draw_squad_summary(debug_image: Optional[np.ndarray], squad_summary: SquadSummary) -> None:
    if debug_image is None:
        return
    cv2.putText(
        debug_image,
        f"{dataclasses.replace(squad_summary, player_stats=None, image=None)}",
        (100, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    for i, stats in enumerate(squad_summary.player_stats):
        for t, c in (3, (0, 0, 0)), (1, (0, 255, 0)):
            cv2.putText(
                debug_image,
                f"{stats}",
                (i * 100, 800 + 30 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                c,
                t,
            )


def _draw_match(debug_image: Optional[np.ndarray], match: float) -> None:
    if debug_image is None:
        return


class SquadSummaryProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))

    TEMPLATES = {
        k: imageops.imread(os.path.join(os.path.dirname(__file__), "data", k + ".png"), 0)
        for k in ["squad_eliminated", "champions_of_the_arena", "match_summary"]
    }

    REQUIRED_MATCH = 0.75

    def eager_load(self):
        self.REGIONS.eager_load()

    def process(self, frame: Frame) -> bool:
        y = frame.image_yuv[:, :, 0]
        champions_eliminated = self.REGIONS["champions_eliminated"].extract_one(y)
        t, thresh = cv2.threshold(champions_eliminated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # cv2.imshow('thresh', thresh)

        match, key = imageops.match_templates(thresh, self.TEMPLATES, cv2.TM_CCORR_NORMED, self.REQUIRED_MATCH)
        frame.squad_summary_match = round(match, 4)
        if match > self.REQUIRED_MATCH:
            champions = key in ["champions_of_the_arena"]

            duos_empty_area = self.REGIONS["duos_empty_area"].extract_one(frame.image_yuv[:, :, 0])
            duos_sum = np.sum(duos_empty_area > 100)
            duos = duos_sum < 100
            logger.debug(f"Got duos_sum={duos_sum} => duos={duos}")

            shunt = 0
            if duos:
                duos_shunt_area = self.REGIONS["duos_shunt_area"].extract_one(frame.image_yuv[:, :, 0])
                duos_shunt_sum = np.sum(duos_shunt_area > 100)
                duos_shunt = duos_shunt_sum < 100
                logger.debug(f"Got duos_shunt_sum={duos_shunt_sum} => duos_shunt={duos_shunt}")
                if duos_shunt:
                    shunt = 50

            frame.squad_summary = SquadSummary(
                champions=champions,
                placed=self._process_yellowtext(self.REGIONS["placed"].extract_one(frame.image)),
                squad_kills=self._process_yellowtext(self.REGIONS["squad_kills"].extract_one(frame.image)),
                player_stats=self._process_player_stats(y, duos, shunt),
                elite=False,
                mode="duos" if duos else None,
                image=lazy_upload(
                    "squad_summary",
                    self.REGIONS.blank_out(frame.image),
                    frame.timestamp,
                    selection="last",
                ),
            )
            self.REGIONS.draw(frame.debug_image)
            _draw_squad_summary(frame.debug_image, frame.squad_summary)
            return True

        return False

    def _process_yellowtext(self, image: np.ndarray) -> Optional[int]:
        # mask out only yellow text (digits)
        yellow = cv2.inRange(image, (0, 40, 150), (90, 230, 255))
        yellow = cv2.dilate(yellow, None)
        yellowtext_image = cv2.bitwise_and(image, cv2.cvtColor(yellow, cv2.COLOR_GRAY2BGR))
        yellowtext_image_g = np.max(yellowtext_image, axis=2)
        yellowtext_image_g = cv2.erode(yellowtext_image_g, np.ones((2, 2)))

        # from overtrack_cv.util import  ps
        # cv2.imshow('yellow', yellow)
        # cv2.imshow('squad_kills_image', squad_kills_image)
        # cv2.imshow('squad_kills_image_g', squad_kills_image_g)
        # debugops.test_tesser_engines(squad_kills_image_g, scale=4)

        # from overtrack_cv.util import debugops
        # debugops.test_tesser_engines(yellowtext_image_g)

        text = imageops.tesser_ocr(
            yellowtext_image_g,
            engine=imageops.tesseract_lstm,
            scale=4,
            blur=4,
            invert=True,
        )
        otext = text
        text = text.upper()
        for s1, s2 in "|1", "I1", "L1", "O0", "S5", "B6":
            text = text.replace(s1, s2)
        for hashchar in "#H":
            text = text.replace(hashchar, "")
        logger.info(f"Got text={otext} -> {text}")

        try:
            return int(text)
        except ValueError:
            logger.warning(f"Could not parse {text!r} as int")
            return None

    def _process_player_stats(self, y: np.ndarray, duos: bool = False, shunt: int = 0) -> Tuple[PlayerStats, ...]:
        prefix = "duos_" if duos else ""

        name_images = self.REGIONS[prefix + "names"].shunt(x=shunt).extract(y)
        names = []
        for im in name_images:
            # self._mask_components_touching_edges(im)
            im = 255 - cv2.bitwise_and(
                im,
                cv2.dilate(
                    cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
                    None,
                ),
            )
            im = cv2.resize(im, (0, 0), fx=2, fy=2)
            im = cv2.GaussianBlur(im, (0, 0), 1)

            # cv2.imshow('name', im)
            # cv2.waitKey(0)
            name = imageops.tesser_ocr(
                im,
                engine=imageops.tesseract_lstm,
            ).replace(" ", "")
            match = np.mean(imageops.tesseract_lstm.AllWordConfidences())
            logger.info(f"Got name {name!r} ~ {match:1.2f}")
            if match < 0.75:
                name = imageops.tesser_ocr(
                    im,
                    engine=imageops.tesseract_only,
                )
                logger.info(f"Using {name!r} instead")
            names.append(name)

        stat_images = self.REGIONS[prefix + "stats"].shunt(x=shunt).extract(y)

        # for im in stat_images:
        #     self._mask_components_touching_edges(im)

        stats = imageops.tesser_ocr_all(
            stat_images,
            engine=ocr.tesseract_ttlakes_digits_specials,
        )

        for i in range(len(stats)):
            value = stats[i]
            if value:
                value = value.lower().replace(" ", "")
                for c1, c2 in "l1", "i1", "o0", (":", ""):
                    value = value.replace(c1, c2)
            if 6 <= i <= 8:
                # survival time
                if stats[i] is not None:
                    seconds_s = value.replace(":", "")
                    try:
                        seconds = int(seconds_s)
                    except ValueError as e:
                        logger.warning(f'Could not parse "{stats[i]}" as int: {e}')
                        seconds = None
                    else:
                        seconds = mmss_to_seconds(seconds)
                        logger.info(f"MM:SS {stats[i]} -> {seconds}")
                    stats[i] = seconds
            else:
                try:
                    stats[i] = int(value)
                except ValueError as e:
                    logger.warning(f'Could not parse {value!r} as int" {e}')
                    stats[i] = None

        # typing: ignore
        # noinspection PyTypeChecker
        count = 3 if not duos else 2
        r = tuple([PlayerStats(names[i], *stats[i::count]) for i in range(count)])
        logger.info(f"Got {pprint.pformat(r)}")
        return r

    def _mask_components_touching_edges(self, im: np.ndarray, threshold=100) -> bool:
        masked = False
        _, t = cv2.threshold(im, threshold, 255, cv2.THRESH_BINARY)
        mask, components = imageops.connected_components(t)
        for c in components[1:]:
            if c.y <= 1 or c.y + c.h >= im.shape[0] - 1:
                mask = (mask != c.label).astype(np.uint8) * 255
                mask = cv2.erode(mask, None)
                im[:] = cv2.bitwise_and(im, mask)
                masked = c.area > 50
        return masked


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(
        "squad_summary",
        SquadSummaryProcessor(),
        "squad_summary",
        "squad_summary_match",
        game="apex",
    )
