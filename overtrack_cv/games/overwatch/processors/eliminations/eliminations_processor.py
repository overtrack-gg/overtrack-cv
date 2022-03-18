import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from overtrack_cv.core import imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch.ocr import big_noodle
from overtrack_cv.games.overwatch.processors.eliminations.models import Eliminations
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


def clamp(mn, x, mx):
    return max(mn, min(x, mx))


class EliminationsProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))

    WERE_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "were.png"), 0)
    WERE_MASK = cv2.dilate(WERE_TEMPLATE, np.ones((4, 4)))

    ELIMINATED_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "eliminated.png"), 0)
    ELIMINATED_MASK = cv2.dilate(ELIMINATED_TEMPLATE, np.ones((4, 4)))

    def process(self, frame: Frame) -> bool:
        region = self.REGIONS["eliminations"].extract_one(frame.image)
        region_b = np.min(region, axis=2)
        _, thresh = cv2.threshold(region_b, 200, 255, cv2.THRESH_BINARY)

        eliminated_locations = self._get_locations(
            imageops.matchTemplate(
                thresh,
                self.ELIMINATED_TEMPLATE,
                cv2.TM_SQDIFF_NORMED,
            ),
            0.6,
            region_name="eliminated",
        )
        if not eliminated_locations:
            return False

        self._draw_locations(
            frame.debug_image,
            self.REGIONS["eliminations"].regions[0][:2],
            eliminated_locations,
            self.ELIMINATED_TEMPLATE.shape,
            "ELIMINATED",
        )

        were_locations = self._get_locations(
            imageops.matchTemplate(
                thresh,
                self.WERE_TEMPLATE,
                cv2.TM_SQDIFF_NORMED,
            ),
            0.6,
            max_matches=2,
            region_name="were",
        )
        self._draw_locations(
            frame.debug_image,
            self.REGIONS["eliminations"].regions[0][:2],
            were_locations,
            self.WERE_TEMPLATE.shape,
            "WERE",
        )

        eliminated_by_image = None
        elimination_images = []
        for ((x, y), m) in eliminated_locations:
            if were_locations:
                is_were_eliminated = min([abs(y - y2) for ((_, y2), _) in were_locations]) < 10
            else:
                is_were_eliminated = False

            if not is_were_eliminated:
                line_x = self.REGIONS["eliminations"].regions[0][0] + x
                line_w = (frame.image.shape[1] // 2 - line_x) * 2
                name_x = line_x + 135
                name_w = line_w - 132
            else:
                line_x = self.REGIONS["eliminations"].regions[0][0] + x - 125
                line_w = (frame.image.shape[1] // 2 - line_x) * 2
                name_x = line_x + 295
                name_w = line_w - 285

            line_y = self.REGIONS["eliminations"].regions[0][1] + y - 4
            line_h = 40

            line = frame.image[line_y : line_y + line_h, name_x : name_x + name_w]
            if not is_were_eliminated:
                elimination_images.append(line)
            else:
                # cv2.imshow('line', frame.image[
                #                    line_y: line_y + line_h,
                #                    name_x: name_x + name_w
                #                    ])
                # cv2.waitKey(0)
                eliminated_by_image = line

            if frame.debug_image is not None:
                cv2.rectangle(frame.debug_image, (name_x, line_y), (name_x + name_w, line_y + line_h), (0, 255, 0))

        if eliminated_by_image is not None:
            elimination_images.append(eliminated_by_image)

        eliminations = big_noodle.ocr_all(elimination_images, channel="max", height=30)
        if eliminated_by_image is not None:
            eliminated_by = eliminations[-1]
            eliminations = eliminations[:-1]
        else:
            eliminated_by = None

        frame.overwatch.eliminations = Eliminations(eliminations, eliminated_by)
        return True

    def _get_locations(
        self, match: np.ndarray, thresh: float, max_matches: int = 6, region_name: str = ""
    ) -> List[Tuple[Tuple[int, int], float]]:
        match[match == np.inf] = 0

        r = []
        for i in range(max_matches):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
            if min_val > thresh:
                logger.debug(
                    f'Rejected match {min_val:.2f} at {min_loc} in region{" " + region_name if region_name else ""}'
                )
                break
            logger.debug(
                f'Found match {min_val:.2f} at {min_loc} in region{" " + region_name if region_name else ""}'
            )

            match[
                clamp(0, min_loc[1] - 20, match.shape[0]) : clamp(0, min_loc[1] + 20, match.shape[0]),
                clamp(0, min_loc[0] - 20, match.shape[1]) : clamp(0, min_loc[0] + 20, match.shape[1]),
            ] = 1
            r.append((min_loc, min_val))

        return r

    def _draw_locations(
        self,
        debug_image: Optional[np.ndarray],
        image_offset: Tuple[int, int],
        locations: List[Tuple[Tuple[int, int], float]],
        template_shape: Tuple[int, int],
        name: str,
    ):

        if debug_image is None:
            return

        # debug_image = debug_image[
        #     image_offset[1]:,
        #     image_offset[0]:
        # ]

        for i, ((x, y), match) in enumerate(locations):
            x += image_offset[0]
            y += image_offset[1]

            cv2.rectangle(debug_image, (x, y), (x + template_shape[1], y + template_shape[0]), (0, 0, 255), 1)
            cv2.putText(
                debug_image,
                f"{(x, y)}: {name}, match={match:.3f}, index={i}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
            )


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(EliminationsProcessor(), "eliminations", show=True)

    # for p in glob.glob('C:/scratch/lines/*.png'):
    #     im = cv2.imread(p)
    #     s = big_noodle.ocr(im, channel='max', height=30, debug=False)
    #     print(s)
    #
    #     s = big_noodle_ctc.ocr(
    #         im
    #     )
    #     print(s)
    #
    #     cv2.imshow('im', im)
    #     cv2.waitKey(0)
    #     print()
