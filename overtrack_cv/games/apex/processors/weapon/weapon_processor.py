import logging
import os
import string
from typing import List, Optional

import cv2
import numpy as np

from overtrack_cv.core import imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.apex.processors.weapon import Weapons
from overtrack_cv.games.processor import Processor

logger = logging.getLogger("WeaponProcessor")


def _draw_weapons(debug_image: Optional[np.ndarray], weapons: Weapons) -> None:
    if debug_image is None:
        return
    for inner in range(2):
        cv2.putText(
            debug_image,
            f"{weapons}",
            (500, 1070),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255) if inner else (0, 0, 0),
            1 if inner else 5,
        )


class WeaponProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "regions", "16_9.zip")
    )
    CLIP_DIGITS = [
        imageops.imread(
            os.path.join(os.path.dirname(__file__), "data", "clip_digits", f"{d}.png"),
            0,
        )
        for d in string.digits
    ]
    AMMO_DIGITS = [
        imageops.imread(
            os.path.join(os.path.dirname(__file__), "data", "ammo_digits", f"{d}.png"),
            0,
        )
        for d in string.digits
    ]

    def __init__(self):
        self.CLIP_WEIGHTS = self.AMMO_WEIGHTS = []
        for typ in "CLIP", "AMMO":
            weights = []
            for im in getattr(self, typ + "_DIGITS"):
                weight = np.zeros(im.shape, dtype=np.float)
                weight[im > 0] = 1 / np.sum(im > 0)
                weight[im == 0] = -3 / np.sum(im == 0)
                weights.append(weight)
            setattr(self, typ + "_WEIGHTS", weights)

    def eager_load(self):
        self.REGIONS.eager_load()

    def process(self, frame: Frame):
        y = cv2.cvtColor(frame.image, cv2.COLOR_BGR2YUV)[:, :, 0]

        weapon_images = self.REGIONS["weapon_names"].extract(y)
        weapon_images = [255 - imageops.normalise(i) for i in weapon_images]

        weapon_names = imageops.tesser_ocr_all(
            weapon_images,
            whitelist=string.ascii_uppercase,
            engine=imageops.tesseract_lstm,
            scale=2,
        )

        selected_weapons_regions = self.REGIONS["selected_weapon_tell"].extract(frame.image)
        selected_weapons_colours = [np.median(r, axis=(0, 1)) for r in selected_weapons_regions]

        def thresh_clip(im):
            im = np.max(im, axis=2)
            threshim = np.tile(im[:, 0], (im.shape[1], 1)).T
            im = cv2.subtract(im, threshim)
            tim = im > 20
            return tim

        frame.apex.weapons = Weapons(
            weapon_names,
            selected_weapons=(
                (
                    int(selected_weapons_colours[0][0]),
                    int(selected_weapons_colours[0][1]),
                    int(selected_weapons_colours[0][2]),
                ),
                (
                    int(selected_weapons_colours[1][0]),
                    int(selected_weapons_colours[1][1]),
                    int(selected_weapons_colours[1][2]),
                ),
            ),
            clip=self._ocr_digits([im > 200 for im in self.REGIONS["clip"].extract(y)], self.CLIP_WEIGHTS),
            ammo=self._ocr_digits(
                [thresh_clip(im) for im in self.REGIONS["ammo"].extract(frame.image)],
                self.AMMO_WEIGHTS,
            ),
        )

        # self.REGIONS.draw(frame.debug_image)
        _draw_weapons(frame.debug_image, frame.apex.weapons)

        return frame.apex.weapons.selected_weapons is not None

    def _ocr_digits(self, ims: List[np.ndarray], weights: List[np.ndarray]) -> Optional[int]:
        digits = []
        # cv2.imshow('digits', np.hstack(ims).astype(np.uint8) * 255)
        for i, im in enumerate(ims):
            if np.sum(im) < 50:
                continue
            best = None
            for d, w in enumerate(weights):
                score = np.sum(np.multiply(im, w))
                if score > 0.95:
                    digits.append(str(d))
                    break
                if not best or score > best[0]:
                    best = score, d
            else:
                if best[0] > 0.1:
                    digits.append(str(best[1]))
                else:
                    logger.warning(f"Unable to OCR clip digit {i} - best match: {best[1]} @ {best[0]:.2f}")
        if not digits:
            return None
        try:
            return int("".join(digits))
        except Exception as e:
            logger.warning(f"Unable to parse OCR of clip: {digits!r}: {e}")
            return None


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(
        WeaponProcessor(),
        "weapons",
        game="apex",
    )

    # util.test_processor('weapons', WeaponProcessor(), 'weapons', game='apex')
    # util.test_processor(WeaponProcessor(), 'weapons')

    # proc = WeaponProcessor()
    #
    # from overtrack_cv.source.shmem.shmem_capture import SharedMemoryCapture
    #
    # cap = SharedMemoryCapture(debug_frames=True)
    # cap.start()
    #
    # while True:
    #     frame = cap.get()
    #     if frame and frame.valid:
    #         proc.process(frame)
    #
    #         cv2.imshow("debug", frame.debug_image)
    #         cv2.waitKey(1)
