import glob
import inspect
import logging
import os
import time

import cv2
import tabulate

from overtrack_cv.frame import CurrentGame, Frame
from overtrack_cv.games.processor import Processor
from overtrack_cv.util.logging_config import config_logger
from overtrack_cv.util.prettyprint import pprint

logger = logging.getLogger(__name__)


def test_processor(
    proc: Processor,
    *fields: str,
    show=True,
    test_all=True,
    wait=True,
    warmup=True,
    game=None,
    images=None,
) -> None:
    proc.eager_load()

    import numpy as np

    if warmup:
        for _ in range(10):
            proc.run(
                Frame.create(
                    np.zeros((1080, 1920, 3), dtype=np.uint8),
                    0,
                    warmup=True,
                )
            )

    if not images:
        images = os.path.join(os.path.dirname(inspect.getfile(proc.__class__)), "samples", "*.png")

    if isinstance(images, list):
        config_logger("test_processor", logging.DEBUG, False)
        paths = images
    else:
        config_logger(os.path.basename(images), logging.DEBUG, False)
        if "*" in images:
            paths = glob.glob(images)
        elif images.endswith(".png"):
            paths = [images]
        elif images[1] == ":":
            paths = glob.glob(images + "/*.png")
        else:
            paths = glob.glob(os.path.join(images, "*"))
        paths.sort(key=lambda p: os.path.getctime(p), reverse=True)

    if test_all:
        paths += glob.glob(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "games",
                game or ".",
                "**",
                "samples",
                "*.png",
            )
        )

    for p in paths:
        time.sleep(0.01)
        print("\n\n" + "-" * 32)
        print(os.path.abspath(p))

        im = cv2.imread(p)

        ratio = im.shape[0] / im.shape[1]
        if round(ratio, 2) != 0.56:
            logger.warning(f"Ignoring {p} had invalid aspect ratio - dimensions were {im.shape}")
            continue

        im = cv2.resize(im, (1920, 1080))
        f = Frame.create(im, os.path.getctime(p), debug=True)
        f.source_image = p
        if "game_time=" in p:
            f.game_time = float(os.path.basename(p).split("=", 1)[1].rsplit(".", 1)[0])

        f.current_game = CurrentGame()

        stats = proc.run(f)
        debug_image = f.debug_image

        f.strip()

        # print(json.dumps(frameload.frames_dump(f, numpy_support=True), indent=2))

        def getval(f: Frame, n: str):
            v = f
            for p in n.split("."):
                v = getattr(v, p, None)
                if not v:
                    return None
            return v

        print(
            tabulate.tabulate(
                [("source_image", p), ("processor_result", stats.result)]  # +
                # [
                # 	(
                # 		n,
                # 		pformat(getval(f, n))
                # 	)
                # 	for n in fields
                # ]))
            )
        )
        pprint(f.timings)
        pprint(stats)
        print()

        if show:
            cv2.imshow("debug", debug_image)
            if wait == "sometimes":
                c = cv2.waitKey(0 if stats.result else 1)
            else:
                c = cv2.waitKey(0 if wait else 1)
            if c == 115:
                os.remove(p)

        print("-" * 32)
