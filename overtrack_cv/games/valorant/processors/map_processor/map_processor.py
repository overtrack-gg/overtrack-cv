import logging
import os
import string
from typing import Dict, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from overtrack_cv.frame import Frame
from overtrack_cv.games.processor import Processor

logger = logging.getLogger("MapProcessor")


class MapProcessor(Processor):
    def process(self, frame: Frame) -> bool:
        if "warmup" in frame:
            return False

        region = frame.image[15:440, 15:440]

        def target_v(im, targetv1_0_255_128, targetv2_0_255_178, voffset_0_50_20, soffset_0_50_15):
            t1 = cv2.inRange(
                im,
                (0, 0, max(targetv1_0_255_128 - voffset_0_50_20, 0)),
                (255, soffset_0_50_15, min(targetv1_0_255_128 + voffset_0_50_20, 255)),
            )
            yield t1
            t2 = cv2.inRange(
                im,
                (0, 0, max(targetv2_0_255_178 - voffset_0_50_20, 0)),
                (255, soffset_0_50_15, min(targetv2_0_255_178 + voffset_0_50_20, 255)),
            )
            yield t2
            yield cv2.bitwise_or(t1, t2)

        # debugops.sliders(
        #     region,
        #     bgr2hsv=lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL),
        #     extract_map=target_v,
        #     erode=lambda im, f_0_15_0: cv2.erode(im, np.ones((f_0_15_0, f_0_15_0)))if f_0_15_0 else im,
        #     dilate=lambda im, f_0_15_5: cv2.dilate(im, np.ones((f_0_15_5, f_0_15_5))) if f_0_15_5 else im,
        # )

        map_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV_FULL)

        h, s, v = cv2.split(map_hsv.astype(np.float) / 255.0)
        s_match = s ** 2
        v_match = np.min(
            np.stack(
                (
                    (v - 0.49) ** 2,
                    (v - 0.69) ** 2,
                ),
                axis=2,
            ),
            axis=2,
        )

        s_match = np.clip(s_match * 100, 0, 1)
        v_match = np.clip(v_match * 100, 0, 1)

        frame.debug_image[15 : 15 + map_hsv.shape[0], 450 : 450 + map_hsv.shape[1]] = cv2.cvtColor(
            (s_match * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        frame.debug_image[15 : 15 + map_hsv.shape[0], 450 + 460 : 450 + 460 + map_hsv.shape[1]] = cv2.cvtColor(
            (v_match * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )

        return False

        V_OFF = 10
        S_OFF = 5
        map_thresh_1 = cv2.inRange(
            map_hsv,
            (0, 0, 128 - V_OFF),
            (255, S_OFF, 128 + V_OFF),
        )
        map_thresh_2 = cv2.inRange(
            map_hsv,
            (0, 0, 178 - V_OFF),
            (255, S_OFF, 178 + V_OFF),
        )
        map_thresh_c = cv2.bitwise_or(map_thresh_1, map_thresh_2)
        map_thresh = cv2.dilate(map_thresh_c, np.ones((5, 5)))

        if frame.debug_image is not None:
            frame.debug_image[15 : 15 + map_thresh.shape[0], 450 : 450 + map_thresh.shape[1]] = cv2.cvtColor(
                map_thresh, cv2.COLOR_GRAY2BGR
            )

            dist = map_hsv[:, :, 1:].astype(np.float) / 255
            dist[:, :, 1] -= 0.5
            dist2 = np.linalg.norm(dist, axis=-1)

            frame.debug_image[450 : 450 + map_thresh.shape[0], 450 : 450 + map_thresh.shape[1]] = cv2.cvtColor(
                (dist2 * (255 / 1.42)).astype(np.uint8),
                cv2.COLOR_GRAY2BGR,
            )

            cnts, _ = cv2.findContours(
                map_thresh.copy(),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            # cv2.drawContours(
            #     frame.debug_image,
            #     cnts,
            #     -1,
            #     (0, 255, 0),
            #     1,
            #     # cv2.LINE_AA,
            #     offset=(15, 15)
            # )

            intr = 255 - map_thresh
            cv2.floodFill(
                intr,
                None,
                (1, 1),
                0,
            )
            frame.debug_image[
                15 : 15 + map_thresh.shape[0], 450 + 460 : 450 + 460 + map_thresh.shape[1]
            ] = cv2.cvtColor(intr, cv2.COLOR_GRAY2BGR)
            cnts, _ = cv2.findContours(
                intr,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            # cv2.drawContours(
            #     frame.debug_image,
            #     cnts,
            #     -1,
            #     (100, 255, 100),
            #     1,
            #     # cv2.LINE_AA,
            #     offset=(15, 15)
            # )

            orb = cv2.ORB_create()
            kp = orb.detect(map_thresh, None)
            kp, des = orb.compute(map_thresh, kp)
            print(kp, des)
            frame.debug_image[15:440, 15:440] = cv2.drawKeypoints(
                frame.image[15:440, 15:440],
                kp,
                frame.debug_image[15:440, 15:440],
                color=(0, 255, 0),
                flags=0,
            )

        return False


def main():
    from overtrack_cv.util.logging_config import config_logger
    from overtrack_cv.util.test_processor import test_processor

    config_logger(os.path.basename(__file__), level=logging.DEBUG, write_to_file=False)
    proc = MapProcessor()

    # util.test_processor('weapon_kills', proc, 'valorant.killfeed', wait=True, test_all=False)
    test_processor(proc, "valorant.killfeed", wait=True, test_all=False)


if __name__ == "__main__":
    main()
