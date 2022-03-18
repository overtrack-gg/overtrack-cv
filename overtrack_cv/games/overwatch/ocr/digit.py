import logging
import os
import string
import time
from typing import List, Optional, Sequence

import cv2
import numpy as np

from overtrack_cv.core import imageops
from overtrack_cv.util.logging_config import config_logger

logger = logging.getLogger(__name__)


class Classifier:
    IMSIZE = (20, 16)
    CHARACTERS = np.array(list(string.digits + "_"))

    _instance = None

    @classmethod
    def get_instance(cls, use_tflite: bool = True) -> "Classifier":
        if cls._instance:
            return cls._instance
        cls._instance = cls(use_tflite)
        return cls._instance

    def __init__(self, use_tflite: bool = True):
        if use_tflite:
            from overtrack_cv.core.tflite import TFLiteModel

            self.model = TFLiteModel(
                os.path.join(os.path.dirname(__file__), "data", "tab_digit.tflite"),
            )
        else:
            from overtrack_cv.core.tf import load_model

            self.model = load_model(
                os.path.join(os.path.dirname(__file__), "data", "tab_digit"),
            )

    def classify(self, images: List[np.ndarray], scale: float = 1.0, debug: bool = False) -> List[Optional[int]]:
        if not len(images):
            return []

        digit_images = []
        digit_buckets = []

        logger.debug(f'Parsing images {", ".join("{1}*{0}".format(*i.shape) for i in images)}')

        for index, im in enumerate(images):
            _, im = cv2.threshold(im, 150, 255, cv2.BORDER_CONSTANT)
            labels, components = imageops.connected_components(im, connectivity=4)

            for c in sorted(components[1:], key=lambda c: c.x):
                num_overlapping = (
                    sum(
                        (c.x <= o.x <= c.x + c.w or c.x <= o.x + o.w <= c.x + c.w)
                        and o.area > 5
                        and o.h * scale > 7
                        for o in components[1:]
                    )
                    - 1
                )
                if num_overlapping:
                    # percent sign
                    continue
                if not 4 < c.w * scale < 18:
                    # too wide (could just be comma)
                    if c.area > 100:
                        logger.warning(f"Found bad digit component {c} (incorrect width)")
                    continue
                if not 15 < c.h * scale < 19:
                    logger.warning(f"Found bad digit component {c} (incorrect height)")
                    continue

                digit_image = (labels[c.y : c.y + c.h, c.x : c.x + c.w] == c.label).astype(np.uint8) * 255
                digit_image = cv2.resize(digit_image, (0, 0), fx=scale, fy=scale)[
                    : self.IMSIZE[0] - 2, : self.IMSIZE[1] - 1
                ]
                digit_image = cv2.copyMakeBorder(
                    digit_image,
                    2,
                    self.IMSIZE[0] - (digit_image.shape[0] + 2),
                    1,
                    self.IMSIZE[1] - (digit_image.shape[1] + 1),
                    cv2.BORDER_CONSTANT,
                )
                digit_images.append(digit_image)
                digit_buckets.append(index)

        if not len(digit_images):
            logger.warning("Did not get any digits to classify")
            return [None for _ in images]

        probs = self.model.predict(np.array(digit_images).astype(np.float32))
        parses = np.argmax(probs, axis=1)
        str_results: List[List[str]] = [[] for _ in images]
        p_groups: List[List[float]] = [[] for _ in images]
        for stat_bucket, clas, p in zip(digit_buckets, parses, probs):
            str_results[stat_bucket].append(str(clas))
            p_groups[stat_bucket].append(np.max(p))

        rstr, pstr = [], []
        for r, ps in zip(str_results, p_groups):
            rstr.append("[ " + "".join(d + "   " for d in r) + " ]")
            pstr.append("[ " + "".join(f"{p:1.1f} " for p in ps) + " ]")

        logger.debug(f'OCR RESULT: {" ".join(rstr)}')
        logger.debug(f'OCR PROBS:  {" ".join(pstr)}')

        results: List[Optional[int]] = []
        for digit_str in str_results:
            # '_' is a valid output for each "digit", meaning *could not classify*
            # Maybe we want an option to ignore instead of error on these tokens
            try:
                results.append(int("".join(digit_str)))
            except ValueError:
                logger.warning(f"Could not parse {digit_str} as a number - ignoring")
                results.append(None)

        return results


def ocr_images(images: Sequence[np.ndarray], scale: float = 1.0) -> List[Optional[int]]:
    instance = Classifier.get_instance()
    t0 = time.perf_counter()
    r = instance.classify([np.min(im, axis=2) if len(im.shape) == 3 else im for im in images], scale=scale)
    t1 = time.perf_counter()
    logger.debug(f"OCR took {(t1 - t0)*1000:1.2f}ms")
    return r


def ocr(image: np.ndarray, scale: float = 1.0) -> Optional[int]:
    return ocr_images([image], scale=scale)[0]


def convert_tflite_model():
    import tensorflow as tf

    c = Classifier(use_tflite=False)
    c.model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(c.model)
    tflite_model = converter.convert()

    with open(os.path.join(os.path.dirname(__file__), "data", "tab_digit.tflite"), "wb") as f:
        f.write(tflite_model)


def main() -> None:
    convert_tflite_model()

    config_logger("digit", logging.DEBUG, False)

    # from tensorflow.python.keras import Input
    # from tensorflow.python.keras.layers import Reshape
    # from tensorflow.python.keras import Model
    # from tensorflow.python.keras.layers import Layer
    # from tensorflow.python.keras.layers import Dense
    # from tensorflow.python.keras.saving import export_saved_model
    #
    # image = Input(shape=(Classifier.IMSIZE[0], Classifier.IMSIZE[1]), name='images', dtype=tf.float32)
    # image_r = Reshape((np.prod(Classifier.IMSIZE), ))(image)
    #
    # logits = Dense(
    #     len(Classifier.CHARACTERS),
    #     activation='softmax',
    #     name='fc1'
    # )(image_r)
    #
    # model = Model(
    #     inputs=image,
    #     outputs=logits
    # )
    # model.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy'
    # )
    #
    # model.summary()
    #
    # p = os.path.join(os.path.dirname(__file__), 'data', 'tab_ocr.npz')
    # restore_dict = dict(np.load(p))
    # print({k: v.shape for k, v in restore_dict.items()})
    # l: Layer = model.layers[2]
    # print(l, [w.shape for w in l.get_weights()])
    # l.set_weights([restore_dict['tab_ocr/W_fc_:0'], restore_dict['tab_ocr/b_fc_:0']])
    #
    # os.makedirs('./data/tab', exist_ok=True)
    # export_saved_model(model, f'./data/tab_digit', serving_only=True)

    image = cv2.imread("C:/tmp/test.png")
    for _ in range(10):
        print(ocr(image))


if __name__ == "__main__":
    main()
