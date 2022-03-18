import logging
import os
import string
import time
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np

from overtrack_cv.core import imageops
from overtrack_cv.core.imageops import ConnectedComponent
from overtrack_cv.util.logging_config import config_logger, intermittent_log

logger = logging.getLogger(__name__)

BIG_NOODLE_DIGITSUBS = "O0", "D0", "I1", "L1", "B8", "A8", "S5"


def big_noodle_digitsub(s: str) -> str:
    for c1, c2 in BIG_NOODLE_DIGITSUBS:
        s = s.replace(c1, c2)
    return s


class Classifier:
    # HEIGHT = 25
    # WIDTH = int(HEIGHT * 0.9 + 0.5)
    CHARACTERS = np.array(list(string.digits[1:] + string.ascii_uppercase))
    # DROPOUT_RATE = 0.1
    # CONV1_SIZE = (HEIGHT - 2, WIDTH - 2)

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

            self.model = TFLiteModel(os.path.join(os.path.dirname(__file__), "data", "big_noodle.tflite"))
        else:
            from overtrack_cv.core.tf import load_model

            self.model = load_model(os.path.join(os.path.dirname(__file__), "data", "big_noodle"))

    def classify(
        self, images: List[np.ndarray], whitelist: Optional[str] = None, debug: bool = False
    ) -> List[str]:
        if not len(images):
            return []
        norm_images = []
        for im in images:
            p = np.percentile(im, 95)
            im = im.astype(np.float) * (255.0 / p)
            im = np.clip(im, 0, 255) / 255.0 - 0.5
            norm_images.append(im)

        probs = self.model.predict(np.array(norm_images).astype(np.float32))
        result = np.argmax(probs, axis=1)
        match = np.max(probs, axis=1)

        logger.debug("OCR RESULT [" + "    ".join(self.CHARACTERS[result]) + "   ]")
        logger.debug("OCR PROBS  [" + " ".join(f"{p :1.2f}" for p in match) + "]")

        if whitelist:
            mask = np.array([c in whitelist for c in self.CHARACTERS], dtype=np.float)
            probs = probs * mask
            result = np.argmax(probs, axis=1)
            match = np.max(probs, axis=1)

            logger.debug(f'Using mask {"".join(str(int(m)) for m in mask)} for whitelist {whitelist}')
            logger.debug("OCR RESULT [" + "    ".join(self.CHARACTERS[result]) + "   ]")
            logger.debug("OCR PROBS  [" + " ".join(f"{p :1.2f}" for p in match) + "]")

        r = []
        for i, (c, p) in enumerate(zip(result, match)):
            if p < 0.1 and not whitelist:
                cstr = "".join(self.CHARACTERS[result])
                logger.warning(f"Ignoring char {i} in {cstr!r} - got p={p:1.2f}")
            else:
                r.append(self.CHARACTERS[c])

        return r


def to_gray(image: np.ndarray, channel: str = None, debug: bool = False) -> np.ndarray:
    if len(image.shape) == 2:
        if channel:
            raise ValueError(f"Cannot convert gray image to gray using { channel }")
        else:
            # image already gray
            pass
    else:
        if not channel or channel == "grey" or channel == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channel in "bgr":
            channel_index = "bgr".index(channel)
            image = image[:, :, channel_index]
        elif channel in "sv":
            channel_index = "hsv".index(channel)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
            image = image_hsv[:, :, channel_index]
        elif channel == "max":
            image = np.max(image, axis=2)
        elif channel == "min":
            image = np.min(image, axis=2)
        else:
            raise ValueError(f"Don't know how to convert BGR image to gray using { channel }")

    return image


def segment(
    gray_image: np.ndarray,
    segmentation: Optional[str] = "connected_components",
    threshold: Optional[str] = "otsu_above_mean",
    min_area: float = 10,
    height: int = None,
    multiline=False,
    debug: bool = False,
) -> List[np.ndarray]:
    segments = []

    # TODO: implement simpler/faster segmentation
    if threshold is None:
        thresh = gray_image
    elif isinstance(threshold, np.ndarray):
        thresh = threshold
    elif isinstance(threshold, int) and 0 <= threshold <= 255:
        _, thresh = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    elif isinstance(threshold, str) and threshold.startswith("otsu"):
        if threshold == "otsu_above_mean":
            thresh = imageops.otsu_thresh_lb_fraction(gray_image, 1)
        elif threshold == "otsu":
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            raise ValueError(f"Don\t know how to threshold image using { threshold }")
    else:
        raise ValueError(f"Don\t know how to threshold image using { threshold }")

    if segmentation == "connected_components":
        labels, components = imageops.connected_components(thresh)

        # TODO: estimate the size of the characters, and discard things that don't match
        components = [c for c in components[1:] if c.area > min_area]
        if height:
            components = [c for c in components if height * 0.9 < c.h < height * 1.1]

        if not len(components):
            intermittent_log(logger, "Could not find any characters", frequency=30)
            return []

        if multiline:
            lines: Dict[ConnectedComponent, int] = {}
            comps = defaultdict(list)
            for c in sorted(components, key=lambda c: c.y):
                for other_c, other_line in lines.items():
                    if other_c.y < c.y + c.h and other_c.y + other_c.h > c.y:
                        lines[c] = lines[other_c]
                        break
                else:
                    lines[c] = max(lines.values()) + 1 if lines else 0
                comps[lines[c]].append(c)

            components = sorted(components, key=lambda c: (lines[c], c.x))
        else:
            components = sorted(components, key=lambda c: c.x)

        average_top = int(np.median([c.y for c in components]))
        average_height = int(np.median([c.h for c in components]))
        # top = Counter([y for (x, y, w, h, a) in stats[1:]]).most_common(1)[0][0]
        # height = Counter([h for (x, y, w, h, a) in stats[1:]]).most_common(1)[0][0]

        border_size = average_height * 0.05
        height_tolerance = average_height * 0.2

        # I is approx 40% as wide as it is high
        # 1 can be 30%
        min_width = average_height * 0.2
        # M is approx 80% as wide as it is high
        max_width = average_height * 0.9

        top = round(max(0.0, average_top - border_size))
        if not height:
            height = round(min(average_height + border_size, gray_image.shape[0]))

        for component in components:
            if abs(min(component.h, gray_image.shape[0] - component.y) - height) > height_tolerance:
                logger.debug(f"Found component with height={component.h} but expected height={height}")
                continue

            if not min_width < component.w < max_width:
                logger.debug(
                    f"Found component with width={component.w} - expected range was [{min_width :1.1f}, {max_width :1.1f}]"
                )
                continue

            if multiline:
                ttop = int(component.y - border_size)
                y1 = ttop
                y2 = ttop + height
            else:
                y1 = top
                y2 = top + height

            x1 = round(max(0.0, component.x - border_size))
            x2 = round(min(component.x + component.w + border_size, gray_image.shape[1]))

            if y2 > gray_image.shape[0]:
                logger.debug(
                    f"Found component with y={component.y}, using height={height} would take this outside the image"
                )
                continue
            elif y2 - y1 < 2 or x2 - x1 < 2:
                logger.debug(f"Found component with height={y2 - y1}, width={x2 - x1} - ignoring")
                continue
            elif y1 < 0 or x1 < 0:
                logger.debug(f"Found component with left={x1}, top={y1} - ignoring")
                continue

            mask = (labels[y1:y2, x1:x2] == component.label).astype(np.uint8) * 255
            mask = cv2.dilate(mask, np.ones((3, 3)))
            character = cv2.bitwise_and(gray_image[y1:y2, x1:x2], mask)

            segments.append(character)

        if debug:
            print("-" * 25)
            print(f"average_top: {average_top}")
            print(f"average_height: {average_height}")
            print(f"border_size: {border_size}")
            print(f"height_tolerance: {height_tolerance}")
            print(f"min_width: {min_width}")
            print(f"max_width: {max_width}")
            print(f"top: {top}")
            print(f"height: {height}")

            import matplotlib.pyplot as plt

            f, (figs1, figs2) = plt.subplots(2, 2)

            ax = figs1[0]
            ax.imshow(gray_image, interpolation="none")
            ax.set_title("segment image")

            ax = figs1[1]
            ax.imshow(thresh, interpolation="none")
            ax.set_title("segment thresh")

            ax = figs2[0]
            ax.imshow(labels, interpolation="none")
            ax.set_title("segment components")

            ax = figs2[1]
            ax.imshow(np.hstack(segments) if len(segments) else np.zeros((1, 1)), interpolation="none")
            ax.set_title("segments")

            plt.show()
    else:
        raise ValueError(f"Don't know how to segment using {segmentation}")

    return segments


def resize_segments(segments: List[np.ndarray], height: int = 25) -> List[np.ndarray]:
    width = int(height * 0.9 + 0.5)
    r = []
    for im in segments:
        s = height / im.shape[0]
        im = cv2.resize(im, (int(im.shape[1] * s), height), interpolation=cv2.INTER_LINEAR)[:, :width]
        im = cv2.copyMakeBorder(im, 0, 0, 0, width - im.shape[1], cv2.BORDER_CONSTANT)
        r.append(im)
    return r


def ocr(
    image: np.ndarray,
    channel: Optional[str] = "gray",
    segmentation: Optional[str] = "connected_components",
    threshold: Optional[str] = "otsu_above_mean",
    whitelist: Optional[str] = None,
    min_area: int = 25,
    height: Optional[int] = None,
    multiline: bool = False,
    invert: bool = False,
    debug: bool = False,
) -> str:

    t0 = time.perf_counter()

    if np.prod(image.shape) == 0:
        logger.warning(f"Trying to OCR image with shape {image.shape}")
        return ""

    if len(image.shape) == 3:
        image = to_gray(image, channel=channel, debug=debug)
    else:
        image = image

    if invert:
        image = 255 - image

    segments = segment(
        image,
        height=height,
        segmentation=segmentation,
        threshold=threshold,
        min_area=min_area,
        multiline=multiline,
        debug=debug,
    )

    logger.debug(
        f"Parsing {image.shape[1]}*{image.shape[0]} image - got {len(segments)} segments of heights {[s.shape[0] for s in segments]}"
    )

    segments_sized = resize_segments(segments, height=25)
    classifier = Classifier.get_instance()

    s = "".join(classifier.classify(segments_sized, whitelist=whitelist, debug=debug))

    t1 = time.perf_counter()
    logger.debug(f"OCR took {(t1 - t0) * 1000:1.2f}ms")

    return s


def ocr_int(
    image: np.ndarray,
    channel: Optional[str] = "gray",
    segmentation: Optional[str] = "connected_components",
    threshold: Optional[str] = "otsu_above_mean",
    whitelist: Optional[str] = None,
    min_area: int = 25,
    height: Optional[int] = None,
    debug: bool = False,
) -> Optional[int]:

    if whitelist is None:
        whitelist = string.digits + "OSAI"

    text = ocr(
        image=image,
        channel=channel,
        segmentation=segmentation,
        threshold=threshold,
        whitelist=whitelist,
        min_area=min_area,
        height=height,
        debug=debug,
    )

    # TODO: use whitelist chars for classification and choose the best of the possible chars
    old_text = text
    text = big_noodle_digitsub(text)
    logger.debug(f"Trying to parse {old_text!r} as int, converting to {text}")
    try:
        return int(text)
    except Exception as e:
        logger.warning(f"Got exception interpreting {text!r} as int - {e}")
        return None


def ocr_all(
    images: List[np.ndarray],
    channel: Optional[str] = "gray",
    segmentation: Optional[str] = "connected_components",
    threshold: Optional[str] = "otsu_above_mean",
    whitelist: Optional[str] = None,
    min_area: int = 25,
    height: Optional[int] = None,
    invert: bool = False,
    debug: bool = False,
) -> List[str]:
    return [
        ocr(
            image=image,
            channel=channel,
            segmentation=segmentation,
            threshold=threshold,
            whitelist=whitelist,
            min_area=min_area,
            height=height,
            invert=invert,
            debug=debug,
        )
        for image in images
    ]


def ocr_all_int(
    images: List[np.ndarray],
    channel: Optional[str] = "gray",
    segmentation: Optional[str] = "connected_components",
    threshold: Optional[str] = "otsu_above_mean",
    whitelist: Optional[str] = None,
    min_area: int = 25,
    height: Optional[int] = None,
    debug: bool = False,
) -> List[Optional[int]]:
    return [
        ocr_int(
            image=image,
            channel=channel,
            segmentation=segmentation,
            threshold=threshold,
            whitelist=whitelist,
            min_area=min_area,
            height=height,
            debug=debug,
        )
        for image in images
    ]


def convert_tflite_model():
    import tensorflow as tf

    c = Classifier(use_tflite=False)
    c.model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(c.model)
    tflite_model = converter.convert()

    with open(os.path.join(os.path.dirname(__file__), "data", "big_noodle.tflite"), "wb") as f:
        f.write(tflite_model)


def main() -> None:
    convert_tflite_model()

    config_logger("big_noodle", logging.DEBUG, False)

    # from tensorflow.python.keras import Input
    # from tensorflow.python.keras.layers import Reshape, Conv2D
    # from tensorflow.python.keras import Model
    # from tensorflow.python.keras.layers import Activation
    # from overtrack.util.custom_layers import MaxAlongDims
    # from tensorflow.python.keras.layers import Layer
    # from tensorflow.python.keras.saving import export_saved_model
    #
    # image = Input(shape=(Classifier.HEIGHT, Classifier.WIDTH), name='images', dtype=tf.float32)
    # image_r = Reshape((Classifier.HEIGHT, Classifier.WIDTH, 1))(image)
    # conv1 = Conv2D(
    #     len(Classifier.CHARACTERS),
    #     Classifier.CONV1_SIZE,
    #     activation=None,
    #     name='conv1'
    # )(image_r)
    # logits = MaxAlongDims((1, 2))(conv1)
    # output = Activation(activation='softmax')(logits)
    #
    # model = Model(
    #     inputs=image,
    #     outputs=output
    # )
    # model.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy'
    # )
    #
    # model.summary()
    #
    # l: Layer = model.layers[2]
    #
    # p = os.path.join(os.path.dirname(__file__), 'data', 'big_noodle.npz')
    # restore_dict = dict(np.load(p))
    # l.set_weights([restore_dict['conv1/kernel:0'], restore_dict['conv1/bias:0']])
    #
    # export_saved_model(model, f'./data/big_noodle', serving_only=True)

    image = cv2.imread("C:/tmp/test.png")
    for _ in range(10):
        print(ocr(image))


if __name__ == "__main__":
    main()
