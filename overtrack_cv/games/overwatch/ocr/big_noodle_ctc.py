import logging
import os
import string
import time
from typing import Any, Dict, List

import cv2
import numpy as np
from tensorflow.python.keras import Model

from overtrack_cv.core import imageops
from overtrack_cv.core.tf import decode_ctc, load_model
from overtrack_cv.util.logging_config import config_logger

logger = logging.getLogger(__name__)


class Decoder:
    IMAGE_HEIGHT = 20
    IMAGE_WIDTH = 300
    CONV1_SIZE = (5, 5)
    CONV1_LAYERS = 2
    CONV2_WIDTH = 10
    MAXPOOL_WIDTH = 1
    MERGE_REPEATED = True
    CHARACTERS = np.array(list(string.digits + string.ascii_uppercase))

    _instance = None

    @classmethod
    def get_instance(cls) -> "Decoder":
        if cls._instance:
            return cls._instance
        cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model: Model = load_model(os.path.join(os.path.dirname(__file__), "data", "big_noodle_ctc"))

    def decode(self, images: List[np.ndarray]) -> List[str]:
        if not len(images):
            return []
        seq_lens = [im.shape[1] - (self.CONV2_WIDTH - 2) for im in images]
        if len(images) > 1:
            maxwidth = max(im.shape[1] for im in images)
            images = [
                cv2.copyMakeBorder(im, 0, 0, 0, maxwidth - im.shape[1], cv2.BORDER_CONSTANT) for im in images
            ]
        logs = self.model.predict([images])
        return ["".join(s) for s in decode_ctc(logs, alphabet=self.CHARACTERS, seq_lens=seq_lens)]


def ocr_all(images: List[np.ndarray], **kwargs: Any) -> List[str]:
    classifier = Decoder.get_instance()

    t0 = time.perf_counter()

    images_sized = []
    for im in images:
        _, thresh = cv2.threshold(np.max(im, axis=2), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, components = imageops.connected_components(thresh)
        components = [c for c in components[1:] if c.area > 25]
        if not components:
            # TODO
            images_sized.append(np.zeros((20, 300, 3), np.uint8))
            continue
        y, h = np.median([c.y for c in components]), np.median([c.h for c in components])
        y -= h * 0.05
        h *= 1.15
        y, h = int(y), int(h)
        im = im[max(0, y) : min(y + h, im.shape[0])]
        s = classifier.IMAGE_HEIGHT / im.shape[0]
        im = cv2.resize(im, (int(im.shape[1] * s), classifier.IMAGE_HEIGHT))
        # im = cv2.copyMakeBorder(im, 0, 0, 0, classifier.IMAGE_WIDTH - im.shape[1], cv2.BORDER_CONSTANT)
        images_sized.append(im)

    r = classifier.decode(images_sized)

    t1 = time.perf_counter()
    logger.debug(f"OCR took {(t1 - t0)*1000:1.2f}ms")

    return r


def ocr(image: np.ndarray, **kwargs: Dict[str, Any]) -> str:
    return ocr_all([image], **kwargs)[0]


def main() -> None:
    config_logger("big_noodle", logging.DEBUG, False)

    # from tensorflow.python.keras import Input
    # from tensorflow.python.keras.layers import Conv2D
    # from tensorflow.python.keras import Model
    # from tensorflow.python.keras.layers import Layer
    # from tensorflow.python.keras.saving import export_saved_model
    # from tensorflow.python.keras.layers import Lambda
    #
    # image = Input(shape=(Decoder.IMAGE_HEIGHT, Decoder.IMAGE_WIDTH, 3), name='images', dtype=tf.float32)
    # image_norm = Lambda(lambda i: i / 255.0 - 0.5)(image)
    # conv1 = Conv2D(
    #     Decoder.CONV1_LAYERS,
    #     Decoder.CONV1_SIZE,
    #     activation='relu',
    #     padding='same',
    #     name='conv1'
    # )(image_norm)
    # conv1_pad = Pad(
    #     ((0, 0), (0, 0), (0, Decoder.CONV2_WIDTH - 1), (0, 0))
    # )(conv1)
    # conv2 = Conv2D(
    #     len(Decoder.CHARACTERS) + 1,
    #     (Decoder.IMAGE_HEIGHT - 2, Decoder.CONV2_WIDTH),
    #     activation=None,
    #     padding='valid',
    #     name='conv2'
    # )(conv1_pad)
    # logits = MaxAlongDims(dims=(1, ))(conv2)
    #
    # model = Model(
    #     inputs=image,
    #     outputs=logits
    # )
    # model.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy'
    # )
    # model.summary()
    #
    # p = os.path.join(os.path.dirname(__file__), 'data', 'big_noodle_ctc.npz')
    # restore_dict = dict(np.load(p))
    # print({k: v.shape for k, v in restore_dict.items()})
    #
    # l: Layer = model.layers[2]
    # print(l, [w.shape for w in l.get_weights()])
    # l.set_weights([restore_dict['conv1/kernel:0'], restore_dict['conv1/bias:0']])
    #
    # l: Layer = model.layers[4]
    # print(l, [w.shape for w in l.get_weights()])
    # l.set_weights([restore_dict['conv2/kernel:0'], restore_dict['conv2/bias:0']])
    #
    # os.makedirs('./data/big_noodle_ctc')
    # export_saved_model(model, f'./data/big_noodle_ctc', serving_only=True)

    image = cv2.imread("C:/tmp/test.png")
    for _ in range(10):
        print(ocr(image))
    for _ in range(10):
        print(ocr_all([image, np.hstack((image, image))]))


if __name__ == "__main__":
    main()
