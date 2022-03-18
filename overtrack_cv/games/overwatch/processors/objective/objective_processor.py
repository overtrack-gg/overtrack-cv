import dataclasses
import logging
import os

import cv2
import numpy as np

from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch.processors.objective import Objective
from overtrack_cv.games.processor import Processor
from overtrack_cv.util.prettyprint import pprint

logger = logging.getLogger(__name__)


class ObjectiveProcessor(Processor):

    OBJECTIVE_TOP_LIVE = 0
    OBJECTIVE_TOP_OWL = 46

    def __init__(self, use_tflite: bool = True, top=OBJECTIVE_TOP_LIVE) -> None:
        self.top = top
        if use_tflite:
            from overtrack_cv.core.tflite import TFLiteModel

            self.model = TFLiteModel(os.path.join(os.path.dirname(__file__), "data", "parse_objective.tflite"))
        else:
            from overtrack_cv.core.tf import load_model

            self.model = load_model(os.path.join(os.path.dirname(__file__), "data", "v18"))

    def process(self, frame: Frame) -> bool:
        if frame.debug_image is not None:
            cv2.rectangle(
                frame.debug_image,
                (frame.image.shape[1] // 2 - 310, self.top),
                (frame.image.shape[1] // 2 + 310, self.top + 250),
                (0, 255, 0),
            )
        objective_image = frame.image[
            self.top : self.top + 250,
            frame.image.shape[1] // 2 - 310 : frame.image.shape[1] // 2 + 310
            # :250,
            # 650:-650
        ]
        # objective_image = cv2.cvtColor(objective_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('o', objective_image)
        result = self.model.predict(np.expand_dims(objective_image, axis=0).astype(np.float32))
        frame.overwatch.objective = Objective.from_result(result)

        # if frame.objective2.competitive and frame.objective2.checkpoint and frame.objective2.started and not frame.get('compressed_image', False):
        #     print(frame.objective2.competitive, frame.objective2.checkpoint, frame.objective2.started)
        #     self.add_checkpoint_ocr(frame.objective2, frame)

        # print(frame.objective2)
        _draw_objective(frame.debug_image, frame.overwatch.objective)

        return frame.overwatch.objective.overwatch

    # def add_checkpoint_ocr(self, objective: Objective3, frame: Frame) -> None:
    #     rows = [
    #         self.REGIONS['checkpoint_time_remaining'].extract(frame.image)[not objective.attacking],
    #         self.REGIONS['checkpoint_progress'].extract(frame.image)[not objective.attacking],
    #     ]
    #     region = np.vstack(rows)
    #     median_color = np.median(region, axis=(0, 1))
    #     cv2.imshow(
    #         'color',
    #         np.full(
    #             (100, 100, 3),
    #             fill_value=median_color,
    #             dtype=np.uint8
    #         )
    #     )
    #     match = cv2.erode(
    #         np.mean(np.abs(region.astype(np.int) - median_color.astype(np.int)), axis=2).astype(np.uint8),
    #         np.ones((5, 5), dtype=np.uint8)
    #     )
    #     cv2.imshow(
    #         'match',
    #         match.astype(np.uint8)
    #     )
    #     match_1d = np.mean(match, axis=0)
    #
    #     import matplotlib.pyplot as plt
    #
    #     plt.figure()
    #     plt.plot(match_1d)
    #     plt.show()
    #
    #     for image in rows:
    #         # cv2.imshow('image', image)
    #         from overtrack.util import debugops
    #         #
    #         # cv2.imshow('chans', np.vstack(
    #         #     list(cv2.split(image)) +
    #         #     list(cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL))) +
    #         #     list(cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YUV))) +
    #         #     [np.min(image, axis=2), np.max(image, axis=2), np.ptp(image, axis=2)]
    #         # ))
    #
    #
    #         debugops.test_tesser_engines(np.ptp(image, axis=2))
    #
    #         # debugops.manual_thresh_adaptive(image)
    #         # cv2.waitKey(0)


_CHECKPOINT_STATES = [
    None,
    "checkpoint_assemble",
    "checkpoint_prepare",
    "checkpoint_attack",
    "checkpoint_defend",
    "koth_assemble",
    "koth_prepare",
    "koth_locked",
    "overtime",
]


def _draw_objective(debug_image: np.ndarray, objective: Objective) -> None:
    if debug_image is None:
        return

    lines = []
    for field in dataclasses.fields(objective):
        lines.append(f"{field.name}: {getattr(objective, field.name)}")
    lines.append("")

    props = []
    for n in dir(objective):
        f = getattr(objective.__class__, n, None)
        if type(f) is property:
            props.append((f.fget.__code__.co_firstlineno, n))
    for _, n in sorted(props):
        if n == "state":
            lines.append(f"{n} => {_CHECKPOINT_STATES[getattr(objective, n)]} ({getattr(objective, n)})")
        else:
            lines.append(f"{n} => {getattr(objective, n)}")

    for i, line in enumerate(lines):
        cv2.putText(
            debug_image, line, (200, 160 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            debug_image, line, (200, 160 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )


def convert_tflite():
    import tensorflow as tf

    p = ObjectiveProcessor(use_tflite=False)
    model = p.model

    model.outputs = [
        tf.keras.layers.Activation(None, name=f'{i}_{o.name.split("/")[0]}')(o)
        for i, o in enumerate(model.outputs)
    ]

    new_model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs)
    new_model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    tflite_model = converter.convert()
    with open(os.path.join(os.path.dirname(__file__), "data", "parse_objective.tflite"), "wb") as f:
        f.write(tflite_model)

    import tflite_runtime.interpreter as tflite

    interpreter = tflite.Interpreter(os.path.join(os.path.dirname(__file__), "data", "parse_objective.tflite"))
    pprint(new_model.outputs)
    pprint(interpreter.get_output_details())


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(ObjectiveProcessor(), "objective")


if __name__ == "__":
    # convert_tflite()

    import tensorflow as tf

    with tf.device("/cpu:0"):
        proc = ObjectiveProcessor()

    # util.test_processor('objective', proc, 'objective2', test_all=False)

    import glob
    import random

    paths = glob.glob("D:/overtrack/overwatch_objective/*/*.objective.png", recursive=True)
    random.shuffle(paths)
    for p in paths:
        print(p)
        pp = p.replace(".png", ".py")
        if os.path.exists(pp):
            with open(pp) as f:
                print(f.read())

        im = np.zeros((1080, 1920, 3), dtype=np.uint8)
        im[:250, 650:-650] = cv2.imread(p)

        f = Frame.create(im, 0, debug=True)
        proc.process(f)

        # print(tabulate.tabulate([(n, f.get(n)) for n in fields]))
        # pprint(f.timings)
        cv2.imshow("debug", f.debug_image)

        if f.objective2.competitive and f.objective2.checkpoint and f.objective2.started:
            cv2.waitKey(0)
        else:
            cv2.waitKey(0)

    from overtrack.source.shmem.shmem_capture import SharedMemoryCapture

    cap = SharedMemoryCapture(debug_frames=True, update_frequency=0.1)
    cap.start()

    while True:
        f = cap.get_blocking()
        proc.process(f)
        cv2.imshow("debug", f.debug_image)
        cv2.waitKey(1)
