import logging
import os

import cv2
import Levenshtein as levenshtein
import numpy as np

from overtrack_cv.core import arrayops, imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.core.uploadable_image import lazy_upload
from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch.ocr import big_noodle
from overtrack_cv.games.overwatch.processors.loading_map import LoadingMap, Teams
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


def _draw_loading_screen(debug_image: np.ndarray, loading_map: LoadingMap) -> None:
    if debug_image is None:
        return

    lines = [((1000, 920), loading_map.game_mode), ((1000, 1000), loading_map.map)]
    if loading_map.teams:
        lines.append(((900, 250), " | ".join(sr for sr in loading_map.teams.average_sr)))

        y = 350
        x1 = 130
        x2 = 1500
        h = 75
        for i in range(6):
            lines.append(((x1, y), loading_map.teams.blue_team[i] + " | " + str(loading_map.teams.blue_ranks[i])))
            lines.append(((x2, y), str(loading_map.teams.red_ranks[i]) + " | " + loading_map.teams.red_team[i]))
            y += h
    lines.append(((100, 100), f"Role queue: {loading_map.is_role_queue}"))
    lines.append(((100, 140), f"In queue: {loading_map.is_in_queue}"))
    for pos, line in lines:
        for t, c in (5, (0, 0, 0)), (2, (0, 255, 0)):
            cv2.putText(debug_image, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, c, t)


class LoadingMapProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))
    ROLE_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "role_images.png"), 0)
    RANK_NAMES = [
        None,
        "bronze",
        "silver",
        "gold",
        "platinum",
        "diamond",
        "master",
        "grandmaster",
        "top500",
        "placement",
    ]

    def __init__(self, use_tflite: bool = True):
        if use_tflite:
            from overtrack_cv.core.tflite import TFLiteModel

            self.model = TFLiteModel(os.path.join(os.path.dirname(__file__), "data", "rank_icons.tflite"))
        else:
            from tensorflow.python.keras import Model

            from overtrack_cv.core.tf import load_model

            self.model: Model = load_model(os.path.join(os.path.dirname(__file__), "data", "rank_icons"))

    def process(self, frame: Frame) -> bool:
        if frame.overwatch.loading_map:
            return True
        elif frame.overwatch.loading_match:
            return False

        if not self.detect_loading_map(frame):
            return False

        map_image = self.REGIONS["map"].extract_one(frame.image)
        if np.mean(map_image) < 10:
            logger.warning(f"Loading map got mean pixel level {np.mean(map_image):.1f} - ignoring")
            return False

        map_name = big_noodle.ocr(map_image, height=82)
        game_mode = big_noodle.ocr(self.REGIONS["game_mode"].extract_one(frame.image), height=46, multiline=True)

        blue_ranks = self.model.predict(
            np.array(self.REGIONS["blue_ranks"].extract(frame.image)).astype(np.float32)
        )
        red_ranks = self.model.predict(np.array(self.REGIONS["red_ranks"].extract(frame.image)).astype(np.float32))

        if levenshtein.ratio(game_mode, "COMPETITIVEPLAY") > 0.75:
            teams = Teams(
                blue_team=big_noodle.ocr_all(self.REGIONS["blue_names"].extract(frame.image), height=37),
                red_team=big_noodle.ocr_all(self.REGIONS["red_names"].extract(frame.image), height=37),
                blue_ranks=[self.RANK_NAMES[arrayops.argmax(r)] for r in blue_ranks],
                red_ranks=[self.RANK_NAMES[arrayops.argmax(r)] for r in red_ranks],
                average_sr=big_noodle.ocr_all(
                    self.REGIONS["average_srs"].extract(frame.image_yuv[:, :, 0]), height=52
                ),
            )
        else:
            teams = None

        rank_images = np.min(
            np.hstack(
                [
                    np.vstack(self.REGIONS["blue_roles"].extract(frame.image)),
                    np.vstack(self.REGIONS["red_roles"].extract(frame.image)),
                ]
            ),
            axis=2,
        )
        _, rank_images_thresh = cv2.threshold(rank_images, 180, 255, cv2.THRESH_BINARY)
        role_match = np.max(cv2.matchTemplate(rank_images_thresh, self.ROLE_TEMPLATE, cv2.TM_CCORR_NORMED))

        frame.overwatch.loading_map = LoadingMap(
            map=map_name,
            game_mode=game_mode,
            teams=teams,
            is_role_queue=role_match > 0.75,
            is_in_queue=self.detect_in_queue(frame),
            image=lazy_upload("teams", self.REGIONS.blank_out(frame.image), frame.timestamp),
        )
        _draw_loading_screen(frame.debug_image, frame.overwatch.loading_map)

        return True

    UNSHARP_BLUR = 10
    UNSHARP_WEIGHT = 4
    LOADING_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "loading_map_template.png"), 0
    )
    LOADING_MATCH_THRESHOLD = 0.6

    def detect_loading_map(self, frame: Frame) -> bool:
        # note: old overwatch had this icon higher - use 'loading_icon_old' for a region that includes that icon position
        region = self.REGIONS["loading_icon"].extract_one(frame.image)

        unsharp = imageops.fast_gaussian(region, self.UNSHARP_BLUR, scale=4)
        im = cv2.addWeighted(region, self.UNSHARP_WEIGHT, unsharp, 1 - self.UNSHARP_WEIGHT, 0)
        gray = np.min(im, axis=2)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # cv2.imshow('thresh', thresh)
        # cv2.waitKey(1)

        match = float(1 - np.min(cv2.matchTemplate(thresh, self.LOADING_TEMPLATE, cv2.TM_SQDIFF_NORMED)))

        frame.overwatch.loading_match = round(match, 5)
        return match > self.LOADING_MATCH_THRESHOLD

    TIME_ELAPSED_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "time_elapsed.png"), 0)
    TIME_ELAPSED_MATCH_THRESHOLD = 0.75

    def detect_in_queue(self, frame: Frame) -> bool:
        region = self.REGIONS["time_elapsed"].extract_one(frame.image_yuv[:, :, 0])

        _, thresh = cv2.threshold(region, 130, 255, cv2.THRESH_BINARY)
        match = np.max(cv2.matchTemplate(thresh, self.TIME_ELAPSED_TEMPLATE, cv2.TM_CCORR_NORMED))
        logger.debug(f"Time elapsed match={match:.1f}")

        return match > self.TIME_ELAPSED_MATCH_THRESHOLD


def convert_model() -> None:
    import tensorflow as tf
    from overtrack.util.tf import BGR2RGB, ByteToFloat, load_model
    from tensorflow.python.keras import Input, Model
    from tensorflow.python.keras.layers import Dense, Layer, Reshape
    from tensorflow.python.keras.saving import export_saved_model

    imsize = (51, 51, 3)
    image = Input(shape=imsize, name="images", dtype=tf.float32)
    image_rgb = BGR2RGB()(image)
    image_r = Reshape((np.prod(imsize),))(image_rgb)
    image_r = ByteToFloat()(image_r)

    logits = Dense(len(LoadingMapProcessor.RANK_NAMES), activation="softmax", name="fc1")(image_r)
    model = Model(inputs=image, outputs=logits)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.summary()

    p = os.path.join(os.path.dirname(__file__), "data", "rank_icons.npz")
    restore_dict = dict(np.load(p))
    print({k: v.shape for k, v in restore_dict.items()})
    l: Layer = model.layers[4]
    print(l, [w.shape for w in l.get_weights()])
    l.set_weights([restore_dict["rank_icons/W_fc_:0"], restore_dict["rank_icons/b_fc_:0"]])
    os.makedirs("./data/rank_icons", exist_ok=True)
    export_saved_model(model, f"./data/rank_icons", serving_only=True)

    # model: Model = load_model(
    #     os.path.join(os.path.dirname(__file__), 'data', 'rank_icons'),
    #     custom_objects={
    #         'ByteToFloat': ByteToFloat,
    #         'BGR2RGB': BGR2RGB
    #     }
    # )
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(os.path.join(os.path.dirname(__file__), "data", "rank_icons.tflite"), "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    # convert_model()

    from overtrack_cv.util.test_processor import test_processor

    test_processor(LoadingMapProcessor(True), "loading_map", "loading_match")
