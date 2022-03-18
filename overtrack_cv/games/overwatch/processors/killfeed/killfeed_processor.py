import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch.processors.killfeed import Killfeed, KillRow, Player
from overtrack_cv.games.processor import Processor
from overtrack_cv.util.logging_config import intermittent_log

try:
    from overtrack_cv.games.overwatch.processors.killfeed.find_peaks import find_peaks
except ImportError:
    from scipy.signal import find_peaks

try:
    from typing import TypedDict
except ImportError:
    TypedDict = Dict


logger = logging.getLogger(__name__)


class InvalidKillRow(ValueError):
    pass


class LayerOutput(TypedDict):
    name: str
    values: List[str]


class RowPredictionsDict(TypedDict):
    text: np.ndarray
    heroes: np.ndarray
    heroes_softmax: np.ndarray
    assists: np.ndarray
    abilities: np.ndarray


class RowDecodedDict(TypedDict):
    heroes: List[str]
    assists: List[str]
    abilities: List[str]


class KillfeedPredictors:
    outputs: List[LayerOutput]

    def predict_row_positions(self, region: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def predict_rows(self, row_ims: List[np.ndarray]) -> Tuple[RowPredictionsDict, RowDecodedDict]:
        raise NotImplementedError()

    def decode_ctc(self, data: Union[np.ndarray, List[np.ndarray]], alphabet: List[str]) -> List[List[str]]:
        raise NotImplementedError()

    def heropos2img(self, x: int) -> int:
        raise NotImplementedError()

    def img2textpos(self, x: int) -> int:
        raise NotImplementedError()


class KillfeedPredictorsKeras(KillfeedPredictors):
    def __init__(self):
        import tensorflow as tf
        from tensorflow.python.keras import Model

        from overtrack_cv.core.tf import (
            all_custom_objects,
            decode_ctc,
            deserialize,
            load_model,
        )

        self.extract: Model = load_model(os.path.join(os.path.dirname(__file__), "data", "extract_v14"))

        model_path = os.path.join(os.path.dirname(__file__), "data", "parse_v6")
        with open(model_path + "/assets/saved_model.json") as f:
            self.parse_config = json.load(f)
        self.parse_config["config"]["layers"][0]["config"]["batch_input_shape"][2] = None
        del self.parse_config["config"]["layers"][0]["config"]["ragged"]
        self.parse: Model = deserialize(self.parse_config, custom_objects=all_custom_objects)
        self.parse.load_weights(model_path + "/variables/variables")
        self.parse.trainable = False
        with open(model_path + "/outputs.json") as f:
            self.outputs = json.load(f)["outputs"]
            self.outputs_dict = {o["name"]: o for o in self.outputs}
        self.parse_layer_defs = {layer["name"]: layer for layer in self.parse_config["config"]["layers"]}
        # self.parse.outputs += [
        #     tf.keras.layers.Activation('softmax', name='heroes_softmax')(self.parse.outputs[1]),
        #     # tf.nn.softmax(self.parse.outputs[1]),  # hero_softmax
        # ]
        # self.parse.build((None, 50, 600, 3))

        self._decode_ctc = decode_ctc
        self._softmax = tf.nn.softmax

    def predict_row_positions(self, region: np.ndarray) -> np.ndarray:
        return self.extract.predict(np.expand_dims(region, axis=0))[0]

    def predict_rows(self, row_ims: List[np.ndarray]) -> Tuple[RowPredictionsDict, RowDecodedDict]:
        predictions_raw = self.parse.predict(np.array(row_ims), 1)
        predictions = {self.outputs[i]["name"]: predictions_raw[i] for i in range(len(self.outputs))}
        predictions["heroes_softmax"] = self._softmax(predictions_raw[1])
        decoded = {
            name: list(self.decode_ctc(predictions[name], self.outputs_dict[name]["values"]))
            for name in ["heroes", "assists", "abilities"]
        }
        return predictions, decoded

    def decode_ctc(self, data: Union[np.ndarray, List[np.ndarray]], alphabet: List[str]) -> List[List[str]]:
        return self._decode_ctc(data, alphabet=alphabet)

    def heropos2img(self, x: int) -> int:
        unmaxpool = x * self.parse_layer_defs["heroes_hidden/input_maxpool"]["config"]["pool_size"][1]
        shared_pos = unmaxpool * self.parse_layer_defs["heroes_hidden/output_maxpool"]["config"]["pool_size"][1]
        return shared_pos

    def img2textpos(self, x: int) -> int:
        maxpooled = x // self.parse_layer_defs["text_hidden/output_maxpool"]["config"]["pool_size"][1]
        return maxpooled


class KillfeedPredictorsTflite(KillfeedPredictors):
    def __init__(self):
        # from tensorflow.python.keras import Model
        # from overtrack.util.tf import load_model
        #
        # self.extract: Model = load_model(
        #     os.path.join(os.path.dirname(__file__), 'data', 'extract_v14')
        # )

        import tflite_runtime.interpreter as tflite

        logger.info(f"Loading {self.__class__.__name__} extract model")
        self.extract = tflite.Interpreter(
            model_path=os.path.join(os.path.dirname(__file__), "data", "extract", "model.tflite")
        )
        logger.info("Done")
        self.extract.allocate_tensors()
        self.extract_input_details = self.extract.get_input_details()
        self.extract_output_details = self.extract.get_output_details()

        self.parse = tflite.Interpreter(
            model_path=os.path.join(os.path.dirname(__file__), "data", "parse", "model.tflite")
        )
        logger.info(f"Loading {self.__class__.__name__} parse model")
        self.parse.allocate_tensors()
        logger.info("Done")
        self.parse_input_details = self.parse.get_input_details()
        self.parse_output_details = self.parse.get_output_details()

        with open(os.path.join(os.path.dirname(__file__), "data", "parse", "model.json")) as f:
            self.model_meta = json.load(f)
        self.outputs = self.model_meta["outputs"]

        for output_details, output_meta in zip(self.parse_output_details, self.outputs):
            assert (
                len(output_meta["values"]) == output_details["shape"][-1] - 1
            ), f'Got wrong number of labels for {output_meta["name"]}'

    def predict_row_positions(self, region: np.ndarray) -> np.ndarray:
        inp = np.expand_dims(region, axis=0).astype(np.float32)
        self.extract.set_tensor(self.extract_input_details[0]["index"], inp)
        self.extract.invoke()
        return self.extract.get_tensor(self.extract_output_details[0]["index"])[0].copy()
        # return self.extract.predict(np.expand_dims(region, axis=0))[0]

    def predict_rows(self, row_ims: List[np.ndarray]) -> Tuple[RowPredictionsDict, RowDecodedDict]:
        # print('predicting', len(row_ims))
        # self.parse.set_tensor(self.parse_input_details[0]['index'], np.array(row_ims).astype(np.float32))
        # self.parse.invoke()

        preds, decoded = defaultdict(list), defaultdict(list)
        for im in row_ims:
            self.parse.set_tensor(
                self.parse_input_details[0]["index"], np.expand_dims(im.astype(np.float32), axis=0)
            )
            self.parse.invoke()

            for output_desc, prediction_desc in zip(self.outputs, self.parse_output_details):
                preds[output_desc["name"]].append(self.parse.get_tensor(prediction_desc["index"])[0].copy())
            preds["heroes_softmax"].append(self.parse.get_tensor(self.parse_output_details[4]["index"])[0].copy())

            for n, prediction_desc in zip(["heroes", "assists", "abilities"], self.outputs[1:]):
                decoded[n].append(self.decode_ctc([preds[n][-1]], alphabet=prediction_desc["values"])[0])

        return {
            "text": np.array(preds["text"]),
            "heroes": np.array(preds["heroes"]),
            "assists": np.array(preds["assists"]),
            "abilities": np.array(preds["abilities"]),
            "heroes_softmax": np.array(preds["heroes_softmax"]),
        }, {
            "heroes": list(decoded["heroes"]),
            "assists": list(decoded["assists"]),
            "abilities": list(decoded["abilities"]),
        }

    def decode_ctc(
        self, data: Union[np.ndarray, List[np.ndarray]], alphabet: List[str], _alphabet_cache={}
    ) -> List[List[str]]:
        if id(alphabet) not in _alphabet_cache:
            _alphabet_cache[id(alphabet)] = np.array(alphabet + [None])
        alphabet_arr = _alphabet_cache[id(alphabet)]
        r = []
        for d in data:
            decoded = alphabet_arr[np.argmax(d, axis=1)]
            r.append([e for e in decoded if e])
        return r

    def heropos2img(self, x: int) -> int:
        unmaxpool = x * self.model_meta["heroes_hidden/input_maxpool/config/pool_size/1"]
        shared_pos = unmaxpool * self.model_meta["heroes_hidden/output_maxpool/config/pool_size/1"]
        return shared_pos

    def img2textpos(self, x: int) -> int:
        maxpooled = x // self.model_meta["text_hidden/output_maxpool/config/pool_size/1"]
        return maxpooled


class KillfeedProcessor(Processor):
    # REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), 'data', 'regions', '16_9.zip'))

    capture_top = 30
    capture_height = 600
    capture_width = 600

    row_height = 50

    def __init__(self, use_tflite=True, extrapolate_rowdetect=False, check_heroonly_killfeed=False):
        self.extrapolate_rowdetect = extrapolate_rowdetect
        self.check_heroonly_killfeed = check_heroonly_killfeed

        if use_tflite:
            self.predictors: KillfeedPredictors = KillfeedPredictorsTflite()
        else:
            self.predictors: KillfeedPredictors = KillfeedPredictorsKeras()

    def process(self, frame: Frame) -> bool:
        region = frame.image[
            self.capture_top : self.capture_top + self.capture_height, frame.image.shape[1] - self.capture_width :
        ]

        ys = self._get_row_ys(region, frame.debug_image)
        k = self._parse_killfeed(region, ys, frame.debug_image, frame.get("hero_only_killfeed"))
        if k:
            frame.overwatch.killfeed = k
            return True

        return False

    def _get_row_ys(self, region: np.ndarray, debug_image: Optional[np.ndarray] = None) -> np.ndarray:
        rows = self.predictors.predict_row_positions(region)

        # peaks, details = scipy.signal.find_peaks(
        peaks, details = find_peaks(rows, height=0.2, distance=20)
        peaks *= 2
        peaks -= 43

        if debug_image is not None:
            debug_image[self.capture_top : self.capture_top + rows.shape[0] * 2, -10:] = cv2.cvtColor(
                cv2.resize((np.expand_dims(rows, -1) * 255).astype(np.uint8), (10, rows.shape[0] * 2)),
                cv2.COLOR_GRAY2BGR,
            )
            for peak, height in zip(peaks, details["peak_heights"]):
                cv2.line(
                    debug_image,
                    (debug_image.shape[1] - self.capture_width, peak + self.capture_top),
                    (debug_image.shape[1] - 15, peak + self.capture_top),
                    (0, 255, 0),
                )
                cv2.line(
                    debug_image,
                    (debug_image.shape[1] - 15, peak + self.capture_top),
                    (debug_image.shape[1] - 10, peak + self.capture_top + 43),
                    (0, 255, 0),
                )
                cv2.putText(
                    debug_image,
                    f"{peak + self.capture_top}: {height:1.2f}",
                    (debug_image.shape[1] - self.capture_width - 100, peak + self.capture_top + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                )

        return peaks

    def parse_kill(
        self, index: int, predictions: RowPredictionsDict, decoded: RowDecodedDict, y: int, extrapolated: bool
    ) -> Optional[KillRow]:
        # hero_pos, details = scipy.signal.find_peaks(1 - predictions['heroes_softmax'][index, :, -1], height=0.5, distance=3)
        hero_pos, details = find_peaks(1 - predictions["heroes_softmax"][index, :, -1], height=0.5, distance=3)

        heroes = decoded["heroes"][index]
        assists = decoded["assists"][index]
        abilities = decoded["abilities"][index]

        if not len(heroes):
            intermittent_log(
                logger,
                f"Ignoring detected kill row {index} (y={y}, extrapolated={extrapolated}) with no heroes "
                f"(best match {np.max(1 - predictions['heroes_softmax'][index, :, -1]):1.2f})",
                frequency=30,
                level=logging.WARNING,
                caller_extra_id=round(y / 10),
            )
            return None

        if len(heroes) > 2:
            raise InvalidKillRow(
                f"Got {len(heroes)} heroes in kill row {index} (y={y}, extrapolated={extrapolated}): {heroes}"
            )

        if len(hero_pos) != len(heroes) and not extrapolated:
            raise InvalidKillRow(
                f"Got {len(hero_pos)} hero peaks in kill row {index} (y={y}, extrapolated={extrapolated}), "
                f"but had {len(heroes)} heroes: {hero_pos}, {heroes}"
            )

        if len(hero_pos) and len(heroes) <= 2:
            if len(hero_pos) == 2:
                splitpos = int(np.mean([self.predictors.heropos2img(x) for x in hero_pos]))
            else:
                splitpos = self.predictors.heropos2img(hero_pos[0])

            splitpos_textspace = self.predictors.img2textpos(splitpos)
            text_logs_left, text_logs_right = (
                predictions["text"][index, :splitpos_textspace],
                predictions["text"][index, splitpos_textspace:],
            )

            text_left = "".join(
                self.predictors.decode_ctc([text_logs_left], alphabet=self.predictors.outputs[0]["values"])[0]
            )
            text_right = "".join(
                self.predictors.decode_ctc([text_logs_right], alphabet=self.predictors.outputs[0]["values"])[0]
            )

            if len(hero_pos) == 1:
                if len(text_left) >= 3:
                    # one hero, but text was to the left of it so there must be a (unknown) right hero
                    hero_left = heroes[0]
                    hero_right = "UNKNOWN"
                else:
                    text_left = None
                    hero_left = None
                    hero_right = heroes[0]
            else:
                hero_left, hero_right = heroes

            ability = None
            if len(abilities) > 1:
                logger.warning(f"Got multiple abilities {abilities} for row {index} (hero={hero_left!r}")
            elif len(abilities):
                ability = abilities[0]
                ability_hero = ability.split(".")[0]
                if ability_hero != "ANY" and ability_hero != hero_left:
                    logger.warning(
                        f"Got mismatching ability {ability!r} but hero was {hero_left!r} for row {index}"
                    )

            return KillRow(
                left=Player(hero_left, text_left) if hero_left else None,
                right=Player(hero_right, text_right),
                y=int(y),
                ability=ability,
                assists=list(assists),
                resurrect=bool(ability and "resurrect" in ability and "mercy" == hero_left),
            )

    def _parse_killfeed(
        self, region: np.ndarray, ys: np.ndarray, debug_image: Optional[np.ndarray] = None, hero_only_killfeed=None
    ) -> Optional[Killfeed]:
        if not len(ys) and not self.check_heroonly_killfeed:
            return None

        full_ys = list(ys)
        extrapolated_y = [False for _ in ys]
        if self.extrapolate_rowdetect and len(ys):
            full_ys.append(ys[-1] + 52)
            extrapolated_y.append(True)

        if self.check_heroonly_killfeed and hero_only_killfeed:
            for row in hero_only_killfeed.killfeed:
                if row.left:
                    y = (row.left.y + row.right.y) / 2
                else:
                    y = row.right.y
                y -= 55.5
                closest = None
                for other in full_ys:
                    if closest is None or abs(other - y) < closest:
                        closest = abs(other - y)
                if closest is None or closest > 10:
                    logger.warning(f"Adding row at y={y:.0f} from HeroOnlyKillfeed")
                    full_ys.append(int(y))
                else:
                    logger.info(f"Found y={y:.0f} in HeroOnlyKillfeed matching row from row extraction")

        row_ims = []
        kill_ys = []
        for i, top in enumerate(full_ys):
            if 0 <= top < region.shape[0] - self.row_height:
                im = region[top : top + self.row_height]
                if im.shape == (50, self.capture_width, 3):
                    row_ims.append(im)
                    kill_ys.append(top + self.capture_top)
                else:
                    logger.error(f"Got kill row {i} with shape={im.shape}, top={top} - ignoring")
            else:
                logger.warning(
                    f"Got kill row {i} with top={top} outside of range (0, {region.shape[0] - self.row_height}) - ignoring"
                )

        if not len(row_ims):
            return None

        predictions, decoded = self.predictors.predict_rows(row_ims)

        killfeed = Killfeed()
        for i, (y, is_extrapolated) in enumerate(zip(kill_ys, extrapolated_y)):
            try:
                kill = self.parse_kill(
                    index=i,
                    predictions=predictions,
                    decoded=decoded,
                    y=y,
                    extrapolated=is_extrapolated,
                )
            except InvalidKillRow as e:
                logger.warning(f"{e}")
                killfeed.unknown_rows.append(y)
            else:
                if kill:
                    logger.debug(f"Got kill {i}: {kill}")
                    killfeed.kills.append(kill)
                    self._draw_kill(debug_image, kill)

        return killfeed

    def _draw_kill(self, debug_image: Optional[np.ndarray], kill: KillRow):
        if debug_image is not None:
            for j, (hero, text) in enumerate(
                [
                    (kill.left.hero if kill.left else "", kill.right.hero),
                    (repr(kill.left.name) if kill.left else "", repr(kill.right.name)),
                    (", ".join(kill.assists), " | " + (kill.ability or "")),
                ]
            ):
                top = kill.y + (j + 1) * 15 - 3
                t = f"{hero} {text}"
                (w, h), baseline = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                dark_region = debug_image[
                    top - h + 2 : top + baseline - 1,
                    debug_image.shape[1] - self.capture_width : debug_image.shape[1] - self.capture_width + w,
                ]
                dark_region[:] = np.clip(dark_region, 75, 255) - 75
                cv2.putText(
                    debug_image,
                    t,
                    (debug_image.shape[1] - self.capture_width, top),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )


def export_model():
    import shutil

    import tensorflow as tf

    from overtrack_cv.core.tf import all_custom_objects, deserialize, save_model

    src = "C:/Users/simon/overtrack_2/training/ctc_ocr/overwatch_killfeed_full_2/checkpoint"
    with open(src + "/assets/saved_model.json") as f:
        parse_config = json.load(f)
    parse_config["config"]["layers"][0]["config"]["batch_input_shape"][2] = None
    del parse_config["config"]["layers"][0]["config"]["ragged"]

    parse: tf.keras.Model = deserialize(parse_config, custom_objects=all_custom_objects)
    parse.load_weights(src + "/variables/variables")
    parse.trainable = False

    new_model = tf.keras.Model(inputs=parse.inputs, outputs=parse.outputs)
    new_model_weights = {l.name: l for l in new_model.layers}
    for layer in parse.layers:
        new_model_weights[layer.name].set_weights(layer.get_weights())

    dst = os.path.join(os.path.dirname(__file__), "data", "parse_v6")
    save_model(new_model, dst, include_optimizer=False)
    shutil.copy(os.path.join(src, "outputs.json"), os.path.join(dst, "outputs.json"))


def export_tflite_model():
    import tensorflow as tf

    p = KillfeedPredictorsKeras()

    p.extract.input.set_shape((1, 600, 600, 3))
    converter = tf.lite.TFLiteConverter.from_keras_model(p.extract)
    tflite_model = converter.convert()
    os.makedirs(os.path.join(os.path.dirname(__file__), "data", "extract"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "data", "extract", "model.tflite"), "wb") as f:
        f.write(tflite_model)

    model = p.parse
    model.input.set_shape((1, 50, 600, 3))
    model.outputs = [
        tf.keras.layers.Activation(None, name="0_0_text")(model.outputs[0]),
        tf.keras.layers.Activation(None, name="0_1_heroes")(model.outputs[1]),
        tf.keras.layers.Activation(None, name="0_2_assists")(model.outputs[2]),
        tf.keras.layers.Activation(None, name="0_3_abilities")(model.outputs[3]),
        # tf.keras.layers.Activation('softmax', name='1_0_text_softmax')(model.outputs[0]),
        tf.keras.layers.Activation("softmax", name="1_1_heroes_softmax")(model.outputs[1]),
        # tf.keras.layers.Activation('softmax', name='1_2_assists_softmax')(model.outputs[2]),
        # tf.keras.layers.Activation('softmax', name='1_3_abilities_softmax')(model.outputs[3]),
    ]
    # model.build((1, 50, 600, 3))

    new_model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs)
    new_model_weights = {l.name: l for l in new_model.layers}
    for layer in model.layers:
        new_model_weights[layer.name].set_weights(layer.get_weights())
    # new_model.build((None, 50, 600, 3))

    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)

    tflite_model = converter.convert()
    os.makedirs(os.path.join(os.path.dirname(__file__), "data", "parse"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "data", "parse", "model.tflite"), "wb") as f:
        f.write(tflite_model)

    outputs_data = p.outputs
    with open(os.path.join(os.path.dirname(__file__), "data", "parse", "model.json"), "w") as f:
        f.write(
            json.dumps(
                {
                    "outputs": outputs_data,
                    "heroes_hidden/input_maxpool/config/pool_size/1": p.parse_layer_defs[
                        "heroes_hidden/input_maxpool"
                    ]["config"]["pool_size"][1],
                    "heroes_hidden/output_maxpool/config/pool_size/1": p.parse_layer_defs[
                        "heroes_hidden/output_maxpool"
                    ]["config"]["pool_size"][1],
                    "text_hidden/output_maxpool/config/pool_size/1": p.parse_layer_defs[
                        "text_hidden/output_maxpool"
                    ]["config"]["pool_size"][1],
                }
            )
        )


if __name__ == "__main__":
    # export_model()
    # export_tflite_model()

    # exit(0)

    # tflite
    # 141.4276374
    # 1.249 B cycles
    # 15.080 104.920
    # time.sleep(5)
    #
    # im = cv2.imread(r'D:\overtrack\frames\1579164403.73_image.png')
    # f = Frame.create(im, 0, debug=True)
    # p = KillfeedProcessor2(use_tflite=True)
    # totaltime = 0
    # sleeptime = 0
    # for _ in range(60):
    #     t0 = time.time()
    #     p.process(f)
    #     t1 = time.time()
    #     del f['killfeed_2']
    #     run = (t1 - t0)
    #     totaltime += run
    #     sleep = 1 - run
    #     sleeptime += sleep
    #     print(f'{run:1.3f} {sleep:1.3f}')
    #     time.sleep(sleep)
    #
    # print()
    # print(f'{totaltime:1.3f} {sleeptime:1.3f}')
    # exit(0)
    from overtrack_cv.util.test_processor import test_processor

    test_processor(KillfeedProcessor(use_tflite=True, extrapolate_rowdetect=True), "killfeed")

    import glob
    import random

    paths = glob.glob("D:/overtrack/overwatch_killfeed/autoclass/vid_low/*.png")
    random.shuffle(paths)

    proc = KillfeedProcessor()

    for p in paths:
        killrow = cv2.imread(p)
        predictions, decoded = proc._predict([killrow])
        kill = proc.parse_kill(0, predictions, decoded, 0, False)
        if kill:
            print(kill)

            debug_image = cv2.copyMakeBorder(killrow, 50, 0, 0, 0, cv2.BORDER_CONSTANT)
            proc._draw_kill(debug_image, kill)
            cv2.imshow("kill", debug_image)
            cv2.waitKey(0)
