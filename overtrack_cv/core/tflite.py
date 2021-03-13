import json
import os
from dataclasses import dataclass, field
from typing import Generic, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


class TFLiteModel:
    def __init__(self, path: str):
        self.interpreter = tflite.Interpreter(path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        assert len(self.input_details) == 1, "TFLiteModel only supports models with a single input"
        self.interpreter.allocate_tensors()

    def predict(self, inps: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        rs = [[] for _ in self.output_details]
        for i in range(len(inps)):
            self.interpreter.set_tensor(self.input_details[0]["index"], inps[i : i + 1].copy())
            self.interpreter.invoke()
            for r, od in zip(rs, self.output_details):
                r.append(self.interpreter.get_tensor(od["index"])[0].copy())
        rs = [np.array(r) for r in rs]
        if len(rs) == 1:
            return rs[0]
        else:
            return rs


def decode_ctc(
    data: Union[np.ndarray, List[np.ndarray]], alphabet: Union[List[str], str], _alphabet_cache={}
) -> List[List[str]]:
    if id(alphabet) not in _alphabet_cache:
        if isinstance(alphabet, str):
            alphabet = list(alphabet)
        _alphabet_cache[id(alphabet)] = np.array(alphabet + [None])
    alphabet_arr = _alphabet_cache[id(alphabet)]
    r = []
    for d in data:
        decoded = alphabet_arr[np.argmax(d, axis=1)]
        r.append([e for e in decoded if e])
    return r


T = TypeVar("T")
_alphabet_cache = {}


@dataclass
class SequenceResult(Generic[T]):
    logits: np.ndarray = field(repr=False)
    alphabet: np.ndarray = field(repr=False)
    scale: float = field(repr=False)

    result: List[T] = field(init=False)
    positions: List[int] = field(init=False)

    def __post_init__(self):
        self.amax = np.argmax(self.logits, axis=1)
        self.result = [e for e in self.alphabet[self.amax] if e]
        self.positions = [round(x * self.scale, 1) for x in np.where(self.amax != len(self.alphabet) - 1)[0]]

    def __getitem__(self, i) -> T:
        return self.result[i]

    def __len__(self) -> int:
        return len(self.result)

    def infer_gaps(self, min_gap: int, gap_item: T = " ") -> List[T]:
        r = []
        last: Optional[int] = None
        for x, e in zip(self.positions, self.result):
            if last and x - last > min_gap:
                last = None
                r.append(gap_item)
            else:
                last = x
            r.append(e)
        return r

    def split_gaps(self, min_gap: int) -> List[List[T]]:
        r = [[]]
        for e in self.infer_gaps(min_gap, None):
            if e:
                r[-1].append(e)
            else:
                r.append([])
        return r

    @property
    def string(self) -> str:
        return "".join(self.result)

    def words(self, min_gap: int) -> List[str]:
        return ["".join(w) for w in self.split_gaps(min_gap)]


class TFLiteSequenceModel(TFLiteModel, Generic[T]):
    def __init__(self, path: str):
        super().__init__(os.path.join(path, "model.tflite"))
        with open(os.path.join(path, "outputs.json")) as f:
            self.outputs = json.load(f)
        self.alphabet = np.array(self.outputs["outputs"][0]["values"] + [None])
        self.scale = self.input_details[0]["shape"][2] / self.output_details[0]["shape"][1]
        print(self.scale)

    def predict(self, inps: np.ndarray) -> List[SequenceResult[T]]:
        if not isinstance(inps, np.ndarray):
            inps = np.array(inps)
        logits = super().predict(inps.astype(np.float32))
        return [SequenceResult(l, self.alphabet, self.scale) for l in logits]

    # def heropos2img(self, x: int) -> int:
    #     unmaxpool = x * self.model_meta['heroes_hidden/input_maxpool/config/pool_size/1']
    #     shared_pos = unmaxpool * self.model_meta['heroes_hidden/output_maxpool/config/pool_size/1']
    #     return shared_pos
    #
    # def img2textpos(self, x: int) -> int:
    #     maxpooled = x // self.model_meta['text_hidden/output_maxpool/config/pool_size/1']
    #     return maxpooled


def main():
    m = TFLiteSequenceModel(r"C:\Users\simon\overtrack_2\training\apex_cl_showpos\apex_cl_showpos\checkpoint.1")
    im2 = cv2.imread("C:/Users/simon/overtrack_2/Apex_OCR/samples/1,194.71 -11545.00 14757.46.png")

    r = m.predict([im2, im2])

    print(r)


if __name__ == "__main__":
    main()
