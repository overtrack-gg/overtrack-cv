from typing import Optional, Tuple, Union

import numpy as np
import tflite_runtime.interpreter as tflite


class TFLiteModel:
    def __init__(self, path: str):
        self.interpreter = tflite.Interpreter(path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        assert len(self.input_details) == 1, "TFLiteModel only supports models with a single input"
        self.interpreter.allocate_tensors()

    def predict(
        self, inps: np.ndarray, batch_size: Optional[int] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
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
