import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

import cv2
import numpy as np

from overtrack_cv.games.apex.apex_frame_data import ApexFrameData
from overtrack_cv.util import s2ts

_init_time = time.time()


class SerializableArray:
    def __init__(self, array):
        self.array = array

    def finalize(self):
        return self.array


@dataclass
class CurrentGame:
    started: float = field(default_factory=time.time)

    def __str__(self):
        return f"CurrentGame(_id={id(self)}, started={self.started})"

    __repr__ = __str__


class Timings(Dict[str, float]):
    @property
    def total(self) -> float:
        return sum([v for k, v in self.items() if k not in ["wait", "fetch", "in_queue"]])

    def __setitem__(self, key: str, value: float) -> None:
        super().__setitem__(key, round(value, 3))

    def __str__(self) -> str:
        return str({k: round(v, 3) for k, v in dict(TOTAL=self.total, **self).items()})

    __repr__ = __str__


class Frame(Dict[str, Any]):
    def __init__(self, timestamp: float, **kwargs: Any) -> None:
        super().__init__(kwargs)
        self.timestamp: float = timestamp
        if "image" not in kwargs:
            self.image: np.ndarray = None
        if "debug_image" not in kwargs:
            self.debug_image: Optional[np.ndarray] = None
        if "_image_yuv" not in kwargs:
            self._image_yuv: Optional[np.ndarray] = None

        self.apex = ApexFrameData()

    # typing hints - access to fields is through object-level dict or __getattr__
    if TYPE_CHECKING:
        timestamp: float
        relative_timestamp: float
        timestamp_str: str
        relative_timestamp_str: str

        game_time: float
        current_game: CurrentGame

        frame_no: int

        image: Optional[np.ndarray]
        debug_image: Optional[np.ndarray]

        apex: ApexFrameData
        timings: Timings

    @classmethod
    def create(
        cls,
        image: np.ndarray,
        timestamp: float,
        debug: bool = False,
        timings: Optional[Dict[str, float]] = None,
        **data: Any,
    ) -> "Frame":
        if image.dtype != np.uint8:
            raise TypeError(f"image must have type uint8 but had type { image.dtype }")

        if image.shape[0] != 1080 or image.shape[2] != 3:
            raise TypeError(f"image must have shape (1080, *, 3) but had { image.shape }")

        f = cls.__new__(cls)
        f.image = image
        f._image_yuv: Optional[np.ndarray] = None

        f.timestamp = timestamp
        f.timestamp_str = (
            datetime.utcfromtimestamp(timestamp).strftime("%Y/%m/%d %H:%M:%S.") + f"{timestamp % 1 :.2f}"[2:]
        )

        if "relative_timestamp" in data:
            relative_timestamp = data["relative_timestamp"]
        else:
            relative_timestamp = f.timestamp - _init_time
        f.relative_timestamp = relative_timestamp
        f.relative_timestamp_str = f"{s2ts(relative_timestamp)}." + f"{relative_timestamp % 1 :.2f}"[2:]

        f.timings = Timings(**timings if timings else {})
        if debug:
            f.debug_image = image.copy()

            s = f"{f.timestamp_str}"

            if "relative_timestamp_str" in f:
                s += f" | {f.relative_timestamp_str}"
                if "image_no" in data:
                    s += f' ({data["image_no"]})'

            if data.get("offset_from_now") is not None:
                s += f' | offset: {s2ts(data["offset_from_now"])}'

            if "source" in data:
                s += f' | {data["source"]}'
            if "source_frame_no" in data:
                s += f' +{data["source_frame_no"]}'
            if "source_timestamp" in data:
                s += f'/+{data["source_timestamp"]:1.2f}s'

            if "offset_from_last" in data and data["offset_from_last"] is not None:
                s += f' | +{data["offset_from_last"]:1.2f}s'

            for t, c in (5, (0, 0, 0)), (1, (255, 0, 255)):
                cv2.putText(f.debug_image, s, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, c, t)
        else:
            f.debug_image = None
        f.update(data)

        f.apex = ApexFrameData()

        return f

    @property
    def debug(self) -> bool:
        return self.get("debug_image") is not None

    def strip(self) -> "Frame":
        """
        Remove all top-level numpy arrays
        """
        for k in "image", "debug_image", "_image_yuv":
            if k in self:
                del self[k]
        for k in list(self.keys()):
            if isinstance(self.get(k), np.ndarray):
                del self[k]
        return self

    def copy(self) -> "Frame":
        return Frame(**self)

    @property
    def image_yuv(self) -> np.ndarray:
        if self._image_yuv is None:
            super().__setitem__("_image_yuv", cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV))
        return self._image_yuv

    def __getattr__(self, item: str) -> Any:
        if item not in self:
            raise AttributeError("Frame does not (yet?) have attribute %r" % (item,))
        return self[item]

    def __setattr__(self, key: str, value: Any) -> None:
        self.__setitem__(key, value)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self:
            raise ValueError(f'Cannot add item "{ key }": already exists')
        super().__setitem__(key, value)
