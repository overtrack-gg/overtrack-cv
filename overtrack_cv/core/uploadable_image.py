import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from overtrack_cv.util.logging_config import intermittent_log

if TYPE_CHECKING:
    import numpy as np

MAXLEN = 15

active_images: Dict[str, "UploadableImage"] = {}
logger = logging.getLogger("Image")


def lazy_upload(
    key: str,
    image: "np.ndarray",
    timestamp: float,
    maxlen: int = MAXLEN,
    selection="middle",
) -> "UploadableImage":
    if key not in active_images:
        active_images[key] = UploadableImage(key, maxlen, selection=selection)

    active_images[key].append(image, timestamp)
    return active_images[key]


def lazy_upload_unique(key: str, image: "np.ndarray") -> "UploadableImage":
    now = time.time()
    ukey = f"{key}_{now:.2f}"
    active_images[ukey] = UploadableImage(key, 1)

    active_images[ukey].append(image, now)
    return active_images[ukey]


class UploadableImage:
    def __init__(self, key: str, maxlen: int, selection="middle"):
        self.key = key
        self.images = deque(maxlen=maxlen)
        self.dropped = 0
        self.selection = selection
        self.url = None
        logger.info(f"Created {self}")

    def append(self, image: "np.ndarray", timestamp: float) -> None:
        intermittent_log(
            logger,
            f"{self}: Adding image @ {timestamp:.1f}",
            frequency=2 if not self.dropped else self.dropped,
            caller_extra_id=self.key,
        )
        if len(self.images) == self.images.maxlen:
            self.dropped += 1
        self.images.append((timestamp, image))

    def make_single(self) -> "np.ndarray":
        # import numpy as np
        # return np.mean([i for t, i in self.images], axis=0).astype(np.uint8)
        if self.selection == "first":
            return self.images[0][1]
        elif self.selection == "last":
            return self.images[-1][1]
        else:
            if self.selection != "middle":
                logger.warning(f'Got unknown selection mode {self.selection} - using "middle"')
            return [i for t, i in self.images][len(self.images) // 2]

    @property
    def timestamps(self) -> List[int]:
        return [t for t, i in self.images]

    @property
    def count(self) -> int:
        return len(self.images)

    def __del__(self):
        try:
            logger.info(f"Disposing of {self}")
        except:
            print(f"Disposing of {self}")

    def __str__(self) -> str:
        if self.url is None:
            ustr = ", uploaded=False"
        else:
            ustr = f", url={self.url}"
        return f"Image(id={id(self)}, key={self.key}{ustr}, count={len(self.images)})"

    __repr__ = __str__

    def _typeddump(self) -> Dict:
        # if self.url is None:
        #     logger.warning(f"Dumping image {self} before it is uploaded")
        return {"key": self.key, "url": self.url, "timestamps": self.timestamps}


@dataclass
class UploadedImage:
    key: str
    url: Optional[str]
    timestamps: List[float]
