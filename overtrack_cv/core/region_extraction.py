import json
import logging
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Region = Tuple[int, int, int, int]


class ExtractionRegions:
    def __init__(
        self,
        name: str,
        image: Optional[np.ndarray],
        regions: List[Region] = None,
        include_when_blanking: bool = True,
    ):
        self.name = name
        self.include_when_blanking = include_when_blanking

        if image is not None:
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    # use alpha channel
                    image = image[:, :, 3]
                else:
                    # use any non-zero pixel (across all color channels)
                    _, image = cv2.threshold(np.max(image, axis=2), 1, 255, cv2.THRESH_BINARY)
            else:
                # use any non-zero pixel
                image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

            self.regions: List[Region] = []

            r, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
            for x, y, w, h, a in stats[1:]:
                if w * h != a:
                    logger.warning(f"ExtractionRegions {name} got non-rectangular region {w}*{h}!={a}")
                self.regions.append((int(x), int(y), int(w), int(h)))

            # sort regions by y, x
            self.regions = sorted(self.regions, key=lambda e: (e[1], e[0]))

        else:
            self.regions = regions

    def shunt(self, x=0, y=0) -> "ExtractionRegions":
        return ExtractionRegions(
            self.name,
            None,
            [(r[0] + x, r[1] + y, r[2], r[3]) for r in self.regions],
            include_when_blanking=self.include_when_blanking,
        )

    def extract(self, image: np.ndarray) -> List[np.ndarray]:
        image_regions = []
        for x, y, w, h in self.regions:
            if y >= image.shape[0]:
                logger.error(
                    f"ExtractionRegions {self.name} unable to extract region at y={y} from image with height={image.shape[0]}"
                )
                image_region = np.zeros((h, y), np.uint8)
            elif x >= image.shape[1]:
                logger.error(
                    f"ExtractionRegions {self.name} unable to extract region at y={y} from image with height={image.shape[0]}"
                )
                image_region = np.zeros((h, y), np.uint8)
            else:
                image_region = image[y : y + h, x : x + w]
            if image_region.shape[:2] != (h, w):
                logger.warning(
                    f"ExtractionRegions {self.name} got region of size {image_region.shape[:2]} from extraction region with h={h}, w={w} - padding image"
                )
                image_region = cv2.copyMakeBorder(
                    image_region,
                    0,
                    0,
                    w - image_region.shape[1],
                    h - image_region.shape[0],
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            image_regions.append(image_region)
        return image_regions

    def extract_one(self, image: np.ndarray) -> np.ndarray:
        return self.extract(image)[0]

    def draw(self, image: np.ndarray) -> None:
        for i, (x, y, w, h) in enumerate(self.regions):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            for t, c in (3, (0, 0, 0)), (1, (0, 255, 0)):
                cv2.putText(
                    image,
                    f"{self.name}[{i}]",
                    (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    c,
                    t,
                )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, {len(self.regions)} regions)"

    __repr__ = __str__

    def fill(self, image: np.ndarray) -> None:
        for i, (x, y, w, h) in enumerate(self.regions):
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)


class ExtractionRegionsCollection:
    def __init__(self, path: str, lazy: bool = True):
        self.path = path
        self.regions: Optional[Dict[str, ExtractionRegions]] = None
        if not lazy:
            self._ensure_loaded()

    def eager_load(self):
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self.regions is not None:
            return
        regions: Dict[str, ExtractionRegions] = {}

        altpath = self.path[:-3] + "json"
        if not os.path.exists(self.path) and os.path.exists(altpath):
            logger.info(f"Using compiled regions: {altpath}")
            with open(altpath) as f:
                regions = {k: ExtractionRegions(k, None, [tuple(e) for e in v]) for k, v in json.load(f).items()}

        else:
            with zipfile.ZipFile(self.path) as z:
                for f in z.namelist():
                    if not f.startswith("L") or not f.endswith(".png"):
                        # not a layer
                        continue
                    layer_props = f.rsplit(".", 1)[0].split(",")
                    layer_name = layer_props[3].replace("%002E", ".")
                    include_when_blanking = "%002A" not in layer_name  # %002A is '*'
                    layer_name = layer_name.replace("%002A", "")
                    if not layer_name.startswith("region."):
                        continue
                    region_name = layer_name[len("region.") :]
                    with z.open(f, "r") as fobj:
                        logger.debug("Loading region %s from %s", region_name, self.path)
                        layer = cv2.imdecode(np.frombuffer(fobj.read(), dtype=np.uint8), -1)
                        regions[region_name] = ExtractionRegions(
                            region_name,
                            layer,
                            include_when_blanking=include_when_blanking,
                        )
        self.regions = regions

    def __getitem__(self, key: str) -> ExtractionRegions:
        self._ensure_loaded()
        if key not in self.regions:
            raise KeyError(f"region {key} was not in regions: {self.regions.keys()}")
        return self.regions[key]

    def __str__(self) -> str:
        self._ensure_loaded()
        return f"{self.__class__.__name__}(regions={self.regions} regions)"

    def to_dict(self) -> Dict[str, object]:
        r = {}
        self._ensure_loaded()
        for region in self.regions.values():
            r[region.name] = region.regions
        return r

    def draw(self, image: Optional[np.ndarray]) -> None:
        if image is None:
            return
        self._ensure_loaded()
        for region in self.regions.values():
            region.draw(image)

    def blank_out(self, image: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(image)
        self._ensure_loaded()
        for region in self.regions.values():
            if region.include_when_blanking:
                region.fill(mask)
        return np.bitwise_and(image, mask)


if __name__ == "__main__":
    print(ExtractionRegionsCollection("../game/tab/data/regions/16_9.zip"))
