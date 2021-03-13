import glob
import os

import cv2

from overtrack_cv.core import imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.apex.processors.map_loading.models import MapLoading
from overtrack_cv.games.processor import Processor


class MapLoadingProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))
    TEMPLATES = {
        str(os.path.basename(p)).split(".")[0]: cv2.imread(p, 0)
        for p in glob.glob(os.path.join(os.path.dirname(__file__), "data", "*.png"))
    }
    REQUIRED_MATCH = 0.9

    def eager_load(self) -> None:
        self.REGIONS.eager_load()

    def process(self, frame: Frame) -> bool:
        y = frame.image_yuv[:, :, 0]
        region = self.REGIONS["map_name"].extract_one(y)
        _, thresh = cv2.threshold(region, 200, 255, cv2.THRESH_BINARY)

        match, map_name = imageops.match_templates(
            thresh, self.TEMPLATES, method=cv2.TM_CCORR_NORMED, required_match=0.95
        )
        if match > self.REQUIRED_MATCH:
            frame.apex.map_loading = MapLoading(map_name)
            return True

        return False


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    test_processor(MapLoadingProcessor(), "apex.map_loading")
