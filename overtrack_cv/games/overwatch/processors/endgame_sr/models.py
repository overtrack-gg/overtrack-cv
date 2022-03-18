from dataclasses import dataclass
from typing import Optional

from overtrack_cv.core.uploadable_image import UploadableImage


@dataclass
class EndgameSR:
    sr: int

    image: Optional[UploadableImage] = None
