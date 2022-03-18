from dataclasses import dataclass
from typing import Optional

from overtrack_cv.core.uploadable_image import UploadableImage


@dataclass(frozen=True)
class MainMenu:
    version: str
    # TODO: rejoin


@dataclass(frozen=True)
class PlayMenu:
    placements: bool
    sr: Optional[int]
    # TODO: group SR

    image: Optional[UploadableImage] = None
