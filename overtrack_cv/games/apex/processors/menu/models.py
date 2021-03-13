from dataclasses import dataclass
from typing import Optional, Tuple

from overtrack_cv.core.uploadable_image import UploadableImage


@dataclass
class PlayMenu:
    player_name: str
    squadmates: Tuple[Optional[str], Optional[str]]
    ready: bool

    rank_text: Optional[str] = None
    rp_text: Optional[str] = None

    rp_image: Optional[UploadableImage] = None
