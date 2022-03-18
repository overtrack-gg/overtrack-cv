from dataclasses import dataclass
from typing import List, Optional

from overtrack_cv.core.uploadable_image import UploadableImage


@dataclass(frozen=True)
class RoleSelect:
    placement_text: List[str]
    sr_text: List[str]
    account_name: Optional[str] = None
    grouped: Optional[bool] = None

    image: Optional[UploadableImage] = None
