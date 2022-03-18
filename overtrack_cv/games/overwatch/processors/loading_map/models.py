from dataclasses import dataclass
from typing import List, Optional

from overtrack_cv.core.uploadable_image import UploadableImage


@dataclass(frozen=True)
class Teams:
    blue_team: List[str]
    red_team: List[str]

    blue_ranks: Optional[List[Optional[str]]] = None
    red_ranks: Optional[List[Optional[str]]] = None
    average_sr: Optional[List[str]] = None


@dataclass(frozen=True)
class LoadingMap:
    map: str
    game_mode: str
    teams: Optional[Teams]

    is_role_queue: Optional[bool] = False
    is_in_queue: Optional[bool] = False

    image: Optional[UploadableImage] = None
