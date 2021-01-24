from dataclasses import dataclass
from typing import Optional, Tuple

from overtrack_cv.core.uploadable_image import UploadableImage


@dataclass
class PlayerStats:
    name: str

    kills: Optional[int]
    damage_dealt: Optional[int]
    survival_time: Optional[int]
    players_revived: Optional[int]
    players_respawned: Optional[int]


@dataclass
class SquadSummary:
    champions: bool
    squad_kills: Optional[int]
    player_stats: Tuple[PlayerStats, ...]

    placed: Optional[int] = None

    elite: Optional[bool] = False
    mode: Optional[str] = None

    image: Optional[UploadableImage] = None
