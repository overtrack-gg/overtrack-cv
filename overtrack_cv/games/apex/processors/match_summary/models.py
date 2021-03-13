from dataclasses import dataclass
from typing import Optional

from overtrack_cv.core.uploadable_image import UploadableImage


@dataclass
class XPStats:
    won: bool = False
    top3_finish: bool = False
    time_survived: Optional[int] = None
    kills: Optional[int] = None
    damage_done: Optional[int] = None
    revive_ally: Optional[int] = None
    respawn_ally: Optional[int] = None


@dataclass
class ScoreReport:
    entry_rank: Optional[str] = None
    kills: Optional[int] = None
    placement: Optional[int] = None
    rp_adjustment: Optional[int] = None
    current_rp: Optional[int] = None


@dataclass
class MatchSummary:
    placed: int
    xp_stats: Optional[XPStats] = None
    score_report: Optional[ScoreReport] = None

    image: Optional[UploadableImage] = None
