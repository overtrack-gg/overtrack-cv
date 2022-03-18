from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Player:
    hero: str
    name: str
    name_match: Optional[float] = None
    blue_team: Optional[bool] = None

    index: Optional[int] = None


@dataclass(frozen=True)
class KillRow:
    left: Optional[Player]
    right: Player
    y: int
    resurrect: Optional[bool] = False
    headshot: bool = False
    assists: List[str] = field(default_factory=list)
    ability: Optional[str] = None
    from_killcam: Optional[bool] = None


@dataclass
class Killfeed:
    kills: List[KillRow] = field(default_factory=list)
    unknown_rows: List[int] = field(default_factory=list)
