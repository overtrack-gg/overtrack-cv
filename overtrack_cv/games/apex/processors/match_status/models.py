from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MatchStatus:
    squads_left: Optional[int]
    players_alive: Optional[int]
    kills: Optional[int]

    ranked: Optional[bool] = None

    assists: Optional[int] = None
    rank_badge_matches: Optional[Tuple[float, ...]] = None
    rank_text: Optional[str] = None
    rp_text: Optional[str] = None

    streak: Optional[int] = None
    solos_players_left: Optional[int] = None

    mode: Optional[str] = None
