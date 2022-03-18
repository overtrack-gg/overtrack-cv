from typing import List, NamedTuple, Optional


class Player(NamedTuple):
    hero: Optional[str]
    match: Optional[float]

    selected: bool

    name: Optional[str]
    # ult: int
    # health: int


class SpectatorBar(NamedTuple):
    left_team: List[Optional[Player]]
    right_team: List[Optional[Player]]
