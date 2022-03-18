from dataclasses import dataclass
from typing import List, Optional

from overtrack_cv.games.valorant.data import AgentName


@dataclass
class KillfeedPlayer:
    agent: AgentName
    agent_match: float
    name: str


@dataclass
class Kill:
    y: int
    row_match: float

    killer_friendly: bool
    killer: KillfeedPlayer
    killed: KillfeedPlayer

    weapon: Optional[str]
    weapon_match: float

    wallbang: bool = False
    wallbang_match: float = 0.0
    headshot: bool = False
    headshot_match: float = 0.0


@dataclass
class Killfeed:
    kills: List[Kill]

    def __iter__(self):
        return iter(self.kills)

    def __len__(self):
        return len(self.kills)
