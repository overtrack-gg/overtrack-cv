from dataclasses import dataclass
from typing import Optional, Tuple

from overtrack_cv.games.valorant.data import AgentName

OAgent = Optional[AgentName]
TeamComp = Tuple[OAgent, OAgent, OAgent, OAgent, OAgent]
Obool = Optional[bool]
FiveOBool = Tuple[Obool, Obool, Obool, Obool, Obool]
OFloat = Optional[float]
FiveOFloat = Tuple[OFloat, OFloat, OFloat, OFloat, OFloat]


@dataclass
class TopHud:
    score: Tuple[Optional[int], Optional[int]]
    teams: Tuple[TeamComp, TeamComp]
    has_ult_match: Optional[Tuple[FiveOFloat, FiveOFloat]] = None
    has_spike_match: Optional[FiveOFloat] = None

    has_ult: Optional[Tuple[FiveOBool, FiveOBool]] = None
    has_spike: Optional[FiveOBool] = None
