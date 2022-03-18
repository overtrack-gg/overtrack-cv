from dataclasses import dataclass
from typing import Optional

from overtrack_cv.games.valorant.processors.agent_select import AgentSelect
from overtrack_cv.games.valorant.processors.home_screen import HomeScreen
from overtrack_cv.games.valorant.processors.killfeed.models import Killfeed
from overtrack_cv.games.valorant.processors.postgame.models import Postgame, Scoreboard
from overtrack_cv.games.valorant.processors.timer.models import Timer
from overtrack_cv.games.valorant.processors.top_hud import TopHud


@dataclass
class ValorantFrameData:
    home_screen: Optional[HomeScreen] = None
    timer: Optional[Timer] = None
    top_hud: Optional[TopHud] = None
    agent_select: Optional[AgentSelect] = None
    postgame: Optional[Postgame] = None
    scoreboard: Optional[Scoreboard] = None
    killfeed: Optional[Killfeed] = None
