from dataclasses import dataclass
from typing import Optional

from overtrack_cv.games.apex.apex_metadata import ApexClientMetadata
from overtrack_cv.games.apex.processors.combat import CombatLog
from overtrack_cv.games.apex.processors.coordinates.models import Coordinates
from overtrack_cv.games.apex.processors.map_loading.models import MapLoading
from overtrack_cv.games.apex.processors.match_status.models import MatchStatus
from overtrack_cv.games.apex.processors.match_summary import MatchSummary
from overtrack_cv.games.apex.processors.menu.models import PlayMenu
from overtrack_cv.games.apex.processors.minimap import Minimap
from overtrack_cv.games.apex.processors.squad.models import Squad
from overtrack_cv.games.apex.processors.squad_summary import SquadSummary
from overtrack_cv.games.apex.processors.weapon import Weapons
from overtrack_cv.games.apex.processors.your_squad import (
    ChampionSquad,
    YourSelection,
    YourSquad,
)


@dataclass
class ApexFrameData:
    match_summary_match: Optional[float] = None
    apex_play_menu_match: Optional[float] = None
    squad_match: Optional[float] = None
    squad_summary_match: Optional[float] = None
    your_squad_match: Optional[float] = None

    match_status: Optional[MatchStatus] = None
    match_summary: Optional[MatchSummary] = None
    apex_play_menu: Optional[PlayMenu] = None
    squad: Optional[Squad] = None
    squad_summary: Optional[SquadSummary] = None
    weapons: Optional[Weapons] = None
    your_squad: Optional[YourSquad] = None
    your_selection: Optional[YourSelection] = None
    champion_squad: Optional[ChampionSquad] = None
    combat_log: Optional[CombatLog] = None
    minimap: Optional[Minimap] = None
    coordinates: Optional[Coordinates] = None
    map_loading: Optional[MapLoading] = None
    apex_metadata: Optional[ApexClientMetadata] = None
