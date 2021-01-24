from dataclasses import dataclass
from typing import Optional

# from overtrack_cv.games.apex.apex_metadata import ApexClientMetadata
# from overtrack_cv.games.apex.combat import CombatLog
# from overtrack_cv.games.apex.match_status.models import MatchStatus
# from overtrack_cv.games.apex.match_summary import MatchSummary
# from overtrack_cv.games.apex.menu.models import PlayMenu
# from overtrack_cv.games.apex.minimap import Minimap
# from overtrack_cv.games.apex.squad.models import Squad
# from overtrack_cv.games.apex.squad_summary import SquadSummary
# from overtrack_cv.games.apex.weapon import Weapons
# from overtrack_cv.games.apex.your_squad import YourSquad, YourSelection, ChampionSquad


@dataclass
class ApexFrameData:
    game_time: Optional[float] = None

    match_summary_match: Optional[float] = None
    apex_play_menu_match: Optional[float] = None
    squad_match: Optional[float] = None
    squad_summary_match: Optional[float] = None
    your_squad_match: Optional[float] = None

    # match_status: Optional[MatchStatus] = None
    # match_summary: Optional[MatchSummary] = None
    # apex_play_menu: Optional[PlayMenu] = None
    # squad: Optional[Squad] = None
    # squad_summary: Optional[SquadSummary] = None
    # weapons: Optional[Weapons] = None
    # your_squad: Optional[YourSquad] = None
    # your_selection: Optional[YourSelection] = None
    # champion_squad = ChampionSquad
    # combat_log: Optional[CombatLog] = None
    # minimap: Optional[Minimap] = None
    # apex_metadata: Optional[ApexClientMetadata] = None
