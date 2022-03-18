from dataclasses import dataclass
from typing import Optional

from overtrack_cv.games.overwatch.overwatch_metadata import OverwatchClientMetadata
from overtrack_cv.games.overwatch.processors.eliminations.models import Eliminations
from overtrack_cv.games.overwatch.processors.endgame import Endgame
from overtrack_cv.games.overwatch.processors.endgame_sr import EndgameSR
from overtrack_cv.games.overwatch.processors.hero import Hero
from overtrack_cv.games.overwatch.processors.hero_select import AssembleYourTeam
from overtrack_cv.games.overwatch.processors.killfeed import Killfeed
from overtrack_cv.games.overwatch.processors.loading_map import LoadingMap
from overtrack_cv.games.overwatch.processors.menu import MainMenu, PlayMenu
from overtrack_cv.games.overwatch.processors.objective import Objective
from overtrack_cv.games.overwatch.processors.role_select import RoleSelect
from overtrack_cv.games.overwatch.processors.score import FinalScore, ScoreScreen
from overtrack_cv.games.overwatch.processors.spectator import SpectatorBar
from overtrack_cv.games.overwatch.processors.tab import TabScreen


@dataclass
class OverwatchFrameData:
    loading_match: Optional[float] = None
    tab_match: Optional[float] = None
    main_menu_match: Optional[float] = None
    play_menu_match: Optional[float] = None
    killcam_match: Optional[float] = None
    score_screen_match: Optional[float] = None
    final_score_match: Optional[float] = None
    endgame_match: Optional[float] = None
    endgame_sr_match: Optional[float] = None
    assemble_your_team_match: Optional[float] = None

    objective: Optional[Objective] = None
    loading_map: Optional[LoadingMap] = None
    tab_screen: Optional[TabScreen] = None
    main_menu: Optional[MainMenu] = None
    play_menu: Optional[PlayMenu] = None
    killfeed: Optional[Killfeed] = None
    spectator_bar: Optional[SpectatorBar] = None
    score_screen: Optional[ScoreScreen] = None
    final_score: Optional[FinalScore] = None
    endgame: Optional[Endgame] = None
    hero: Optional[Hero] = None
    endgame_sr: Optional[EndgameSR] = None
    assemble_your_team: Optional[AssembleYourTeam] = None
    role_select: Optional[RoleSelect] = None
    eliminations: Optional[Eliminations] = None
    overwatch_metadata: Optional[OverwatchClientMetadata] = None
    replay_metadata: Optional[object] = None
