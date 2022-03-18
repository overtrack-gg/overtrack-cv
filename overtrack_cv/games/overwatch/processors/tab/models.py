from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from overtrack_cv.games.overwatch.processors.endgame import Stats


@dataclass(frozen=True)
class NameImages:
    blue_team: List[np.ndarray]
    red_team: List[np.ndarray]
    ult_images: List[np.ndarray]
    player_name_image: np.ndarray
    player_hero_image: np.ndarray
    hero_icons_blue: List[np.ndarray]
    hero_icons_red: List[np.ndarray]


@dataclass(frozen=True)
class TabScreen:
    map: Optional[str]
    mode: Optional[str]

    blue_team: List[str]
    blue_team_hero: List[Optional[str]]
    blue_team_ults: List[int]
    red_team: List[str]
    red_team_hero: List[Optional[str]]

    player_name: str
    player_hero: Optional[str]

    stats: Optional[Stats]
