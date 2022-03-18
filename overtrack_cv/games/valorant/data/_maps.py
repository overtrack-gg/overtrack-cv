from typing import List

from overtrack_cv.util.compat import Literal

MapName = Literal["Split", "Bind", "Haven", "Ascent"]


class _Maps(List[MapName]):
    split: MapName = "Split"
    bind: MapName = "Bind"
    haven: MapName = "Haven"
    ascent: MapName = "Ascent"

    def __init__(self):
        super().__init__([self.split, self.bind, self.haven, self.ascent])


maps = _Maps()

GameModeName = Literal["unrated", "competitive", "custom", "spike rush"]


class _GameModes(List[GameModeName]):
    unrated: GameModeName = "unrated"
    competitive: GameModeName = "competitive"
    custom: GameModeName = "custom"
    spike_rush = "spike rush"

    def __init__(self):
        super().__init__([self.unrated, self.competitive, self.custom, self.spike_rush])


game_modes = _GameModes()
