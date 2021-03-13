from dataclasses import dataclass

from overtrack_cv.util.compat import Literal

MapName = Literal["King's Canyon", "Olympus", "World's Edge"]


@dataclass
class MapLoading:
    map: MapName
