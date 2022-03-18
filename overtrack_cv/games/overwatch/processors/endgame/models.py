from dataclasses import dataclass, fields
from typing import Dict, Optional


@dataclass(frozen=True)
class Stats:
    hero: str
    eliminations: Optional[int]
    objective_kills: Optional[int]
    objective_time: Optional[int]
    hero_damage_done: Optional[int]
    healing_done: Optional[int]
    deaths: Optional[int]
    hero_specific_stats: Optional[Dict[str, Optional[int]]]

    @property
    def all_valid(self) -> bool:
        for f in fields(self):
            if getattr(self, f.name) is None:
                return False
        return True

    # TODO: flag if has more stats or if this is the only stat)


@dataclass(frozen=True)
class Endgame:
    result: str
    map: str
    stats: Optional[Stats]
