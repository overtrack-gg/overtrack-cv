import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from overtrack_cv.core import arrayops
from overtrack_cv.games.apex import data


@dataclass
class Squad:
    name: Optional[str]
    champion: List[float]
    squadmate_names: Tuple[Optional[str], Optional[str]]
    squadmate_champions: Tuple[List[float], List[float]]

    bodyshield_level: Optional[int] = None
    helmet_level: Optional[int] = None
    health: Optional[float] = None
    shields: Optional[float] = None

    @property
    def champion_name(self) -> str:
        return list(data.champions.keys())[arrayops.argmax(self.champion)]

    @property
    def squadmate_champions_names(self) -> Tuple[str, str]:
        # noinspection PyTypeChecker
        return tuple(list(data.champions.keys())[arrayops.argmax(arr)] for arr in self.squadmate_champions)

    def __str__(self) -> str:
        strdcls = dataclasses.replace(
            self,
            champion=f"{self.champion_name} ({np.argmax(self.champion):1.4f})",
            squadmate_champions=(
                f"{self.squadmate_champions_names[0]} ({np.max(self.squadmate_champions[0])}), "
                f"{self.squadmate_champions_names[1]} ({np.max(self.squadmate_champions[1])})), "
            ),
        )
        return (
            self.__class__.__qualname__
            + "("
            + ", ".join([f"{f.name}={getattr(strdcls, f.name)!r}" for f in dataclasses.fields(self)])
            + ")"
        )

    __repr__ = __str__
