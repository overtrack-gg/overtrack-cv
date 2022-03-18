import string
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

from overtrack_cv.util.compat import Literal

MapType = Literal["Escort", "Hybrid", "Assault", "Control", "Training"]
StageName = Literal["A", "B", "C"]


class ControlStage(NamedTuple):
    letter: StageName
    name: str


@dataclass
class Map:
    name: str
    type: Optional[MapType]

    @property
    def code(self) -> str:
        return "".join(c for c in self.name.replace(" ", "_") if c in string.ascii_letters + string.digits + "_")

    @property
    def is_known(self) -> bool:
        return self.type is not None


@dataclass
class ControlMap(Map):
    stages: List[ControlStage]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, type={self.type}, stages=...)"

    @property
    def stage_dict(self) -> Dict[StageName, str]:
        return {s.letter: f"{s.name} ({s.letter})" for s in self.stages}


practice_range = Map(name="Practice Range", type="Training")

maps: List[Map] = [
    Map(name="Hanamura", type="Assault"),
    Map(name="Horizon Lunar Colony", type="Assault"),
    Map(name="Paris", type="Assault"),
    Map(name="Temple of Anubis", type="Assault"),
    Map(name="Volskaya Industries", type="Assault"),
    Map(name="Dorado", type="Escort"),
    Map(
        name="Havana",
        type="Escort",
    ),
    Map(name="Junkertown", type="Escort"),
    Map(name="Rialto", type="Escort"),
    Map(name="Route 66", type="Escort"),
    Map(name="Watchpoint: Gibraltar", type="Escort"),
    Map(name="Blizzard World", type="Hybrid"),
    Map(name="Eichenwalde", type="Hybrid"),
    Map(name="Hollywood", type="Hybrid"),
    Map(name="King's Row", type="Hybrid"),
    Map(name="Numbani", type="Hybrid"),
    ControlMap(
        name="Busan",
        type="Control",
        stages=[
            ControlStage(letter="A", name="Downtown"),
            ControlStage(letter="B", name="Sanctuary"),
            ControlStage(letter="C", name="Meka Base"),
        ],
    ),
    ControlMap(
        name="Ilios",
        type="Control",
        stages=[
            ControlStage(letter="A", name="Lighthouse"),
            ControlStage(letter="B", name="Well"),
            ControlStage(letter="C", name="Ruins"),
        ],
    ),
    ControlMap(
        name="Lijiang Tower",
        type="Control",
        stages=[
            ControlStage(letter="A", name="Night Market"),
            ControlStage(letter="B", name="Garden"),
            ControlStage(letter="C", name="Control Center"),
        ],
    ),
    ControlMap(
        name="Oasis",
        type="Control",
        stages=[
            ControlStage(letter="A", name="City Center"),
            ControlStage(letter="B", name="University"),
            ControlStage(letter="C", name="Gardens"),
        ],
    ),
    ControlMap(
        name="Nepal",
        type="Control",
        stages=[
            ControlStage(letter="A", name="Village"),
            ControlStage(letter="B", name="Shrine"),
            ControlStage(letter="C", name="Sanctum"),
        ],
    ),
    # Map(
    #     name='Estádio das Rãs',
    #     type='Lúcioball'
    # ),
]
