from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from overtrack_cv.frame import SerializableArray
from overtrack_cv.util import round_floats


@dataclass(frozen=True)
@round_floats
class RingsComposite:
    images: Dict[int, SerializableArray] = field(default_factory=dict)

    def __str__(self):
        return (
            "RingsComposite("
            + ", ".join(
                f"{k}=Composite(shape={v.array.shape}, pixels=~{(v.array >= 1).sum()})"
                for k, v in self.images.items()
            )
            + ")"
        )

    __repr__ = __str__


@dataclass(frozen=True)
@round_floats
class Circle:
    coordinates: Tuple[float, float]
    r: float

    residual: Optional[float] = None
    points: Optional[int] = None


@dataclass(frozen=True)
@round_floats
class Location:
    coordinates: Tuple[int, int]
    match: float
    bearing: Optional[int]

    zoom: Optional[float] = None

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]


@dataclass(frozen=True)
class Minimap:
    location: Location
    inner_circle: Optional[Circle]
    outer_circle: Optional[Circle]
    spectate: bool = False
    rings_composite: Optional[RingsComposite] = None

    version: int = 0
