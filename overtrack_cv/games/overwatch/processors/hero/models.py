from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Hero:
    hero: Optional[str]
    ult: Optional[int]

    potg: bool
    spectating: bool
    killcam: bool
