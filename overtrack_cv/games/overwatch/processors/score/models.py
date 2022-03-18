from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ScoreScreen:
    blue: Optional[int]
    red: Optional[int]

    round_number: Optional[int]


@dataclass(frozen=True)
class FinalScore:
    blue: Optional[int]
    red: Optional[int]
    result: Optional[str]
