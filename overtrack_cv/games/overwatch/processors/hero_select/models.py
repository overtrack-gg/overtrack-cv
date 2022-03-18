from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AssembleYourTeam:
    map: Optional[str]
    mode: Optional[str]
    blue_names: List[Optional[str]]
    is_in_queue: bool = False
