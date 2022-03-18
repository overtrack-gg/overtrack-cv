from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Eliminations:
    eliminations: List[str]
    eliminated_by: Optional[str] = None
