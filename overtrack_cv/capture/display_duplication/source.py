from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DisplayDuplicationSource:
    hwnd: Optional[int]

    pid: Optional[int]
    name: Optional[str]
    window_title: Optional[str]

    style: Optional[int]
    ex_style: Optional[int]
    rect: Optional[Tuple[int, int, int, int]]
    monitor: Tuple[int, int, int, int]
