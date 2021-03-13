from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SharedMemorySource:
    original_size: Tuple[int, int]
    width: int
    height: int
    linesize: int
    shmem_name: Optional[str] = None
