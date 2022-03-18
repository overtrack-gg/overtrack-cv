from dataclasses import dataclass
from typing import List, Optional, Tuple

from overtrack_models.dataclasses.valorant import AgentName

from overtrack_cv.core.uploadable_image import UploadableImage


@dataclass
class AgentSelect:
    agent: AgentName
    locked_in: bool

    map: Optional[str]
    game_mode: Optional[str] = None

    player_names: Optional[List[str]] = None
    agents: Optional[List[Optional[str]]] = None
    ranks: Optional[List[Tuple[str, float]]] = None

    image: Optional[UploadableImage] = None
