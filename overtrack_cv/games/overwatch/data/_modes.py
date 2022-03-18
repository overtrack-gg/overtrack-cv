from dataclasses import dataclass
from typing import List


@dataclass
class Mode:
    name: str
    is_arcade: bool = False
    is_vs_ai: bool = False

    @property
    def code(self) -> str:
        return self.name.upper().replace(" ", "")


competitive = Mode("Competitive Play")
competitive_open_queue = Mode("Competitive Play (Open Queue)")
quickplay = Mode("Quick Play")
custom_game = Mode("Custom Game")
quickplay_classic = Mode("Quick Play Classic", is_arcade=True)
arcade_modes = [
    Mode("Team Deathmatch", is_arcade=True),
    Mode("Mystery Heroes", is_arcade=True),
    quickplay_classic,
    Mode("Elimination", is_arcade=True),
    Mode("Petra Deathmatch", is_arcade=True),
    Mode("CTF: Ayutthaya Only", is_arcade=True),
]
ai_modes = [
    Mode("VS AI Easy", is_vs_ai=True),
    Mode("VS AI Medium", is_vs_ai=True),
    Mode("VS AI Hard", is_vs_ai=True),
]

tutorial_mode = Mode("Tutorial")
practice = Mode("Practice")
unknown_mode = Mode("Unknown")
replay_mode = Mode("Replay")

modes: List[Mode] = [
    competitive,
    quickplay,
    custom_game,
] + arcade_modes
