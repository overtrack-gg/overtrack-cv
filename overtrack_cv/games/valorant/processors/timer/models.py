import re
from dataclasses import dataclass
from typing import Optional

COUNTDOWN_PATTERN = re.compile(r"^(\d\.)|([01]:)\d\d$")


@dataclass
class Timer:
    spike_planted: bool
    buy_phase: bool
    countdown: Optional[str]

    @property
    def valid(self) -> bool:
        return (
            self.spike_planted
            or self.buy_phase
            or (self.countdown and COUNTDOWN_PATTERN.fullmatch(self.countdown.replace(" ", "")) is not None)
        )
