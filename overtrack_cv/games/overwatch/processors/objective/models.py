from dataclasses import Field, dataclass, fields
from typing import List, Optional, Tuple
from warnings import warn

import numpy as np

# from overtrack.util import round_floats, cached_property
cached_property = property

PROBABILITY_PRECISION = 2

STATE_NOT_OVERWATCH = 0
STATE_CHECKPOINT_ASSEMBLE = 1
STATE_CHECKPOINT_PREPARE = 2
STATE_CHECKPOINT_ATTACK = 3
STATE_CHECKPOINT_DEFEND = 4
STATE_KOTH_ASSEMBLE = 5
STATE_KOTH_PREPARE = 6
STATE_KOTH_LOCKED = 7
STATE_OVERTIME = 8

CHECKPOINT_NONE = 0
CHECKPOINT_POINT = 1
CHECKPOINT_PAYLOAD = 2

SCORES = [None] + list(range(20)) + [-1]

ScoreTuple = Tuple[tuple([float] * (1 + 20 + 1))]


@dataclass(frozen=True)
class Objective:
    p_state: Tuple[float, float, float, float, float, float, float, float, float]
    p_competitive: float

    p_contested: float

    p_checkpoint_mode: Tuple[float, float, float]
    p_payload_direction: Tuple[float, float, float]
    p_controlpoint_ticks: Tuple[float, float, float]

    p_koth_point: Tuple[float, float, float, float]
    p_koth_owner: Tuple[float, float, float]

    p_score_blue: ScoreTuple
    p_score_red: ScoreTuple

    @classmethod
    def from_result(cls, result: List[np.ndarray]) -> "Objective":
        args = {}
        field: Field
        for r, field in zip(result, fields(cls)):
            r = r[0]
            if len(r) == 2:
                args[field.name] = round(float(r[1]), PROBABILITY_PRECISION)
            else:
                args[field.name] = tuple([round(e, PROBABILITY_PRECISION) for e in r.tolist()])
        return cls(**args)

    @cached_property
    def state(self) -> int:
        return int(np.argmax(self.p_state))

    @cached_property
    def checkpoint_mode(self) -> int:
        return int(np.argmax(self.p_checkpoint_mode))

    # Intepereted states

    @property
    def overwatch(self) -> bool:
        return self.state > 0 or bool(self.koth_point)

    @property
    def competitive(self) -> bool:
        return self.p_competitive > 0.5

    @property
    def koth(self) -> bool:
        return self.state in [STATE_KOTH_ASSEMBLE, STATE_KOTH_PREPARE, STATE_KOTH_LOCKED] or bool(self.koth_point)

    @property
    def checkpoint(self) -> bool:
        return self.state in [
            STATE_CHECKPOINT_ASSEMBLE,
            STATE_CHECKPOINT_PREPARE,
            STATE_CHECKPOINT_ATTACK,
            STATE_CHECKPOINT_DEFEND,
        ]

    @cached_property
    def payload(self) -> bool:
        return self.checkpoint_mode == CHECKPOINT_PAYLOAD

    @property
    def assault_point(self) -> bool:
        return self.checkpoint_mode == CHECKPOINT_POINT

    @property
    def started(self) -> bool:
        return self.overwatch and (
            self.state in [STATE_CHECKPOINT_ATTACK, STATE_CHECKPOINT_DEFEND]
            or (self.state in [STATE_KOTH_LOCKED] or bool(self.koth_point))
        )

    @property
    def contested(self) -> bool:
        return self.p_contested > 0.5

    @property
    def overtime(self) -> bool:
        return self.state == STATE_OVERTIME

    @property
    def attacking(self) -> Optional[bool]:
        if self.state == STATE_CHECKPOINT_ATTACK:
            return True
        elif self.state == STATE_CHECKPOINT_DEFEND:
            return False
        elif self.state in [STATE_OVERTIME, STATE_CHECKPOINT_ASSEMBLE, STATE_CHECKPOINT_PREPARE]:
            return None
        else:
            warn("Trying to get attacking status for KOTH objective")
            return None

    @property
    def koth_point(self) -> Optional[int]:
        return [None, "A", "B", "C"][int(np.argmax(self.p_koth_point))]

    @property
    def koth_owner(self) -> Optional[int]:
        return [None, "blue", "red"][int(np.argmax(self.p_koth_owner))]

    @cached_property
    def ticks(self) -> Optional[int]:
        if self.assault_point or self.overtime:
            return int(np.argmax(self.p_controlpoint_ticks))
        else:
            warn("Trying to get ticks for incompatible objective")
            return None

    @cached_property
    def payload_direction(self) -> Optional[int]:
        if self.payload or self.overtime:
            i = np.argmax(self.p_payload_direction)
            if i == 1:
                return 1
            elif i == 2:
                return -1
            else:
                return 0
        return None

    @cached_property
    def score_blue(self) -> Optional[int]:
        return SCORES[np.argmax(self.p_score_blue)]

    @cached_property
    def score_red(self) -> Optional[int]:
        return SCORES[np.argmax(self.p_score_red)]
