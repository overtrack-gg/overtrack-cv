from dataclasses import dataclass
from typing import Optional

from overtrack_cv.util import ts2s

ROUND_START_DELAY = 4.0
CLOSING_DELAY = 5.0


@dataclass
class Round:
    index: int

    ring_countdown: float
    close_time: Optional[float]

    radius: Optional[int]

    start_time: float = -1
    end_time: float = -1

    @property
    def final_outer_radius(self) -> Optional[int]:
        if self.radius:
            return self.radius + 3
        else:
            return None


rounds = [
    Round(0, ring_countdown=10, close_time=50 - CLOSING_DELAY, radius=None),
    Round(
        1,
        ring_countdown=ts2s("3:00"),
        close_time=ts2s("3:42"),
        radius=380,
    ),
    Round(
        2,
        ring_countdown=ts2s("2:30"),
        close_time=ts2s("1:02"),
        radius=218,
    ),
    Round(
        3,
        ring_countdown=ts2s("2:14"),
        close_time=ts2s("0:41"),
        radius=132,
    ),
    Round(
        4,
        ring_countdown=ts2s("1:59"),
        close_time=ts2s("0:33"),
        radius=65,
    ),
    Round(
        5,
        ring_countdown=ts2s("1:29"),
        close_time=25.0,
        radius=24,
    ),
    Round(
        6,
        ring_countdown=ts2s("1:28"),
        close_time=7.5,
        radius=10,
    ),
    Round(
        7,
        ring_countdown=120,
        close_time=8.0,
        radius=2,
    ),
    Round(
        8,
        ring_countdown=20,
        close_time=None,
        radius=None,
    ),
]

current_time = 0
for current_round in rounds:
    current_round.start_time = current_time
    if current_round.close_time:
        current_time += current_round.ring_countdown + CLOSING_DELAY + current_round.close_time + ROUND_START_DELAY
        current_round.end_time = current_time
    else:
        current_round.end_time = 2000


@dataclass(frozen=True)
class RoundState:
    round: int

    ring_radius: Optional[int]
    next_ring_radius: Optional[int]

    ring_index: Optional[int]
    next_ring_index: Optional[int]

    ring_closing: bool

    time_into: float
    time_to_next: float


def get_round_state(t: float) -> RoundState:
    current_time = 0
    current_ring_index = -1
    current_ring_radius = 0
    for current_round in rounds:
        next_state_time = current_time + current_round.ring_countdown + CLOSING_DELAY
        if t <= next_state_time or current_round.close_time is None:
            return RoundState(
                round=current_round.index,
                ring_radius=current_ring_radius,
                next_ring_radius=current_round.radius,
                ring_index=current_ring_index if (current_ring_index > 0) else None,
                next_ring_index=current_ring_index + 1 if (current_ring_index >= 0) else None,
                ring_closing=False,
                time_into=t - current_time,
                time_to_next=next_state_time - t,
            )

        current_time = next_state_time
        next_state_time = current_time + current_round.close_time + ROUND_START_DELAY

        if current_ring_radius and current_round.radius:
            frac = (t - current_time) / current_round.close_time
            ring_radius = max(
                current_round.radius + (current_ring_radius - current_round.radius) * (1 - frac),
                current_round.radius + 3,
            )
        else:
            ring_radius = None
        if t <= next_state_time:
            return RoundState(
                round=current_round.index,
                ring_radius=ring_radius,
                next_ring_radius=current_round.radius,
                ring_index=current_ring_index if (current_ring_index > 0) else None,
                next_ring_index=current_ring_index + 1 if (current_ring_index >= 0) else None,
                ring_closing=current_round.index != 0,
                time_into=t - current_time,
                time_to_next=next_state_time - t,
            )

        current_time = next_state_time
        if current_round.radius:
            current_ring_radius = current_round.radius + 3
        else:
            current_ring_radius = None
        current_ring_index += 1


if __name__ == "__main__":
    print()

    # current = 0
    # for r in ROUNDS:
    #     if r is None:
    #         current += COUNTDOWN + FIRST_RING_SPAWN
    #     else:
    #         print(f'Round {r.index} start: {current:.1f}')
    #         current += ROUND_START_DELAY + r.ring_countdown
    #         print(f'Round {r.index} closing: {current:.1f}')
    #         if r.close_time:
    #             current += CLOSING_DELAY + r.close_time

    import matplotlib.pyplot as plt

    ts, next_radius, current_radius, round_index, closing, current_index, next_index = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for t in range(1500):
        state = get_round_state(t)
        ts.append(t)
        current_radius.append(state.ring_radius)
        next_radius.append(state.next_ring_radius)
        round_index.append(state.round * 10)
        closing.append(state.ring_closing * -10)

        if state.ring_index:
            current_index.append(state.ring_index * 10)
        else:
            current_index.append(float("nan"))

        if state.next_ring_index:
            next_index.append(state.next_ring_index * 10)
        else:
            next_index.append(float("nan"))

    plt.figure()

    plt.plot(ts, closing, color="gray", label="closing")

    plt.plot(ts, current_index, color="orange", label="current index")
    plt.scatter(ts, current_radius, color="orange", label="current radius")

    plt.plot(ts, next_index, color="blue", label="next index")
    plt.scatter(ts, next_radius, color="blue", label="inner")

    plt.scatter(ts, round_index, color="green", label="index")
    plt.legend()
    plt.show()
