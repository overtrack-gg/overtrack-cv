from typing import Sequence

from overtrack_cv.games.overwatch.processors.eliminations.eliminations_processor import (
    EliminationsProcessor,
)
from overtrack_cv.games.overwatch.processors.endgame.endgame_processor import (
    EndgameProcessor,
)
from overtrack_cv.games.overwatch.processors.endgame_sr.endgame_sr_processor import (
    EndgameSRProcessor,
)
from overtrack_cv.games.overwatch.processors.hero.hero_processor import HeroProcessor
from overtrack_cv.games.overwatch.processors.hero_select.hero_select_processor import (
    HeroSelectProcessor,
)
from overtrack_cv.games.overwatch.processors.killfeed.killfeed_processor import (
    KillfeedProcessor,
)
from overtrack_cv.games.overwatch.processors.loading_map.loading_map_processor import (
    LoadingMapProcessor,
)
from overtrack_cv.games.overwatch.processors.menu.menu_processor import MenuProcessor
from overtrack_cv.games.overwatch.processors.objective.objective_processor import (
    ObjectiveProcessor,
)
from overtrack_cv.games.overwatch.processors.role_select.role_select_processor import (
    RoleSelectProcessor,
)
from overtrack_cv.games.overwatch.processors.score.score_processor import ScoreProcessor
from overtrack_cv.games.overwatch.processors.tab.tab_processor import TabProcessor
from overtrack_cv.games.processor import (
    ConditionalProcessor,
    EveryN,
    OrderedProcessor,
    Processor,
    ShortCircuitProcessor,
)


def create_pipeline(extra_processors: Sequence[Processor] = (), training=False) -> Processor:
    pipeline = OrderedProcessor(
        ShortCircuitProcessor(
            MenuProcessor(),
            RoleSelectProcessor(),
            LoadingMapProcessor(),
            ScoreProcessor(),
            EndgameProcessor(),
            ObjectiveProcessor(),
            EndgameSRProcessor(),
            HeroSelectProcessor(),
            order_defined=False,
        ),
        EveryN(HeroProcessor(), 3),
        ConditionalProcessor(
            OrderedProcessor(
                TabProcessor(),
                KillfeedProcessor(extrapolate_rowdetect=training, check_heroonly_killfeed=training),
                EliminationsProcessor(),
            ),
            condition=lambda f: ("objective2" in f and f.overwatch.objective.overwatch)
            or (f.overwatch.hero and f.overwatch.hero.hero)
            or training,
            lookbehind=7,
            lookbehind_behaviour=any,
            default_without_history=True,
        ),
    )
    if extra_processors:
        pipeline.processors = tuple(list(pipeline.processors) + list(extra_processors))
    return pipeline
