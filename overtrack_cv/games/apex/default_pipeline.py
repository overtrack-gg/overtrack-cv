from overtrack_cv.games.apex.processors.combat.combat_processor import CombatProcessor
from overtrack_cv.games.apex.processors.coordinates.coordinate_processor import (
    CoordinateProcessor,
)
from overtrack_cv.games.apex.processors.map_loading.map_loading_processor import (
    MapLoadingProcessor,
)
from overtrack_cv.games.apex.processors.match_status.match_status_processor import (
    MatchStatusProcessor,
)
from overtrack_cv.games.apex.processors.match_summary.match_summary_processor import (
    MatchSummaryProcessor,
)
from overtrack_cv.games.apex.processors.menu.menu_processor import MenuProcessor
from overtrack_cv.games.apex.processors.squad.squad_processor import SquadProcessor
from overtrack_cv.games.apex.processors.squad_summary.squad_summary_processor import (
    SquadSummaryProcessor,
)
from overtrack_cv.games.apex.processors.weapon.weapon_processor import WeaponProcessor
from overtrack_cv.games.apex.processors.your_squad.your_squad_processor import (
    YourSquadProcessor,
)
from overtrack_cv.games.processor import (
    ConditionalProcessor,
    EveryN,
    OrderedProcessor,
    Processor,
    ShortCircuitProcessor,
)


def create_pipeline(interleave_processors: bool = True) -> Processor:
    pipeline = OrderedProcessor(
        ShortCircuitProcessor(
            MenuProcessor(),
            MapLoadingProcessor(),
            YourSquadProcessor(),
            MatchSummaryProcessor(),
            SquadSummaryProcessor(),
            OrderedProcessor(
                EveryN(MatchStatusProcessor(), 4 if interleave_processors else 1),
                CoordinateProcessor(),
                condition=all,
            ),
            order_defined=False,
        ),
        ConditionalProcessor(
            OrderedProcessor(
                EveryN(SquadProcessor(), 4 if interleave_processors else 1, offset=0),
                CombatProcessor(),
                EveryN(
                    WeaponProcessor(),
                    4 if interleave_processors else 1,
                    offset=1,
                    override_condition=lambda f: f.apex.combat_log,
                ),
            ),
            condition=lambda f: f.apex.coordinates or f.apex.match_status or f.apex.minimap,
            lookbehind=15,
            lookbehind_behaviour=any,
            default_without_history=True,
        ),
    )
    return pipeline


# def create_lightweight_pipeline() -> Processor:
# 	pipeline = OrderedProcessor(
# 		ShortCircuitProcessor(
# 			EveryN(MenuProcessor(), 2),
# 			YourSquadProcessor(),
# 			EveryN(MatchSummaryProcessor(), 5),
# 			SquadSummaryProcessor(),
# 			EveryN(
# 				OrderedProcessor(
# 					EveryN(MatchStatusProcessor(), 2),
# 					MinimapProcessor(),
# 				),
# 				2,
# 			),
# 			order_defined=False,
# 		),
# 		ConditionalProcessor(
# 			OrderedProcessor(
# 				EveryN(SquadProcessor(), 7),
# 				EveryN(WeaponProcessor(), 3),
# 				CombatProcessor(),
# 			),
# 			condition=lambda f: ("location" in f) or ("match_status" in f),
# 			lookbehind=15,
# 			lookbehind_behaviour=any,
# 			default_without_history=True,
# 		),
# 	)
# 	return pipeline


def main() -> None:
    import glob

    from overtrack_cv.util.test_processor import test_processor

    test_processor(
        # [p for p in glob.glob("D:/overtrack/frames7/*.png") if "debug" not in p][40:],
        create_pipeline(interleave_processors=True),
        "frame",
        "minimap",
        "location",
        "match_status",
        "minimap",
        "apex_play_menu",
        "your_squad",
        "champion_squad",
        "match_summary",
        "squad_summary",
        "combat",
        "squad",
        game="apex",
        images=glob.glob("S:/Downloads/apexframesdump/*.png")
        # images=glob.glob('./processors/**/samples/*.png')
    )


if __name__ == "__main__":
    main()
