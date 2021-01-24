import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class Season:
    index: int
    start: float
    end: float
    has_ranked: bool = True
    season_name: Optional[str] = None

    @property
    def name(self) -> str:
        return self.season_name or f"Season {self.index}"


_PDT = datetime.timezone(datetime.timedelta(hours=-7))
_PST = datetime.timezone(datetime.timedelta(hours=-8))
_season_1_start = datetime.datetime.strptime(
    # https://twitter.com/PlayApex/status/1107733497450356742
    "Mar 19 2019 10:00AM",
    "%b %d %Y %I:%M%p",
).replace(tzinfo=_PDT)
_season_2_start = datetime.datetime.strptime(
    # https://twitter.com/PlayApex/status/1107733497450356742
    "Jul 2 2019 10:00AM",
    "%b %d %Y %I:%M%p",
).replace(tzinfo=_PDT)
_season_3_start = 1569956446
_season_4_start = datetime.datetime.strptime("Feb 4 2020 10:00AM", "%b %d %Y %I:%M%p").replace(tzinfo=_PST)
_season_5_start = datetime.datetime.strptime(
    # https://twitter.com/ApexLegendNews/status/1257370233628700678
    "May 12 2020 10:00AM",
    "%b %d %Y %I:%M%p",
).replace(tzinfo=_PDT)
_season_6_start = datetime.datetime.strptime("Aug 18 2020 10:00AM", "%b %d %Y %I:%M%p").replace(tzinfo=_PDT)
_season_7_start = datetime.datetime.strptime("Nov 10 2020 10:00AM", "%b %d %Y %I:%M%p").replace(tzinfo=_PDT)

seasons = [
    Season(0, 0, _season_1_start.timestamp(), has_ranked=False),
    Season(1, _season_1_start.timestamp(), _season_2_start.timestamp(), has_ranked=False),
    Season(2, _season_2_start.timestamp(), _season_3_start),
    Season(3, _season_3_start, _season_4_start.timestamp()),
    Season(4, _season_4_start.timestamp(), _season_5_start.timestamp()),
    Season(5, _season_5_start.timestamp(), _season_6_start.timestamp()),
    Season(6, _season_6_start.timestamp(), _season_7_start.timestamp()),
    Season(7, _season_7_start.timestamp(), float("inf")),
    Season(1002, 0, 0, season_name="Season 2 Solos", has_ranked=False),
    Season(
        1003,
        _season_3_start,
        _season_4_start.timestamp(),
        season_name="Season 3 Duos",
        has_ranked=False,
    ),
    Season(
        1004,
        _season_4_start.timestamp(),
        1589302800.0,
        season_name="Season 4 Duos",
        has_ranked=False,
    ),
    Season(1005, 1589302800.0, 1597766400.0, season_name="Season 5 Duos", has_ranked=False),
    Season(1006, 1597726800.0, 1604984400.0, season_name="Season 6 Duos", has_ranked=False),
    # Season(2000, _season_3_start, float('inf'), season_name='Scrims', has_ranked=False),
]
current_season = sorted([s for s in seasons if s.index < 100], key=lambda s: s.end)[-1]


def main():
    from pprint import pprint

    pprint(seasons)


if __name__ == "__main__":
    main()
