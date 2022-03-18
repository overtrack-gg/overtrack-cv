import datetime
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

_UTC = datetime.timezone(datetime.timedelta(hours=0))


def _parse_utc(s: str) -> int:
    return int(datetime.datetime.strptime(s, "%b %d %Y %I:%M%p").replace(tzinfo=_UTC).timestamp())


@dataclass
class Season:
    name: str
    start: float
    end: float

    index: Optional[int] = None
    display: bool = True
    off_season: bool = False
    is_222: bool = False

    filterable_by_index: bool = False

    def __contains__(self, timestamp: float) -> bool:
        return self.start <= timestamp < self.end


def get_hour(day: datetime.date) -> datetime.time:
    if day.year >= 2021 and day.month > 1:
        return datetime.time(hour=12 + 7)
    else:
        return datetime.time(hour=12 + 6)


def get_season_start_dates(start: datetime.date, end: datetime.date) -> Iterator[datetime.datetime]:
    current = start

    while current <= end:
        if current.month % 2 != 0 and current.day < 8 and current.weekday() == 3:
            yield datetime.datetime.combine(current, get_hour(current))
        current += datetime.timedelta(days=1)


def get_season_pairs(
    start: datetime.date, end: datetime.date
) -> Iterator[Tuple[datetime.datetime, datetime.datetime]]:
    start_dates = get_season_start_dates(start, end)
    previous = next(start_dates)

    for current in start_dates:
        yield previous, current
        previous = current


seasons = [
    Season(name="Pre Season 4", start=0, end=1488193200, index=1, display=False, off_season=True),
    Season(name="Season 4", start=1488193200, end=1496059200, index=4),
    Season(name="Season 4-5 Off Season", start=1496059200, end=1496275199, index=104, off_season=True),
    Season(name="Season 5", start=1496275199, end=1503964799, index=5),
    Season(name="Season 5-6 Off Season", start=1503964799, end=1504224000, index=105, off_season=True),
    Season(name="Season 6", start=1504224000, end=1509237000, index=6),
    Season(name="Season 6-7 Off Season", start=1509237000, end=1509494400, index=106, off_season=True),
    Season(name="Season 7", start=1509494400, end=1514458800, index=7),
    Season(name="Season 7-8 Off Season", start=1514458800, end=1514764800, index=107, off_season=True),
    Season(name="Season 8", start=1514764800, end=1519556400, index=8),
    Season(name="Season 8-9 Off Season", start=1519556400, end=1519862400, index=108, off_season=True),
    Season(name="Season 9", start=1519862400, end=1524875400, index=9),
    Season(name="Season 9-10 Off Season", start=1524875400, end=1525132800, index=109, off_season=True),
    Season(name="Season 10", start=1525132800, end=1530144600, index=10),
    Season(name="Season 10-11 Off Season", start=1530144600, end=1530403170, index=110, off_season=True),
    Season(name="Season 11", start=1530403170, end=1535501400, index=11),
    Season(name="Season 11-12 Off Season", start=1535501400, end=1535759970, index=111, off_season=True),
    Season(name="Season 12", start=1535759970, end=1540768200, index=12),
    Season(name="Season 12-13 Off Season", start=1540768200, end=1541026770, index=112, off_season=True),
    Season(name="Season 13", start=1541026770, end=1546293600, index=13),
    Season(name="Season 13-14 Off Season", start=1546293600, end=1546300800, index=113, off_season=True),
    Season(name="Season 14", start=1546300800, end=1551398400, index=14),
    Season(name="Season 15", start=1551398400, end=1556668800, index=15),
    Season(name="Season 16", start=1556668800, end=1561939200, index=16),
    Season(name="Season 17", start=1561939200, end=1565712000, index=17),
    Season(name="Role Queue Beta", start=1565712000, end=1567554000, index=117, is_222=True),
    Season(
        name="Season 18",
        start=1567554000 - 24 * 60 * 60,
        end=_parse_utc("Nov 7 2019 6:00PM"),
        index=18,
        is_222=True,
    ),
]
for n, (start, end) in enumerate(
    get_season_pairs(datetime.date(2019, 11, 7), datetime.date.today() + datetime.timedelta(days=180))
):
    if start.date() > datetime.date.today():
        break
    season_index = 19 + n
    seasons.append(
        Season(
            name=f"Season {season_index}",
            start=int(start.replace(tzinfo=_UTC).timestamp()),
            end=int(end.replace(tzinfo=_UTC).timestamp()),
            index=season_index,
            is_222=True,
        )
    )

extra_seasons = [
    Season(name="Role Queue Beta (PTR)", start=1563600800, end=1567209600, index=1001, is_222=True),
    Season(name="Season 18 False Start", start=1567296000, end=1567304000, index=1018, is_222=True),
]

seasons_by_index: Dict[int, Season] = {s.index: s for s in seasons + extra_seasons}


def main() -> None:
    from overtrack_cv.util.prettyprint import pprint

    for season in seasons:
        print(
            f"{datetime.datetime.fromtimestamp(season.start).replace(tzinfo=_UTC)} -> {datetime.datetime.fromtimestamp(season.end).replace(tzinfo=_UTC)}"
        )
        pprint(season)
    # return

    from typing import Iterable

    from overtrack_models.orm.overwatch_game_summary import OverwatchGameSummary
    from overtrack_models.orm.user import User
    from tqdm import tqdm

    uid = set()

    bar: Iterable[OverwatchGameSummary] = tqdm(
        OverwatchGameSummary.season_time_index.query(
            seasons[-2].index,
            # _parse_utc("Mar 4 2021 5:00PM")
            OverwatchGameSummary.time > seasons[-1].start,
        ),
        desc="Loading games to edit",
    )
    bar: Iterable[OverwatchGameSummary] = tqdm(list(bar), desc="Editing")

    for g in bar:
        print(g)
        # bar.set_description(str(g))

        g.season = seasons[-1].index
        g.save()
        if g.user_id not in uid:
            uid.add(g.user_id)
            u: User = User.user_id_index.get(g.user_id)
            if u.overwatch_last_season != g.season:
                u.overwatch_seasons.add(g.season)
                u.overwatch_last_season = g.season
                u.save()


if __name__ == "__main__":
    main()
