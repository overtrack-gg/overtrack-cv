import datetime
from dataclasses import dataclass

_UTC = datetime.timezone(datetime.timedelta(hours=0))
_NZT = datetime.timezone(datetime.timedelta(hours=+12))


def _parse_utc(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(s, "%b %d %Y %I:%M%p").replace(tzinfo=_UTC)


def _parse_nzt(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(s, "%b %d %Y %I:%M%p").replace(tzinfo=_NZT)


@dataclass
class GameVersion:
    name: str
    published: datetime.datetime


game_versions = [
    GameVersion("00.00.0-beta", _parse_utc("Jan 1 2020 1:00AM")),
    GameVersion("01.00.0", _parse_utc("Jun 2 2020 1:00AM")),
    GameVersion("01.01.0", _parse_nzt("Jun 10 2020 4:11AM")),
    GameVersion("01.02.0", _parse_nzt("Jun 24 2020 1:00AM")),
    GameVersion("01.03.0", _parse_nzt("Jul 10 2020 10:00AM")),
    GameVersion("01.05.0", _parse_nzt("Aug 5 2020 10:00AM")),
    GameVersion("02.00.0", _parse_nzt("Jan 13 2021 01:01AM")),
]


def get_version(t: datetime.datetime) -> GameVersion:
    for v in reversed(game_versions):
        if t > v.published:
            return v
    else:
        return game_versions[0]


def main():
    print(game_versions[-1].published)
    print(datetime.datetime.now())
    print(datetime.datetime.now().replace(tzinfo=_UTC) - game_versions[-1].published)


if __name__ == "__main__":
    main()
