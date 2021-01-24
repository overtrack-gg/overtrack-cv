import dataclasses
import datetime
import functools
import logging
import typing
from functools import wraps
from typing import Tuple

logger = logging.getLogger(__name__)


def round_floats(_cls=None, *, precision: int = 2):
    import numpy as np

    def wrap(cls):
        orig__post_init__ = getattr(cls, "__post_init__", None)

        def __post_init__(self, *initvars):
            if orig__post_init__:
                orig__post_init__(self, *initvars)
            for field in dataclasses.fields(cls):
                is_float = field.type is float or (
                    getattr(field.type, "__origin__", None) == typing.Union
                    and float in field.type.__args__
                    and isinstance(getattr(self, field.name), float)
                )
                if is_float:
                    object.__setattr__(
                        self,
                        field.name,
                        round(float(getattr(self, field.name)), precision),
                    )
                elif getattr(field.type, "__origin__", None) is typing.Tuple:
                    object.__setattr__(
                        self,
                        field.name,
                        tuple(
                            round(float(e), precision)
                            if isinstance(e, (float, np.float, np.float16, np.float32, np.float64))
                            else e
                            for e in getattr(self, field.name)
                        ),
                    )
                elif getattr(field.type, "__origin__", None) is typing.List:
                    object.__setattr__(
                        self,
                        field.name,
                        [
                            round(float(e), precision)
                            if isinstance(e, (float, np.float, np.float16, np.float32, np.float64))
                            else e
                            for e in getattr(self, field.name)
                        ],
                    )

        setattr(cls, "__post_init__", __post_init__)
        return cls

    if _cls is None:
        return wrap
    return wrap(_cls)


def cached_property(f):
    """A memoize decorator for class properties."""

    @functools.wraps(f)
    def get(self):
        try:
            return self._cache[f]
        except AttributeError:
            self._cache = {}
        except KeyError:
            pass
        ret = self._cache[f] = f(self)
        return ret

    return property(get)


def validate_fields(a):
    __init__ = a.__init__

    @wraps(__init__)
    def _check_init(self, *args, **kwargs):
        __init__(self, *args, **kwargs)
        for f in dataclasses.fields(self):
            if not hasattr(self, f.name):
                raise AttributeError(
                    f"Construction of dataclass '{self.__class__.__qualname__}' incomplete: field '{f.name}' not defined"
                )

    a.__init__ = _check_init
    return a


def humansize(nbytes: float, suffixes: Tuple[str, ...] = ("B", "KB", "MB", "GB", "TB", "PB")) -> str:
    # http://stackoverflow.com/a/14996816
    if nbytes == 0:
        return "0 B"
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.0
        i += 1
    f = ("%.2f" % nbytes).rstrip("0").rstrip(".")
    return "%s %s" % (f, suffixes[i])


def s2ts(s: float, ms: bool = False, zpad: bool = True, sign: bool = False) -> str:
    prepend = ""
    if s < 0:
        prepend = "-"
        s = -s
    elif sign:
        prepend = "+"

    m = s / 60
    h = m / 60
    if zpad or int(h):
        ts = "%s%02d:%02d:%02d" % (prepend, h, m % 60, s % 60)
    else:
        ts = "%s%02d:%02d" % (prepend, m % 60, s % 60)

    if ms:
        return ts + f"{s % 1 :1.3f}"[1:]
    else:
        return ts


def ms2ts(ms: float) -> str:
    return s2ts(ms / 1000)


def ts2s(ts: str) -> float:
    if ts.count(":") == 3:
        hs, ms, ss = ts.split(":")
    else:
        hs = 0
        ms, ss = ts.split(":")
    h, m, s = int(hs), int(ms), float(ss)
    m = m + 60 * h
    s = m * 60 + s
    return s


def ts2ms(ts: str) -> int:
    return int(ts2s(ts) * 1000)


def dhms2timedelta(s: str) -> datetime.timedelta:
    td = datetime.timedelta()
    current = ""
    for c in s:
        if c.isdigit():
            current += c
        else:
            if c == "d":
                td += datetime.timedelta(days=int(current))
            elif c == "h":
                td += datetime.timedelta(hours=int(current))
            elif c == "m":
                td += datetime.timedelta(minutes=int(current))
            elif c == "s":
                td += datetime.timedelta(seconds=int(current))
            else:
                raise ValueError('Unknown timedelta specifier "%s"', c)
            current = ""
    return td


def bgr2html(color: Tuple[int, int, int]) -> str:
    return "#" + "".join(f"{c:02x}" for c in color[::-1])


def html2bgr(hex_str: str) -> Tuple[int, int, int]:
    if hex_str[0] == "#":
        hex_str = hex_str[1:]
    return int(hex_str[4:6], 16), int(hex_str[2:4], 16), int(hex_str[0:2], 16)
