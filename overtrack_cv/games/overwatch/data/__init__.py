from typing import TYPE_CHECKING

from ._heroes import *
from ._maps import *
from ._modes import *
from ._seasons import *

if TYPE_CHECKING:
    from typing_extensions import Literal

    Rank = Literal["bronze", "silver", "gold", "platinum", "diamond", "master", "grandmaster"]
else:
    Rank = str


def sr_to_rank(sr: int) -> Rank:
    if sr < 0:
        raise ValueError(f"Invalid SR: cannot be negative")
    elif sr < 1499:
        return "bronze"
    elif sr < 1999:
        return "silver"
    elif sr < 2499:
        return "gold"
    elif sr < 2999:
        return "platinum"
    elif sr < 3499:
        return "master"
    elif sr <= 5000:
        return "grandmaster"
    else:
        raise ValueError(f"Invalid SR: cannot be over 5000")
