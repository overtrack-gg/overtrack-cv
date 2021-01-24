from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from overtrack_cv.core import arrayops

# Version 1 used a single int for intensity
# Version 2 uses a colour
SelectedWeaponTell = Union[int, Tuple[int, int, int]]


@dataclass(frozen=True)
class Weapons:
    weapon_names: List[str]
    selected_weapons: Tuple[SelectedWeaponTell, SelectedWeaponTell]
    clip: Optional[int]  # oscar mike, ladies
    ammo: Optional[int]

    @property
    def selected_weapon_index(self) -> Optional[int]:
        if np.max(self.selected_weapons) > 190 and np.min(self.selected_weapons) < 100:
            return arrayops.argmin(self.selected_weapons)
        else:
            return None
