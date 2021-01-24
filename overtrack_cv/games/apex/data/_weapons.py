import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Weapon:
    name: str
    type: str
    ammo_type: str

    full_name: Optional[str] = None

    def __post_init__(self):
        if not self.full_name:
            self.full_name = self.name

    # attachments


pistols = [
    Weapon("RE-45", "pistol", "light", full_name="RE-45 Auto"),
    Weapon("P2020", "pistol", "light"),
    Weapon("Wingman", "pistol", "heavy"),
]
shotguns = [
    Weapon("Mozambique", "shotgun", "shotgun"),
    Weapon("EVA-8 Auto", "shotgun", "shotgun"),
    Weapon("Peacekeeper", "shotgun", "special"),
    Weapon("Mastiff", "shotgun", "shotgun"),
]
ars = [
    Weapon("Hemlock", "ar", "heavy"),
    Weapon("Flatline", "ar", "heavy"),
    Weapon("Havoc", "ar", "energy"),
    Weapon("R-301", "ar", "light", full_name="R-301 Carbine"),
    Weapon("G7 Scout", "ar", "light"),
]
lmgs = [
    Weapon("Spitfire", "lmg", "heavy"),
    Weapon("Devotion", "lmg", "special"),
    Weapon("L-STAR", "lmg", "special"),
]
smgs = [
    Weapon("Alternator", "smg", "light"),
    Weapon("Prowler", "smg", "heavy"),
    Weapon("R-99", "smg", "light"),
    Weapon("Volt", "smg", "energy"),
]
snipers = [
    Weapon("Triple Take", "sniper", "sniper"),
    Weapon("Charge Rifle", "sniper", "sniper"),
    Weapon("Longbow", "sniper", "sniper"),
    Weapon("Sentinel", "sniper", "sniper"),
    Weapon("Kraber", "sniper", "special"),
]
weapons = pistols + shotguns + ars + lmgs + smgs + snipers
weapon_names = [w.name.upper() for w in weapons]
