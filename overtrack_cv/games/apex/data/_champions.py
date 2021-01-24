import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Champion:
    name: str


champions = {
    "bloodhound": Champion("Bloodhound"),
    "gibraltar": Champion("Gibraltar"),
    "lifeline": Champion("Lifeline"),
    "pathfinder": Champion("Pathfinder"),
    "wraith": Champion("Wraith"),
    "bangalore": Champion("Bangalore"),
    "mirage": Champion("Mirage"),
    "caustic": Champion("Caustic"),
    "octane": Champion("Octane"),
    "wattson": Champion("Wattson"),
    "crypto": Champion("Crypto"),
    "revenant": Champion("Revenant"),
    "loba": Champion("Loba"),
    "rampart": Champion("Rampart"),
}
