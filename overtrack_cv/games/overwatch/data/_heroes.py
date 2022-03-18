from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from overtrack_cv.util.compat import Literal

StatType = Literal["maximum", "average", "best", "duration"]
Role = Literal["tank", "damage", "support"]


@dataclass
class Stat:
    name: str
    v1_name: Optional[str] = None
    stat_type: StatType = "maximum"
    is_percent: bool = False

    def __post_init__(self):
        if not self.v1_name:
            self.v1_name = self.name


@dataclass
class Hero:
    name: str
    key: str
    role: Role
    ult: Optional[str]
    can_heal: bool

    stats: Tuple[List[Stat], List[Stat]]

    ability_1: Optional[str] = None
    ability_2: Optional[str] = None


generic_stats = [
    Stat("eliminations", "elims"),
    Stat("objective kills", "objective_kills"),
    Stat(
        "objective time",
        "objective_time",
        stat_type="duration",
    ),
    Stat(
        "hero damage done",
        "damage",
    ),
    Stat("healing done", "healing"),
    Stat("deaths", "deaths"),
]

tanks: Dict[str, Hero] = {
    "dva": Hero(
        name="D.Va",
        key="dva",
        role="tank",
        ability_1="boosters",
        ability_2="micro_missiles",
        ult="self_destruct",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("damage blocked"),
            ],
            [Stat("self-destruct kills"), Stat("mechs called")],
        ),
    ),
    "orisa": Hero(
        name="Orisa",
        key="orisa",
        role="tank",
        ult="supercharger",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("damage blocked"),
            ],
            [Stat("offensive assists"), Stat("damage amplified")],
        ),
    ),
    "reinhardt": Hero(
        name="Reinhardt",
        key="reinhardt",
        role="tank",
        ability_1="charge",
        ability_2="firestrike",
        ult="earthshatter",
        can_heal=False,
        stats=(
            [Stat("damage blocked"), Stat("kill streak - best", stat_type="best"), Stat("charge kills")],
            [Stat("fire strike kills"), Stat("earthshatter kills")],
        ),
    ),
    "roadhog": Hero(
        name="Roadhog",
        key="roadhog",
        role="tank",
        ability_1="chain_hook",
        ult="whole_hog",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("enemies hooked"),
            ],
            [
                Stat("hook accuracy", stat_type="average", is_percent=True),
                Stat("self healing"),
                Stat("whole hog kills"),
            ],
        ),
    ),
    "sigma": Hero(
        name="Sigma",
        key="sigma",
        role="tank",
        ability_2="accretion",
        ult="gravitic_flux",
        can_heal=False,
        stats=(
            [
                Stat(
                    "damage blocked",
                ),
                Stat("kill streak - best", stat_type="best"),
                Stat("damage absorbed"),
            ],
            [
                Stat(
                    "accretion kills",
                ),
                Stat("gravitic flux kills"),
            ],
        ),
    ),
    "winston": Hero(
        name="Winston",
        key="winston",
        role="tank",
        ability_1="jump_pack",
        ult="primal_rage",
        can_heal=False,
        stats=(
            [Stat("damage blocked"), Stat("kill streak - best", stat_type="best"), Stat("melee kills")],
            [
                Stat("players knocked back"),
                Stat("primal rage kills"),
            ],
        ),
    ),
    "hammond": Hero(
        name="Wrecking Ball",
        key="hammond",
        role="tank",
        ult="minefield",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("final blows"),
            ],
            [
                Stat(
                    "grappling claw kills",
                ),
                Stat("piledriver kills"),
                Stat("minefield kills"),
            ],
        ),
    ),
    "zarya": Hero(
        name="Zarya",
        key="zarya",
        role="tank",
        ult="graviton_surge",
        can_heal=False,
        stats=(
            [Stat("damage blocked"), Stat("kill streak - best", stat_type="best"), Stat("high energy kills")],
            [Stat("average energy"), Stat("graviton surge kills")],
        ),
    ),
}
supports: Dict[str, Hero] = {
    "ana": Hero(
        name="Ana",
        key="ana",
        role="support",
        ult="nano_boost",
        can_heal=True,
        stats=(
            [
                Stat("unscoped accuracy", stat_type="average", is_percent=True),
                Stat("scoped accuracy", stat_type="average", is_percent=True),
                Stat("defensive assists"),
            ],
            [Stat("nano boost assists"), Stat("enemies slept")],
        ),
        ability_1="sleep_dart",
        ability_2="biotic_grenade",
    ),
    "baptiste": Hero(
        name="Baptiste",
        key="baptiste",
        role="support",
        ult="amplification_matrix",
        can_heal=True,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("healing accuracy", stat_type="average", is_percent=True),
                Stat("damage amplified"),
            ],
            [
                Stat("amplification matrix assists"),
                Stat("defensive assists"),
                Stat("immportality field deaths prevented"),
            ],
        ),
    ),
    "brigitte": Hero(
        name="Brigitte",
        key="brigitte",
        role="support",
        ult="rally",
        can_heal=True,
        stats=(
            [Stat("offensive assists"), Stat("defensive assists"), Stat("damage blocked")],
            [Stat("armor provided"), Stat("inspire uptime percentage", stat_type="average", is_percent=True)],
        ),
        ability_1="whip_shot",
    ),
    "lucio": Hero(
        name="Lucio",
        key="lucio",
        role="support",
        ult="sound_barrier",
        can_heal=True,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("sound barriers provided"),
            ],
            [Stat("offensive assists"), Stat("defensive assists")],
        ),
    ),
    "mercy": Hero(
        name="Mercy",
        key="mercy",
        role="support",
        ult="valkyrie",
        can_heal=True,
        stats=(
            [Stat("offensive assists"), Stat("defensive assists"), Stat("players resurrected")],
            [Stat("blaster kills"), Stat("damage amplified")],
        ),
    ),
    "moira": Hero(
        name="Moira",
        key="moira",
        role="support",
        ability_2="biotic_orb",
        ult="coalescence",
        can_heal=True,
        stats=(
            [
                Stat("secondary fire accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("defensive assists"),
            ],
            [Stat("coalescence kills"), Stat("coalescence healing"), Stat("self healing")],
        ),
    ),
    "zenyatta": Hero(
        name="Zenyatta",
        key="zenyatta",
        role="support",
        ult="transcendence",
        can_heal=True,
        stats=(
            [
                Stat("secondary fire accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("offensive assists"),
            ],
            [Stat("defensive assists"), Stat("transcendence healing")],
        ),
    ),
}
dps: Dict[str, Hero] = {
    "ashe": Hero(
        name="Ashe",
        key="ashe",
        role="damage",
        ability_1="coach_gun",
        ability_2="dynamite",
        ult="bob",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat(
                    "final blows",
                ),
                Stat("scoped accuracy", stat_type="average", is_percent=True),
            ],
            [Stat("scoped critical hits"), Stat("dynamite kills"), Stat("bob kills")],
        ),
    ),
    "bastion": Hero(
        name="Bastion",
        key="bastion",
        role="damage",
        ult="configuration_tank",
        can_heal=True,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("recon kills"),
            ],
            [Stat("sentry kills"), Stat("tank kills"), Stat("self healing")],
        ),
    ),
    "doomfist": Hero(
        name="Doomfist",
        key="doomfist",
        role="damage",
        ability_1="rising_uppercut",
        ability_2="seismic_slam",
        ult="meteor_strike",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("final blows"),
            ],
            [Stat("ability damage done"), Stat("meteor strike kills"), Stat("shields created")],
        ),
    ),
    "echo": Hero(
        name="Echo",
        key="echo",
        role="damage",
        ability_1="flight",
        ability_2="focusing_beam",
        ult="duplicate",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("final blows"),
            ],
            [Stat("sticky bomb kills"), Stat("focusing beam kills"), Stat("duplicate kills")],
        ),
    ),
    "genji": Hero(
        name="Genji",
        key="genji",
        role="damage",
        ability_1="swift_strike",
        ability_2="deflect",
        ult="dragonblade",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("final blows"),
            ],
            [Stat("damage reflected"), Stat("dragonblade kills")],
        ),
    ),
    "hanzo": Hero(
        name="Hanzo",
        key="hanzo",
        role="damage",
        ability_1="sonic_arrow",
        ability_2="storm_arrow",
        ult="dragonstrike",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("final blows"),
            ],
            [Stat("critical hits"), Stat("recon assists"), Stat("dragonstrike kills")],
        ),
    ),
    "junkrat": Hero(
        name="Junkrat",
        key="junkrat",
        role="damage",
        ability_1="concussion_mine",
        ability_2="bear_trap",
        ult="riptire",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("final blows"),
            ],
            [Stat("enemies trapped"), Stat("rip-tire kills")],
        ),
    ),
    "mccree": Hero(
        name="Mccree",
        key="mccree",
        role="damage",
        ability_2="flashbang",
        ult="deadeye",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("final blows"),
            ],
            [Stat("critical hits"), Stat("deadeye kills"), Stat("fan the hammer kills")],
        ),
    ),
    "mei": Hero(
        name="Mei",
        key="mei",
        role="damage",
        ult="blizzard",
        can_heal=True,
        stats=(
            [Stat("damage blocked"), Stat("kill streak - best", stat_type="best"), Stat("final blows")],
            [Stat("enemies frozen"), Stat("blizzard kills"), Stat("self healing")],
        ),
    ),
    "pharah": Hero(
        name="Pharah",
        key="pharah",
        role="damage",
        ult="rocket_barrage",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("final blows"),
            ],
            [Stat("barrage kills"), Stat("rocket direct hits")],
        ),
    ),
    "reaper": Hero(
        name="Reaper",
        key="reaper",
        role="damage",
        ult="death_blossom",
        can_heal=True,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("final blows"),
            ],
            [Stat("death blossom kills"), Stat("self healing")],
        ),
    ),
    "soldier": Hero(
        name="Soldier 76",
        key="soldier",
        role="damage",
        ult="tactical_visor",
        can_heal=True,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("final blows"),
            ],
            [Stat("helix rocket kills"), Stat("tactical visor kills")],
        ),
    ),
    "sombra": Hero(
        name="Sombra",
        key="sombra",
        role="damage",
        ult="emp",
        can_heal=True,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("offensive assists"),
            ],
            [Stat("enemies hacked"), Stat("enemies emp'd")],
        ),
    ),
    "symmetra": Hero(
        name="Symmetra",
        key="symmetra",
        role="damage",
        ult="photon_barrier",
        can_heal=False,
        stats=(
            [Stat("sentry turret kills"), Stat("kill streak - best", stat_type="best"), Stat("damage blocked")],
            [
                Stat("players teleported"),
                Stat("primary fire accuracy", stat_type="average", is_percent=True),
                Stat("secondary fire accuracy", stat_type="average", is_percent=True),
            ],
        ),
    ),
    "torbjorn": Hero(
        name="Torbjörn",
        key="torbjorn",
        role="damage",
        ult="molten_core",
        can_heal=False,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("torbjorn kills"),
            ],
            [Stat("turret kills"), Stat("molten core kills"), Stat("turret damage")],
        ),
    ),
    "tracer": Hero(
        name="Tracer",
        key="tracer",
        role="damage",
        ult="pulse_bomb",
        can_heal=True,
        stats=(
            [
                Stat("weapon accuracy", stat_type="average", is_percent=True),
                Stat("kill streak - best", stat_type="best"),
                Stat("final blows"),
            ],
            [Stat("pulse bomb kills"), Stat("pulse bombs attached")],
        ),
    ),
    "widowmaker": Hero(
        name="Widowmaker",
        key="widowmaker",
        role="damage",
        ult="infra_sight",
        can_heal=False,
        stats=(
            [Stat("recon assists"), Stat("kill streak - best", stat_type="best"), Stat("final blows")],
            [Stat("scoped accuracy", stat_type="average", is_percent=True), Stat("scoped critical hits")],
        ),
    ),
}

heroes: Dict[str, Hero] = dict()
heroes.update(tanks)
heroes.update(dps)
heroes.update(supports)


def main() -> None:
    # print(list(heroes.keys()))
    import overtrack.util.textops

    print({h: overtrack.util.textops.strip_string(heroes[h].name.upper()) for h in heroes})
    # for h in heroes:
    #     print(h.title())


if __name__ == "__main__":
    main()
