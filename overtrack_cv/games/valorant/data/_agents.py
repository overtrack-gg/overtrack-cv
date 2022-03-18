from dataclasses import dataclass
from typing import Dict

from overtrack_cv.util.compat import Literal

Role = Literal[
    "Controller" "Duelist",
    "Initiator",
    "Sentinel",
]

# Order based on hero select order in practice range
AgentName = Literal[
    "Brimstone",
    "Cypher",
    "Jett",
    "Phoenix",
    "Raze",
    "Sage",
    "Sova",
    "Breach",
    "Omen",
    "Viper",
    "Reyna",
    "Killjoy",
    "Skye",
    "Yoru",
]


@dataclass
class Agent:
    name: AgentName
    role: Role


agents: Dict[AgentName, Agent] = {
    a.name: a
    for a in [
        Agent("Brimstone", "Controller"),
        Agent("Cypher", "Sentinel"),
        Agent("Jett", "Duelist"),
        Agent("Phoenix", "Duelist"),
        Agent("Raze", "Duelist"),
        Agent("Sage", "Sentinel"),
        Agent("Sova", "Initiator"),
        Agent("Breach", "Initiator"),
        Agent("Omen", "Controller"),
        Agent("Viper", "Controller"),
        Agent("Reyna", "Duelist"),
        Agent("Killjoy", "Sentinel"),
        Agent("Skye", "Initiator"),
        Agent("Yoru", "Duelist"),
    ]
}
