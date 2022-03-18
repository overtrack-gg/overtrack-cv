from dataclasses import dataclass
from typing import Optional


@dataclass
class LoginCacheAccount:
    name: str
    environment: str
    battle_tag: str
    account_id_hi: int
    account_id_lo: int

    def __str__(self) -> str:
        return (
            f"LoginCacheAccount("
            f"name={self.name}, "
            f"environment={self.environment}, "
            f"battle_tag={self.battle_tag.encode()}, "
            f"account_id_hi={self.account_id_hi}, "
            f"account_id_lo={self.account_id_lo}"
            f")"
        )

    __repr__ = __str__


@dataclass
class OverwatchClientMetadata:
    account_id: int
    account: Optional[LoginCacheAccount] = None
