ranks = ["bronze", "silver", "gold", "platinum", "diamond", "master", "apex_predator"]

rank_tiers = ["IV", "III", "II", "I"]

rank_rp = {
    "bronze": (0, 1200),
    "silver": (1200, 2800),
    "gold": (2800, 4800),
    "platinum": (4800, 7200),
    "diamond": (7200, 10_000),
    "master": (10_000, 99_999),
    "apex_predator": (10_000, 99_999),
}

rank_entry_cost = {
    "bronze": 0,
    "silver": 12,
    "gold": 24,
    "platinum": 36,
    "diamond": 48,
    "master": 60,
    "apex_predator": 60,
}

rank_rewards = {
    10: 10,
    9: 10,
    8: 20,
    7: 20,
    6: 30,
    5: 30,
    4: 40,
    3: 40,
    2: 60,
    1: 100,
}
for placement in range(11, 21):
    rank_rewards[placement] = 0
