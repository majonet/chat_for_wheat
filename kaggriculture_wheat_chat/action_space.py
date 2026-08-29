from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


MOVE_ACTIONS = ("NORTH", "SOUTH", "EAST", "WEST")
UNIT_ACTIONS = list(MOVE_ACTIONS) + [
    "PLANT",
    "WATER",
    "HARVEST",
    "PICKUP",
    "DROP",
    "PASS",
]


@dataclass(frozen=True)
class MarketAction:
    op: str
    qty: int = 0


# A single market order per RL decision. The simulator supports up to 10 orders,
# but one parameterized order can express all wheat-only economic actions needed here.
MARKET_ACTIONS = [MarketAction("NONE", 0)]
for q in (1, 2, 4, 8, 16, 32, 64, 100):
    MARKET_ACTIONS.append(MarketAction("BUY_SEED", q))
for q in (1, 2, 4, 8, 16, 32, 64, 100):
    MARKET_ACTIONS.append(MarketAction("SELL", q))
MARKET_ACTIONS.extend([MarketAction("HIRE"), MarketAction("BUY_LAND")])

UNIT_TO_ID = {a: i for i, a in enumerate(UNIT_ACTIONS)}
MARKET_TO_ID = {(a.op, a.qty): i for i, a in enumerate(MARKET_ACTIONS)}


def decode_market(index: int) -> MarketAction:
    if not 0 <= index < len(MARKET_ACTIONS):
        raise IndexError(index)
    return MARKET_ACTIONS[index]


def build_action(unit_action_ids: Sequence[int], market_index: int, n_hands: int) -> dict:
    """Convert policy indices into Kaggriculture's exact action schema."""
    if not unit_action_ids:
        unit_action_ids = [UNIT_TO_ID["PASS"]]

    farmer_op = UNIT_ACTIONS[int(unit_action_ids[0])]
    farmer = ["PLANT", "WHEAT"] if farmer_op == "PLANT" else (["PICKUP", "WHEAT", 1] if farmer_op == "PICKUP" else [farmer_op])

    hands: List[list] = []
    for idx in range(n_hands):
        op = UNIT_ACTIONS[int(unit_action_ids[idx + 1])] if idx + 1 < len(unit_action_ids) else "PASS"
        hands.append(["PLANT", "WHEAT"] if op == "PLANT" else (["PICKUP", "WHEAT", 1] if op == "PICKUP" else [op]))

    m = decode_market(int(market_index))
    if m.op == "NONE":
        market: List[list] = []
    elif m.op == "BUY_SEED":
        market = [["BUY_SEED", "WHEAT", int(m.qty)]]
    elif m.op == "SELL":
        market = [["SELL", "WHEAT", int(m.qty)]]
    elif m.op in {"HIRE", "BUY_LAND"}:
        market = [[m.op]]
    else:
        raise ValueError(f"Unsupported market action: {m}")

    return {"farmer": farmer, "hands": hands, "market": market}
