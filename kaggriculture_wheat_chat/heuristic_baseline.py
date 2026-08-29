from __future__ import annotations

from typing import Dict, List, Tuple


def _toward(x: int, y: int, tx: int, ty: int) -> str:
    if x < tx:
        return "EAST"
    if x > tx:
        return "WEST"
    if y < ty:
        return "SOUTH"
    if y > ty:
        return "NORTH"
    return "PASS"


def agent(obs: dict) -> dict:
    """Wheat-only baseline using the actual full-game API.

    Prioritizes: water > harvest > plant > move to shed/drop. Market sells only shed wheat.
    """
    p = int(obs["player"])
    farm = obs["farms"][p]
    private = obs["private"]
    day = int(obs.get("day", 0))
    money = float(farm["money"])
    seed_count = int(private.get("seeds", {}).get("WHEAT", 0))
    shed_wheat = int(private.get("shed", {}).get("WHEAT", 0))
    fx, fy = farm["farmer"]
    tile = farm["tiles"][fy][fx]

    orders: List[list] = []
    if seed_count < 8 and money >= 80:
        orders.append(["BUY_SEED", "WHEAT", 8])
    if shed_wheat > 0:
        orders.append(["SELL", "WHEAT", shed_wheat])

    if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
        age = day - int(tile.get("planted_day", day))
        if age >= 2 and int(tile.get("yield_units", 0)) > 0:
            return {"farmer": ["HARVEST"], "hands": [], "market": orders}
        if not tile.get("watered_today", False):
            return {"farmer": ["WATER"], "hands": [], "market": orders}

    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        # DIG is intentionally not used: it is outside the restricted agent economy.
        return {"farmer": ["PASS"], "hands": [], "market": orders}

    # Plant the first available unlocked tile.
    if seed_count > 0:
        for y, row in enumerate(farm["tiles"]):
            for x, t in enumerate(row):
                if t is None:
                    op = _toward(fx, fy, x, y)
                    if op == "PASS":
                        op = "PLANT"
                    action = ["PLANT", "WHEAT"] if op == "PLANT" else [op]
                    return {"farmer": action, "hands": [], "market": orders}

    # Get harvested inventory to the shed before selling.
    invs = private.get("inventories", [{}])
    carried = int(invs[0].get("WHEAT", 0)) if invs else 0
    h = len(farm["tiles"]) // 2
    shed = sorted([(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)])[0]
    if carried > 0:
        sx, sy = shed
        op = _toward(fx, fy, sx, sy)
        if op == "PASS":
            return {"farmer": ["DROP"], "hands": [], "market": orders}
        return {"farmer": [op], "hands": [], "market": orders}

    return {"farmer": ["PASS"], "hands": [], "market": orders}
