from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from action_space import MARKET_ACTIONS, UNIT_ACTIONS, build_action
from state_encoder import StateEncoder

try:
    from kaggle_environments import make
except ImportError as exc:  # pragma: no cover
    make = None
    _IMPORT_ERROR = exc


class KaggricultureWrapper:
    """Thin wrapper around the real Kaggriculture simulator.

    The wrapper does not reimplement crop growth, market mechanics, land purchase,
    hiring or end-of-day transitions. All game mechanics remain in Kaggle's simulator.
    """

    def __init__(self, opponent: str = "starter", seed: int = 17, episode_steps: int = 720,
                 board_size: int = 10, max_hands: int = 8):
        if make is None:
            raise ImportError(
                "kaggle-environments is required. Install with: pip install -U kaggle-environments"
            ) from _IMPORT_ERROR
        self.opponent = opponent
        self.seed = seed
        self.episode_steps = episode_steps
        self.board_size = board_size
        self.max_hands = max_hands
        self.env = None
        self.trainer = None
        self.encoder = StateEncoder(board_size, max_hands)
        self.last_obs: Optional[dict] = None
        self.stats: Dict[str, float] = {}
        self._make_env()

    def _make_env(self) -> None:
        # Do not depend on unsupported configuration keys. The environment uses its own
        # seed resolution mechanism; passing a seed is supported by current source builds.
        config = {"episodeSteps": self.episode_steps, "boardSize": self.board_size}
        try:
            config["seed"] = int(self.seed)
        except Exception:
            pass
        self.env = make("kaggriculture", configuration=config, debug=False)
        self.trainer = self.env.train([None, self.opponent])

    @property
    def obs_dim(self) -> int:
        return self.encoder.dim

    def reset(self) -> Tuple[dict, dict]:
        raw = self.trainer.reset()
        obs = self._to_dict(raw)
        self.last_obs = obs
        self.stats = {
            "revenue": 0.0,
            "seed_cost": 0.0,
            "hire_cost": 0.0,
            "land_cost": 0.0,
            "wheat_planted": 0.0,
            "wheat_harvested": 0.0,
            "wheat_sold": 0.0,
            "sell_revenue": 0.0,
            "wheat_seed_bought": 0.0,
            "workers_hired": 0.0,
            "land_purchases": 0.0,
            "weeds": 0.0,
            "invalid_actions": 0.0,
            "movement_actions": 0.0,
        }
        return self.encoder.encode(obs, 0), obs

    @staticmethod
    def _to_dict(obj: Any) -> dict:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        try:
            return dict(obj)
        except Exception:
            # Kaggle observation objects generally support mapping-style access.
            keys = getattr(obj, "__dict__", None)
            if keys is not None:
                return dict(keys)
            raise TypeError(f"Unsupported observation type: {type(obj)!r}")

    @staticmethod
    def _fib(n: int) -> int:
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    @classmethod
    def _hire_cost(cls, n_already_today: int, mult: int = 1) -> int:
        return mult * cls._fib(n_already_today)

    @staticmethod
    def _land_cost(unlocked_count: int) -> int:
        return {1: 1000, 2: 2000, 3: 4000}.get(unlocked_count, 10**9)

    def _shed_adjacent(self, pos: Tuple[int, int]) -> bool:
        h = self.board_size // 2
        return pos in {(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)}

    def valid_masks(self, obs: dict) -> Tuple[List[np.ndarray], np.ndarray]:
        player = int(obs["player"])
        farm = obs["farms"][player]
        private = obs.get("private", {})
        day = int(obs.get("day", 0))
        money = float(farm.get("money", 0.0))
        seed_count = int(private.get("seeds", {}).get("WHEAT", 0))
        shed_wheat = int(private.get("shed", {}).get("WHEAT", 0))

        def unit_mask(pos: Tuple[int, int]) -> np.ndarray:
            x, y = pos
            m = np.zeros(len(UNIT_ACTIONS), dtype=bool)
            # Boundary-safe movement. Locked cells remain passable in the simulator.
            m[UNIT_ACTIONS.index("NORTH")] = y > 0
            m[UNIT_ACTIONS.index("SOUTH")] = y < self.board_size - 1
            m[UNIT_ACTIONS.index("WEST")] = x > 0
            m[UNIT_ACTIONS.index("EAST")] = x < self.board_size - 1
            m[UNIT_ACTIONS.index("PASS")] = True

            tile = farm["tiles"][y][x]
            m[UNIT_ACTIONS.index("PLANT")] = tile is None and seed_count > 0
            is_wheat = isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT"
            m[UNIT_ACTIONS.index("WATER")] = is_wheat and not bool(tile.get("watered_today", False))
            age = day - int(tile.get("planted_day", day)) if is_wheat else -1
            m[UNIT_ACTIONS.index("HARVEST")] = is_wheat and age >= 2 and int(tile.get("yield_units", 0)) > 0

            adjacent = self._shed_adjacent((x, y))
            inv = private.get("inventories", [])
            unit_idx = 0 if pos == tuple(farm["farmer"]) else None
            # For PICKUP/DROP we only care whether the unit carries any wheat.
            # This policy does not buy or use other item types.
            carried = False
            if adjacent:
                # Shared mask is used for all units; precise inventory check is handled below by slot-specific action mask.
                carried = True
            m[UNIT_ACTIONS.index("PICKUP")] = adjacent and shed_wheat > 0
            m[UNIT_ACTIONS.index("DROP")] = adjacent
            return m

        masks = [unit_mask(tuple(farm["farmer"]))]
        for hand in farm.get("hands", [])[: self.max_hands]:
            masks.append(unit_mask(tuple(hand)))

        # Refine DROP by the actual corresponding unit inventory.
        inventories = private.get("inventories", [])
        for idx, pos in enumerate(self._positions(farm)):
            if idx < len(masks) and idx < len(inventories):
                masks[idx][UNIT_ACTIONS.index("DROP")] = (
                    self._shed_adjacent(pos) and sum(int(v) for v in inventories[idx].values()) > 0
                )

        market = np.zeros(len(MARKET_ACTIONS), dtype=bool)
        market[0] = True  # NONE
        for i, a in enumerate(MARKET_ACTIONS[1:], start=1):
            if a.op == "BUY_SEED":
                market[i] = money >= a.qty * 10
            elif a.op == "SELL":
                market[i] = shed_wheat >= a.qty
            elif a.op == "HIRE":
                market[i] = money >= self._hire_cost(int(farm.get("hires_today", 0)))
            elif a.op == "BUY_LAND":
                market[i] = len(farm.get("unlocked_quadrants", [])) < 4 and money >= self._land_cost(len(farm.get("unlocked_quadrants", [])))
        return masks, market

    @staticmethod
    def _positions(farm: dict) -> List[Tuple[int, int]]:
        return [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", [])]

    def _metrics_before(self, obs: dict) -> dict:
        p = int(obs["player"])
        f = obs["farms"][p]
        priv = obs.get("private", {})
        wheat_on_tiles = 0
        weeds = 0
        wheat_yield = 0
        for row in f.get("tiles", []):
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
                    wheat_on_tiles += 1
                    wheat_yield += int(tile.get("yield_units", 0))
                elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                    weeds += 1
        inventories = priv.get("inventories", [])
        carried = sum(int(inv.get("WHEAT", 0)) for inv in inventories)
        return {
            "money": float(f.get("money", 0.0)),
            "shed_wheat": int(priv.get("shed", {}).get("WHEAT", 0)),
            "seed_wheat": int(priv.get("seeds", {}).get("WHEAT", 0)),
            "wheat_on_tiles": wheat_on_tiles,
            "carried_wheat": carried,
            "yield_units": wheat_yield,
            "hands": len(f.get("hands", [])),
            "hires_today": int(f.get("hires_today", 0)),
            "unlocked": len(f.get("unlocked_quadrants", [])),
            "weeds": weeds,
        }

    def step(self, unit_action_ids: List[int], market_index: int) -> Tuple[dict, float, bool, dict, dict]:
        if self.last_obs is None:
            raise RuntimeError("Call reset() before step().")

        player = int(self.last_obs["player"])
        before = self._metrics_before(self.last_obs)
        farm = self.last_obs["farms"][player]
        action = build_action(unit_action_ids, market_index, len(farm.get("hands", [])))

        raw_obs, _raw_reward, done, info = self.trainer.step(action)
        obs = self._to_dict(raw_obs)
        after = self._metrics_before(obs)

        # Only actual bank change drives the learning reward; final money is the terminal metric.
        delta_money = after["money"] - before["money"]
        reward = delta_money / 100.0
        if done:
            reward += after["money"] / 1000.0

        # Metrics are derived from simulator state + the exact requested market order.
        seed_delta = max(0, after["seed_wheat"] - before["seed_wheat"])
        hands_delta = max(0, after["hands"] - before["hands"])
        land_delta = max(0, after["unlocked"] - before["unlocked"])
        planted_delta = max(0, after["wheat_on_tiles"] - before["wheat_on_tiles"])
        harvested_delta = max(0, before["wheat_on_tiles"] + before["carried_wheat"] - after["wheat_on_tiles"] - after["carried_wheat"])

        market_order = []
        try:
            market_order = action.get("market", [])[0]
        except Exception:
            pass
        sold = 0
        sale_revenue = 0.0
        if len(market_order) >= 3 and market_order[0] == "SELL" and market_order[1] == "WHEAT":
            requested = max(0, int(market_order[2]))
            sold = min(requested, before["shed_wheat"])
            if sold > 0:
                # Market quotes before commit; for a single-player order this is exactly
                # the sale quantity multiplied by the observed bank increase.
                sale_revenue = max(0.0, delta_money)

        self.stats["wheat_sold"] += sold
        self.stats["sell_revenue"] += sale_revenue
        self.stats["revenue"] = self.stats["sell_revenue"]
        self.stats["wheat_seed_bought"] += seed_delta
        self.stats["seed_cost"] += seed_delta * 10
        self.stats["workers_hired"] += hands_delta
        if hands_delta:
            self.stats["hire_cost"] += self._hire_cost(before["hires_today"])
        self.stats["land_purchases"] += land_delta
        if land_delta:
            self.stats["land_cost"] += self._land_cost(before["unlocked"])
        self.stats["wheat_planted"] += planted_delta
        self.stats["wheat_harvested"] += harvested_delta
        self.stats["weeds"] = after["weeds"]
        self.last_obs = obs
        return self.encoder.encode(obs, 0), float(reward), bool(done), obs, {**(info or {}), "metrics": dict(self.stats), "action": action}
