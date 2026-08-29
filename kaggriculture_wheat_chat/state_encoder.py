from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


N_TILE_CHANNELS = 11


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _pos(x: Sequence[int]) -> Tuple[int, int]:
    return int(x[0]), int(x[1])


class StateEncoder:
    """Fixed-size feature encoder with a spatial 10x10 tile tensor flattened for storage.

    Channel order:
        0 empty/unlocked
        1 locked
        2 wheat plant
        3 weed
        4 harvestable wheat
        5 watered today
        6 unwatered >= 1 day
        7 unwatered >= 2 days
        8 yield units normalized into the tile channel
        9 crop age normalized into the tile channel
       10 shed-access tile
    """

    def __init__(self, board_size: int = 10, max_hands: int = 8):
        self.board_size = int(board_size)
        self.max_hands = int(max_hands)
        self.tile_channels = N_TILE_CHANNELS
        self.spatial_size = self.tile_channels * self.board_size * self.board_size
        self.global_dim = 27
        self.unit_dim = 10
        self.unit_id_dim = 1 + self.max_hands
        self.dim = self.global_dim + self.spatial_size + self.unit_dim + self.unit_id_dim

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _shed_access(self) -> set[Tuple[int, int]]:
        h = self.board_size // 2
        return {(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)}

    def _unlocked(self, farm: dict, x: int, y: int) -> bool:
        tile = farm["tiles"][y][x]
        return tile != "LOCKED"

    def _unit_positions(self, farm: dict) -> List[Tuple[int, int]]:
        return [_pos(farm["farmer"])] + [_pos(p) for p in farm.get("hands", [])[: self.max_hands]]

    def _global_features(self, obs: dict, farm: dict, private: dict, unit_positions: List[Tuple[int, int]]) -> List[float]:
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        step = int(obs.get("step", day * 24 + hour))
        remaining = max(0, 720 - step)

        money = _safe_float(farm.get("money", 0.0))
        price = _safe_float(obs.get("market", {}).get("prices", {}).get("WHEAT", 25.0), 25.0)
        market_inv = _safe_float(obs.get("market", {}).get("inventory", {}).get("WHEAT", 10000), 10000.0)
        seed_count = _safe_float(private.get("seeds", {}).get("WHEAT", 0))
        shed_wheat = _safe_float(private.get("shed", {}).get("WHEAT", 0))

        wheat_tiles: List[Tuple[int, int, dict]] = []
        harvest_tiles: List[Tuple[int, int]] = []
        thirsty_tiles: List[Tuple[int, int]] = []
        weeds = 0
        free = 0

        for y, row in enumerate(farm.get("tiles", [])[: self.board_size]):
            for x, tile in enumerate(row[: self.board_size]):
                if tile == "LOCKED":
                    continue
                if tile is None:
                    free += 1
                    continue
                if isinstance(tile, dict) and tile.get("kind") == "WEED":
                    weeds += 1
                    continue
                if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
                    wheat_tiles.append((x, y, tile))
                    if not bool(tile.get("watered_today", False)):
                        thirsty_tiles.append((x, y))
                    age = day - int(tile.get("planted_day", day))
                    if age >= 2 and int(tile.get("yield_units", 0)) > 0:
                        harvest_tiles.append((x, y))

        unit_pos = unit_positions[0]
        d_water = min((self._manhattan(unit_pos, t) for t in thirsty_tiles), default=0)
        d_harvest = min((self._manhattan(unit_pos, t) for t in harvest_tiles), default=0)
        d_shed = min((self._manhattan(unit_pos, t) for t in self._shed_access()), default=0)

        expected_harvest = len(wheat_tiles) * self._expected_tile_yield(day, wheat_tiles) * price
        expected_profit_tile = max(0.0, self._expected_tile_yield(day, wheat_tiles) * price - 10.0)
        hire_cost = self._hire_cost(int(farm.get("hires_today", 0)))
        next_land_cost = self._land_cost(len(farm.get("unlocked_quadrants", [])))
        liquid_wealth = money + shed_wheat * price

        return [
            day / 30.0,
            hour / 24.0,
            step / 720.0,
            remaining / 720.0,
            money / 5000.0,
            price / 100.0,
            (price - 25.0) / 25.0,
            market_inv / 10000.0,
            (market_inv - 10000.0) / 10000.0,
            seed_count / 100.0,
            shed_wheat / 100.0,
            len(wheat_tiles) / 100.0,
            len(harvest_tiles) / 100.0,
            len(thirsty_tiles) / 100.0,
            weeds / 100.0,
            free / 100.0,
            len(farm.get("unlocked_quadrants", [])) / 4.0,
            len(unit_positions) / (1 + self.max_hands),
            int(farm.get("hires_today", 0)) / 8.0,
            d_water / 20.0,
            d_harvest / 20.0,
            d_shed / 20.0,
            liquid_wealth / 10000.0,
            expected_harvest / 10000.0,
            expected_profit_tile / 100.0,
            hire_cost / 100.0,
            next_land_cost / 4000.0,
        ]

    def _expected_tile_yield(self, day: int, wheat_tiles: Sequence[Tuple[int, int, dict]]) -> float:
        if not wheat_tiles:
            return 6.0
        # Feature only: a conservative estimate, not a simulator rule.
        vals = []
        for _, _, tile in wheat_tiles:
            age = max(0, day - int(tile.get("planted_day", day)))
            vals.append(min(6, 1 + max(0, age - 1)))
        return float(np.mean(vals))

    @staticmethod
    def _hire_cost(n_already_today: int) -> int:
        a, b = 1, 1
        for _ in range(n_already_today):
            a, b = b, a + b
        return a

    @staticmethod
    def _land_cost(n_unlocked: int) -> int:
        return {1: 1000, 2: 2000, 3: 4000}.get(n_unlocked, 10**9)

    def _spatial(self, obs: dict, farm: dict) -> np.ndarray:
        board = np.zeros((self.tile_channels, self.board_size, self.board_size), dtype=np.float32)
        day = int(obs.get("day", 0))
        shed_access = self._shed_access()
        for y, row in enumerate(farm.get("tiles", [])[: self.board_size]):
            for x, tile in enumerate(row[: self.board_size]):
                if tile == "LOCKED":
                    board[1, y, x] = 1.0
                elif tile is None:
                    board[0, y, x] = 1.0
                elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                    board[3, y, x] = 1.0
                elif isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
                    board[2, y, x] = 1.0
                    age = max(0, day - int(tile.get("planted_day", day)))
                    board[9, y, x] = min(age / 4.0, 1.0)
                    board[8, y, x] = min(max(0, int(tile.get("yield_units", 0))) / 6.0, 1.0)
                    board[5, y, x] = float(bool(tile.get("watered_today", False)))
                    cu = int(tile.get("consecutive_unwatered", 0))
                    board[6, y, x] = float(cu >= 1)
                    board[7, y, x] = float(cu >= 2)
                    if age >= 2 and int(tile.get("yield_units", 0)) > 0:
                        board[4, y, x] = 1.0
                if (x, y) in shed_access:
                    board[10, y, x] = 1.0
        return board

    def _unit_features(self, farm: dict, private: dict, unit_index: int) -> List[float]:
        positions = self._unit_positions(farm)
        if unit_index >= len(positions):
            return [0.0] * self.unit_dim
        x, y = positions[unit_index]
        invs = private.get("inventories", [])
        inv = invs[unit_index] if unit_index < len(invs) else {}

        wheat_tiles: List[Tuple[int, int]] = []
        harvest_tiles: List[Tuple[int, int]] = []
        for yy, row in enumerate(farm.get("tiles", [])[: self.board_size]):
            for xx, tile in enumerate(row[: self.board_size]):
                if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
                    wheat_tiles.append((xx, yy))
                    age = int(self._obs_day_cache) - int(tile.get("planted_day", self._obs_day_cache))
                    if age >= 2 and int(tile.get("yield_units", 0)) > 0:
                        harvest_tiles.append((xx, yy))

        thirsty = [q for q in wheat_tiles if not bool(farm["tiles"][q[1]][q[0]].get("watered_today", False))]
        return [
            x / max(1, self.board_size - 1),
            y / max(1, self.board_size - 1),
            _safe_float(inv.get("WHEAT", 0.0)) / 20.0,
            min((self._manhattan((x, y), q) for q in thirsty), default=0) / 20.0,
            min((self._manhattan((x, y), q) for q in harvest_tiles), default=0) / 20.0,
            min((self._manhattan((x, y), q) for q in self._shed_access()), default=0) / 20.0,
            float((x, y) in self._shed_access()),
            float(unit_index > 0),
            len(farm.get("hands", [])) / max(1, self.max_hands),
            _safe_float(farm.get("hires_today", 0)) / 8.0,
        ]

    def encode(self, obs: dict, unit_index: int = 0) -> np.ndarray:
        player = int(obs.get("player", 0))
        farm = obs["farms"][player]
        private = obs.get("private", {})
        positions = self._unit_positions(farm)
        self._obs_day_cache = int(obs.get("day", 0))
        g = self._global_features(obs, farm, private, positions)
        spatial = self._spatial(obs, farm).reshape(-1).tolist()
        unit = self._unit_features(farm, private, unit_index)
        unit_id = [0.0] * self.unit_id_dim
        if 0 <= unit_index < self.unit_id_dim:
            unit_id[unit_index] = 1.0
        return np.asarray(g + spatial + unit + unit_id, dtype=np.float32)

    def encode_market(self, obs: dict) -> np.ndarray:
        # Zero unit slot, plus an all-zero unit id. Market head uses global+spatial only.
        return self.encode(obs, unit_index=-1)
