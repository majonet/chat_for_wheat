from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Transition:
    unit_states: np.ndarray       # [max_units, obs_dim], zero-padded
    market_state: np.ndarray      # [obs_dim]
    unit_actions: np.ndarray      # [max_units], -1 for inactive slots
    market_action: int
    reward: float
    next_unit_states: np.ndarray
    next_market_state: np.ndarray
    done: bool
    unit_masks: np.ndarray        # [max_units, n_unit_actions]
    next_unit_masks: np.ndarray
    market_mask: np.ndarray
    next_market_mask: np.ndarray


class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-5):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.data: List[Transition | None] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, transition: Transition, priority: float | None = None) -> None:
        p = float(priority if priority is not None else (self.priorities.max() if self.size else 1.0))
        self.data[self.pos] = transition
        self.priorities[self.pos] = max(self.eps, p)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        if self.size < batch_size:
            raise ValueError("Not enough replay samples")
        p = self.priorities[: self.size] ** self.alpha
        p /= p.sum()
        idx = np.random.choice(self.size, batch_size, replace=False, p=p)
        weights = (self.size * p[idx]) ** (-beta)
        weights /= weights.max()
        return [self.data[i] for i in idx], idx.astype(np.int64), weights.astype(np.float32)

    def update(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = np.maximum(self.eps, priorities.astype(np.float32))

    def __len__(self) -> int:
        return self.size
