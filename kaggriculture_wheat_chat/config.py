from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # Environment
    episode_steps: int = 720
    turns_per_day: int = 24
    board_size: int = 10
    starting_money: float = 3000.0
    max_hands: int = 8
    shed_capacity: int = 100
    max_market_orders_per_turn: int = 10

    # Wheat-only hard constraints
    wheat_seed_cost: int = 10
    wheat_base_price: int = 25
    wheat_first_yield_day: int = 2
    wheat_max_yield_day: int = 4
    wheat_max_yield: int = 6

    # RL
    gamma: float = 0.995
    lr: float = 3e-4
    batch_size: int = 128
    hidden: int = 256
    learning_starts: int = 10_000
    train_freq: int = 4
    target_update_freq: int = 2_000
    grad_clip: float = 10.0
    weight_decay: float = 1e-5

    # Prioritized replay
    replay_capacity: int = 250_000
    per_alpha: float = 0.6
    per_beta0: float = 0.4
    per_beta_steps: int = 500_000
    per_eps: float = 1e-5

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 350_000

    # Market quantity action bins. One order is enough because quantity is encoded directly.
    quantity_bins: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 100)

    # Reproducibility / output
    seed: int = 17
    device: str = "auto"
    checkpoint_dir: str = "artifacts"
    log_every_episodes: int = 10
    save_every_episodes: int = 100

    # Training opponents can be overridden from CLI.
    opponents: Tuple[str, ...] = field(default_factory=lambda: ("starter", "random"))
