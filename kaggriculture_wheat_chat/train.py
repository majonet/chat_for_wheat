from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from agent import FactorizedDQNAgent
from config import Config
from env_wrapper import KaggricultureWrapper
from replay_buffer import PrioritizedReplay, Transition


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def beta_by_step(cfg: Config, step: int) -> float:
    frac = min(1.0, step / max(1, cfg.per_beta_steps))
    return cfg.per_beta0 + frac * (1.0 - cfg.per_beta0)


def train(episodes: int, cfg: Config, opponent: str, checkpoint: str | None = None) -> Path:
    seed_everything(cfg.seed)
    out_dir = Path(cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = KaggricultureWrapper(
        opponent=opponent,
        seed=cfg.seed,
        episode_steps=cfg.episode_steps,
        board_size=cfg.board_size,
        max_hands=cfg.max_hands,
    )
    agent = FactorizedDQNAgent(env.encoder, cfg)
    if checkpoint:
        agent.load(checkpoint, load_optimizer=True)

    replay = PrioritizedReplay(cfg.replay_capacity, cfg.per_alpha, cfg.per_eps)
    rows: List[Dict[str, float]] = []
    last_loss = 0.0

    for episode in range(1, episodes + 1):
        # Recreate the wrapper to avoid accidentally pinning every episode to one simulator instance/seed.
        env = KaggricultureWrapper(
            opponent=opponent,
            seed=cfg.seed + episode,
            episode_steps=cfg.episode_steps,
            board_size=cfg.board_size,
            max_hands=cfg.max_hands,
        )
        _, obs = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            player = int(obs["player"])
            n_units = 1 + len(obs["farms"][player].get("hands", []))
            unit_states = [env.encoder.encode(obs, i) for i in range(n_units)]
            market_state = env.encoder.encode_market(obs)
            unit_masks, market_mask = env.valid_masks(obs)

            unit_actions, market_action = agent.act(
                unit_states, market_state, unit_masks, market_mask, explore=True
            )
            next_state, reward, done, next_obs, info = env.step(unit_actions, market_action)
            next_player = int(next_obs["player"])
            next_units = 1 + len(next_obs["farms"][next_player].get("hands", []))
            next_unit_masks, next_market_mask = env.valid_masks(next_obs)

            max_units = 1 + cfg.max_hands
            unit_states_arr = np.zeros((max_units, env.obs_dim), dtype=np.float32)
            next_unit_states_arr = np.zeros((max_units, env.obs_dim), dtype=np.float32)
            unit_states_arr[:n_units] = np.stack(unit_states)
            next_unit_states = [env.encoder.encode(next_obs, i) for i in range(next_units)]
            next_unit_states_arr[:next_units] = np.stack(next_unit_states) if next_units else 0.0

            ua = np.full((max_units,), -1, dtype=np.int64)
            ua[:len(unit_actions)] = np.asarray(unit_actions, dtype=np.int64)
            um = np.zeros((max_units, len(unit_masks[0])), dtype=np.bool_)
            num = np.zeros_like(um)
            um[:len(unit_masks)] = np.stack(unit_masks)
            num[:len(next_unit_masks)] = np.stack(next_unit_masks)

            replay.add(
                Transition(
                    unit_states=unit_states_arr,
                    market_state=market_state.astype(np.float32),
                    unit_actions=ua,
                    market_action=int(market_action),
                    reward=float(reward),
                    next_unit_states=next_unit_states_arr,
                    next_market_state=env.encoder.encode_market(next_obs),
                    done=bool(done),
                    unit_masks=um,
                    next_unit_masks=num,
                    market_mask=market_mask.astype(np.bool_),
                    next_market_mask=next_market_mask.astype(np.bool_),
                )
            )

            agent.steps += 1
            episode_return += reward
            obs = next_obs

            if len(replay) >= cfg.learning_starts and agent.steps % cfg.train_freq == 0:
                batch, indices, weights = replay.sample(
                    cfg.batch_size,
                    beta_by_step(cfg, agent.steps),
                )
                last_loss, priorities = agent.learn(batch, weights)
                replay.update(indices, priorities)

            if agent.steps % cfg.target_update_freq == 0:
                agent.hard_update()

        final_money = float(obs["farms"][int(obs["player"])]["money"])
        row = {
            "episode": episode,
            "final_money": final_money,
            "return": episode_return,
            "loss": last_loss,
            "epsilon": agent.epsilon(),
            **env.stats,
        }
        rows.append(row)

        if episode % cfg.log_every_episodes == 0:
            print(
                f"episode={episode:5d} final_money={final_money:9.2f} "
                f"return={episode_return:9.3f} loss={last_loss:8.4f} eps={agent.epsilon():.3f}"
            )

        if episode % cfg.save_every_episodes == 0:
            agent.save(out_dir / "wheat_dqn_latest.pt")

    final_path = out_dir / "wheat_dqn.pt"
    agent.save(final_path)
    csv_path = out_dir / "training_log.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with (out_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(cfg), f, indent=2)
    return final_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--opponent", type=str, default="starter")
    p.add_argument("--checkpoint", type=str, default=None)
    args = p.parse_args()
    cfg = Config()
    path = train(args.episodes, cfg, args.opponent, args.checkpoint)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
