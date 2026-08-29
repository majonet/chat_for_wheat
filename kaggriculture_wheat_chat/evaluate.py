from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agent import FactorizedDQNAgent
from config import Config
from env_wrapper import KaggricultureWrapper


def evaluate(model_path: str, episodes: int = 50, opponent: str = "starter") -> dict:
    cfg = Config()
    results = []
    details = []
    agent = None

    for episode in range(episodes):
        env = KaggricultureWrapper(
            opponent=opponent,
            seed=cfg.seed + 10_000 + episode,
            episode_steps=cfg.episode_steps,
            board_size=cfg.board_size,
            max_hands=cfg.max_hands,
        )
        _, obs = env.reset()
        if agent is None:
            agent = FactorizedDQNAgent(env.encoder, cfg)
            agent.load(model_path)
        done = False
        while not done:
            p = int(obs["player"])
            n_units = 1 + len(obs["farms"][p].get("hands", []))
            unit_states = [env.encoder.encode(obs, i) for i in range(n_units)]
            market_state = env.encoder.encode_market(obs)
            unit_masks, market_mask = env.valid_masks(obs)
            acts, market = agent.act(unit_states, market_state, unit_masks, market_mask, explore=False)
            _, _, done, obs, _ = env.step(acts, market)

        final_money = float(obs["farms"][int(obs["player"])]["money"])
        results.append(final_money)
        details.append({"episode": episode, "final_money": final_money, **env.stats})

    a = np.asarray(results, dtype=np.float64)
    summary = {
        "mean_final_money": float(a.mean()),
        "median_final_money": float(np.median(a)),
        "best_final_money": float(a.max()),
        "worst_final_money": float(a.min()),
        "std_final_money": float(a.std()),
        "episodes": int(episodes),
        "opponent": opponent,
        "model": str(model_path),
    }
    print(json.dumps(summary, indent=2))
    out = Path(model_path).with_name("evaluation.json")
    out.write_text(json.dumps({"summary": summary, "episodes": details}, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--opponent", default="starter")
    args = p.parse_args()
    evaluate(args.model, args.episodes, args.opponent)


if __name__ == "__main__":
    main()
