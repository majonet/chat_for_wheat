from __future__ import annotations

import argparse
import json

from agent import FactorizedDQNAgent
from config import Config
from env_wrapper import KaggricultureWrapper


def run(model_path: str, opponent: str = "starter") -> None:
    cfg = Config()
    env = KaggricultureWrapper(opponent=opponent, seed=cfg.seed + 42, episode_steps=720, board_size=10, max_hands=cfg.max_hands)
    _, obs = env.reset()
    agent = FactorizedDQNAgent(env.encoder, cfg)
    agent.load(model_path)
    done = False
    t = 0
    while not done:
        p = int(obs["player"])
        n_units = 1 + len(obs["farms"][p].get("hands", []))
        unit_states = [env.encoder.encode(obs, i) for i in range(n_units)]
        market_state = env.encoder.encode_market(obs)
        masks, mm = env.valid_masks(obs)
        acts, market = agent.act(unit_states, market_state, masks, mm, explore=False)
        _, _, done, next_obs, info = env.step(acts, market)
        raw_action = info.get("action", {})
        if raw_action.get("farmer") != ["PASS"] or raw_action.get("hands") or raw_action.get("market"):
            print(f"Day {int(obs.get('day', 0)):02d} turn {int(obs.get('hour', 0)):02d}: {json.dumps(raw_action)}")
        obs = next_obs
        t += 1
    print("final_money:", obs["farms"][int(obs["player"])]["money"])
    print("stats:", json.dumps(env.stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--opponent", default="starter")
    args = parser.parse_args()
    run(args.model, args.opponent)
