"""Kaggriculture submission entry point.

Set MODEL_PATH to a packaged checkpoint before submitting a model-backed agent.
The fallback is intentionally wheat-only and legal.
"""
from __future__ import annotations

import os
from pathlib import Path

from config import Config
from env_wrapper import KaggricultureWrapper
from heuristic_baseline import agent as heuristic_agent
from state_encoder import StateEncoder

MODEL_PATH = os.environ.get("KAGGRICULTURE_MODEL", "artifacts/wheat_dqn.pt")

_agent = None
_wrapper = None


def agent(obs: dict) -> dict:
    global _agent, _wrapper
    if _agent is None:
        try:
            from agent import FactorizedDQNAgent
            cfg = Config()
            _wrapper = type("EncoderHolder", (), {"encoder": StateEncoder(cfg.board_size, cfg.max_hands)})()
            _agent = FactorizedDQNAgent(_wrapper.encoder, cfg)
            _agent.load(Path(MODEL_PATH))
        except Exception:
            # A submission without packaged torch/weights remains a valid wheat-only agent.
            _agent = False

    if _agent is False:
        return heuristic_agent(obs)

    p = int(obs["player"])
    n_units = 1 + len(obs["farms"][p].get("hands", []))
    unit_states = [_wrapper.encoder.encode(obs, i) for i in range(n_units)]
    market_state = _wrapper.encoder.encode_market(obs)
    masks, market_mask = _wrapper.valid_masks(obs)
    unit_actions, market_action = _agent.act(unit_states, market_state, masks, market_mask, explore=False)

    from action_space import build_action
    return build_action(unit_actions, market_action, len(obs["farms"][p].get("hands", [])))
