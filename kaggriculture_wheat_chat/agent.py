from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from action_space import MARKET_ACTIONS, UNIT_ACTIONS
from config import Config
from network import SpatialDuelingQ


class FactorizedDQNAgent:
    def __init__(self, encoder, cfg: Config, device: str | None = None):
        self.cfg = cfg
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        common = dict(
            obs_dim=encoder.dim,
            board_size=encoder.board_size,
            tile_channels=encoder.tile_channels,
            global_dim=encoder.global_dim,
            unit_dim=encoder.unit_dim,
            unit_id_dim=encoder.unit_id_dim,
            hidden=cfg.hidden,
        )
        self.unit = SpatialDuelingQ(n_actions=len(UNIT_ACTIONS), **common).to(self.device)
        self.unit_target = SpatialDuelingQ(n_actions=len(UNIT_ACTIONS), **common).to(self.device)
        self.market = SpatialDuelingQ(n_actions=len(MARKET_ACTIONS), **common).to(self.device)
        self.market_target = SpatialDuelingQ(n_actions=len(MARKET_ACTIONS), **common).to(self.device)
        self.hard_update()
        self.optimizer = torch.optim.AdamW(
            list(self.unit.parameters()) + list(self.market.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.steps = 0

    def hard_update(self) -> None:
        self.unit_target.load_state_dict(self.unit.state_dict())
        self.market_target.load_state_dict(self.market.state_dict())

    def epsilon(self) -> float:
        frac = min(1.0, self.steps / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @staticmethod
    def _masked_argmax(q: torch.Tensor, mask: np.ndarray) -> int:
        q = q.clone()
        q[~torch.as_tensor(mask, dtype=torch.bool, device=q.device)] = -1e9
        return int(torch.argmax(q).item())

    @torch.no_grad()
    def act(self, unit_states: List[np.ndarray], market_state: np.ndarray,
            unit_masks: List[np.ndarray], market_mask: np.ndarray, explore: bool = True) -> Tuple[List[int], int]:
        eps = self.epsilon() if explore else 0.0
        unit_actions: List[int] = []
        for state, mask in zip(unit_states, unit_masks):
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            if explore and random.random() < eps:
                valid = np.flatnonzero(mask)
                unit_actions.append(int(np.random.choice(valid)))
            else:
                q = self.unit(x)[0]
                unit_actions.append(self._masked_argmax(q, mask))

        mx = torch.as_tensor(market_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if explore and random.random() < eps:
            valid = np.flatnonzero(market_mask)
            market_action = int(np.random.choice(valid))
        else:
            market_action = self._masked_argmax(self.market(mx)[0], market_mask)
        return unit_actions, market_action

    def learn(self, batch, weights: np.ndarray) -> Tuple[float, np.ndarray]:
        device = self.device
        w = torch.as_tensor(weights, dtype=torch.float32, device=device)

        market_states = torch.as_tensor(np.stack([b.market_state for b in batch]), dtype=torch.float32, device=device)
        next_market_states = torch.as_tensor(np.stack([b.next_market_state for b in batch]), dtype=torch.float32, device=device)
        rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32, device=device)
        dones = torch.as_tensor([b.done for b in batch], dtype=torch.float32, device=device)

        market_mask = torch.as_tensor(np.stack([b.market_mask for b in batch]), dtype=torch.bool, device=device)
        next_market_mask = torch.as_tensor(np.stack([b.next_market_mask for b in batch]), dtype=torch.bool, device=device)
        ma = torch.as_tensor([b.market_action for b in batch], dtype=torch.long, device=device)
        mq = self.market(market_states, market_mask).gather(1, ma[:, None]).squeeze(1)
        with torch.no_grad():
            next_online = self.market(next_market_states, next_market_mask)
            next_a = next_online.argmax(dim=1)
            next_q = self.market_target(next_market_states, next_market_mask).gather(1, next_a[:, None]).squeeze(1)
            mt = rewards + self.cfg.gamma * (1.0 - dones) * next_q
        mtd = mq - mt
        market_loss = (F.smooth_l1_loss(mq, mt, reduction="none") * w).mean()

        max_slots = batch[0].unit_states.shape[0]
        unit_losses = []
        td_errors = [torch.abs(mtd).detach().cpu().numpy()]
        for slot in range(max_slots):
            active = np.array([b.unit_actions[slot] >= 0 for b in batch], dtype=bool)
            if not active.any():
                continue
            idx = np.flatnonzero(active)
            x = torch.as_tensor(np.stack([batch[i].unit_states[slot] for i in idx]), dtype=torch.float32, device=device)
            nx = torch.as_tensor(np.stack([batch[i].next_unit_states[slot] for i in idx]), dtype=torch.float32, device=device)
            masks_np = np.stack([batch[i].unit_masks[slot] for i in idx])
            nmasks_np = np.stack([batch[i].next_unit_masks[slot] for i in idx])
            masks = torch.as_tensor(masks_np, dtype=torch.bool, device=device)
            nmasks = torch.as_tensor(nmasks_np, dtype=torch.bool, device=device)
            aa = torch.as_tensor([batch[i].unit_actions[slot] for i in idx], dtype=torch.long, device=device)
            ww = w[idx]
            rr = rewards[idx]
            dd = dones[idx]

            q = self.unit(x, masks).gather(1, aa[:, None]).squeeze(1)
            with torch.no_grad():
                has_next = nmasks.any(dim=1)
                if has_next.any():
                    online_next = self.unit(nx, nmasks)
                    next_a = online_next.argmax(dim=1)
                    boot = self.unit_target(nx, nmasks).gather(1, next_a[:, None]).squeeze(1)
                    boot = torch.where(has_next, boot, torch.zeros_like(boot))
                else:
                    boot = torch.zeros_like(rr)
                target = rr + self.cfg.gamma * (1.0 - dd) * boot
            td = q - target
            unit_losses.append((F.smooth_l1_loss(q, target, reduction="none") * ww).mean())
            td_errors.append(torch.abs(td).detach().cpu().numpy())

        unit_loss = torch.stack(unit_losses).mean() if unit_losses else torch.tensor(0.0, device=device)
        loss = unit_loss + market_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.unit.parameters()) + list(self.market.parameters()), self.cfg.grad_clip
        )
        self.optimizer.step()
        priorities = np.maximum.reduce(td_errors)
        return float(loss.item()), priorities + self.cfg.per_eps

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "unit": self.unit.state_dict(),
                "unit_target": self.unit_target.state_dict(),
                "market": self.market.state_dict(),
                "market_target": self.market_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "config": vars(self.cfg),
            },
            path,
        )

    def load(self, path: str | Path, load_optimizer: bool = False) -> None:
        payload = torch.load(path, map_location=self.device)
        self.unit.load_state_dict(payload["unit"])
        self.market.load_state_dict(payload["market"])
        if "unit_target" in payload:
            self.unit_target.load_state_dict(payload["unit_target"])
        else:
            self.unit_target.load_state_dict(payload["unit"])
        if "market_target" in payload:
            self.market_target.load_state_dict(payload["market_target"])
        else:
            self.market_target.load_state_dict(payload["market"])
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.steps = int(payload.get("steps", 0))
