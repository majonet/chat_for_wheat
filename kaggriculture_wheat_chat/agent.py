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
    """
    Factorized Dueling Double-DQN agent.

    Two Q-networks are maintained:

    1. Unit network:
       Shared policy for Farmer + Farm Hands.

    2. Market network:
       BUY_SEED / SELL / HIRE / BUY_LAND decisions.

    Prioritized replay receives exactly one scalar priority per transition.
    For a transition, priority is:

        max(
            abs(market TD-error),
            abs(unit TD-errors for all active units)
        )
    """

    def __init__(
        self,
        encoder,
        cfg: Config,
        device: str | None = None,
    ):
        self.cfg = cfg

        # ------------------------------------------------------------
        # Device
        # ------------------------------------------------------------
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # ------------------------------------------------------------
        # Network dimensions
        # ------------------------------------------------------------
        common_kwargs = dict(
            obs_dim=encoder.dim,
            board_size=encoder.board_size,
            tile_channels=encoder.tile_channels,
            global_dim=encoder.global_dim,
            unit_dim=encoder.unit_dim,
            unit_id_dim=encoder.unit_id_dim,
            hidden=cfg.hidden,
        )

        # ------------------------------------------------------------
        # Online networks
        # ------------------------------------------------------------
        self.unit = SpatialDuelingQ(
            n_actions=len(UNIT_ACTIONS),
            **common_kwargs,
        ).to(self.device)

        self.market = SpatialDuelingQ(
            n_actions=len(MARKET_ACTIONS),
            **common_kwargs,
        ).to(self.device)

        # ------------------------------------------------------------
        # Target networks
        # ------------------------------------------------------------
        self.unit_target = SpatialDuelingQ(
            n_actions=len(UNIT_ACTIONS),
            **common_kwargs,
        ).to(self.device)

        self.market_target = SpatialDuelingQ(
            n_actions=len(MARKET_ACTIONS),
            **common_kwargs,
        ).to(self.device)

        self.hard_update()

        # ------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            list(self.unit.parameters())
            + list(self.market.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # Number of gradient updates performed.
        #
        # This is also used for epsilon scheduling in the current
        # training implementation.
        self.steps = 0

    # ==================================================================
    # Target network
    # ==================================================================

    @torch.no_grad()
    def hard_update(self) -> None:
        """Copy online network parameters into target networks."""
        self.unit_target.load_state_dict(self.unit.state_dict())
        self.market_target.load_state_dict(self.market.state_dict())

    # ==================================================================
    # Exploration
    # ==================================================================

    def epsilon(self) -> float:
        """
        Linear epsilon decay.

        Example:
            epsilon = 1.0 at step 0
            epsilon -> 0.05 by eps_decay_steps
        """
        frac = min(
            1.0,
            self.steps / max(1, self.cfg.eps_decay_steps),
        )

        return (
            self.cfg.eps_start
            + frac * (self.cfg.eps_end - self.cfg.eps_start)
        )

    # ==================================================================
    # Mask helpers
    # ==================================================================

    @staticmethod
    def _masked_argmax(
        q: torch.Tensor,
        mask: np.ndarray | torch.Tensor,
    ) -> int:
        """
        Argmax over valid actions only.

        Invalid actions receive a very negative Q-value.
        """
        q = q.clone()

        if isinstance(mask, np.ndarray):
            mask_tensor = torch.as_tensor(
                mask,
                dtype=torch.bool,
                device=q.device,
            )
        else:
            mask_tensor = mask.to(
                device=q.device,
                dtype=torch.bool,
            )

        # Safety fallback.
        #
        # A correctly implemented environment should always provide
        # at least one valid action (normally PASS / NONE).
        if not bool(mask_tensor.any().item()):
            return 0

        q[~mask_tensor] = -1e9

        return int(torch.argmax(q).item())

    @staticmethod
    def _masked_argmax_batch(
        q: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched masked argmax.

        q:
            [B, A]

        mask:
            [B, A]

        returns:
            [B]
        """
        mask = mask.bool()

        # We expect every row to have at least one legal action.
        # For defensive programming, make action 0 legal if a malformed
        # all-false mask is encountered.
        no_valid = ~mask.any(dim=1)

        if no_valid.any():
            mask = mask.clone()
            mask[no_valid, 0] = True

        masked_q = q.masked_fill(~mask, -1e9)

        return masked_q.argmax(dim=1)

    # ==================================================================
    # Action selection
    # ==================================================================

    @torch.no_grad()
    def act(
        self,
        unit_states: List[np.ndarray],
        market_state: np.ndarray,
        unit_masks: List[np.ndarray],
        market_mask: np.ndarray,
        explore: bool = True,
    ) -> Tuple[List[int], int]:
        """
        Select actions for all units and the market.

        Parameters
        ----------
        unit_states:
            List of encoded states, one for each active unit.

        market_state:
            Encoded market/global state.

        unit_masks:
            Legal-action mask for each unit.

        market_mask:
            Legal-action mask for market actions.

        explore:
            Whether epsilon-greedy exploration is enabled.

        Returns
        -------
        unit_actions, market_action
        """
        eps = self.epsilon() if explore else 0.0

        # --------------------------------------------------------------
        # Unit actions
        # --------------------------------------------------------------
        unit_actions: List[int] = []

        for state, mask in zip(unit_states, unit_masks):
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            # Epsilon exploration over VALID actions only.
            if explore and random.random() < eps:
                valid = np.flatnonzero(mask)

                if len(valid) == 0:
                    # Defensive fallback.
                    action = 0
                else:
                    action = int(np.random.choice(valid))

                unit_actions.append(action)
                continue

            # Greedy action.
            q = self.unit(state_tensor)[0]

            action = self._masked_argmax(
                q,
                mask,
            )

            unit_actions.append(action)

        # --------------------------------------------------------------
        # Market action
        # --------------------------------------------------------------
        market_tensor = torch.as_tensor(
            market_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        if explore and random.random() < eps:
            valid = np.flatnonzero(market_mask)

            if len(valid) == 0:
                market_action = 0
            else:
                market_action = int(np.random.choice(valid))

        else:
            market_q = self.market(market_tensor)[0]

            market_action = self._masked_argmax(
                market_q,
                market_mask,
            )

        return unit_actions, market_action

    # ==================================================================
    # Learning
    # ==================================================================

    def learn(
        self,
        batch,
        weights: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Perform one Double-DQN update over a PER batch.

        The replay buffer contains factorized transitions:

            market transition
            +
            zero-padded unit transitions

        We calculate:

            market loss
            +
            mean unit loss

        and return exactly one PER priority for every sampled transition.
        """

        device = self.device

        batch_size = len(batch)

        if batch_size == 0:
            raise ValueError("learn() received an empty batch")

        # --------------------------------------------------------------
        # Importance-sampling weights
        # --------------------------------------------------------------
        weights_t = torch.as_tensor(
            weights,
            dtype=torch.float32,
            device=device,
        )

        if weights_t.ndim != 1 or weights_t.shape[0] != batch_size:
            raise ValueError(
                "PER weights must have shape "
                f"[{batch_size}], got {tuple(weights_t.shape)}"
            )

        # ==============================================================
        # MARKET BRANCH
        # ==============================================================

        market_states = torch.as_tensor(
            np.stack(
                [b.market_state for b in batch],
                axis=0,
            ),
            dtype=torch.float32,
            device=device,
        )

        next_market_states = torch.as_tensor(
            np.stack(
                [b.next_market_state for b in batch],
                axis=0,
            ),
            dtype=torch.float32,
            device=device,
        )

        rewards = torch.as_tensor(
            np.asarray(
                [b.reward for b in batch],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=device,
        )

        dones = torch.as_tensor(
            np.asarray(
                [b.done for b in batch],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=device,
        )

        market_masks = torch.as_tensor(
            np.stack(
                [b.market_mask for b in batch],
                axis=0,
            ),
            dtype=torch.bool,
            device=device,
        )

        next_market_masks = torch.as_tensor(
            np.stack(
                [b.next_market_mask for b in batch],
                axis=0,
            ),
            dtype=torch.bool,
            device=device,
        )

        market_actions = torch.as_tensor(
            np.asarray(
                [b.market_action for b in batch],
                dtype=np.int64,
            ),
            dtype=torch.long,
            device=device,
        )

        # --------------------------------------------------------------
        # Current Q(s,a)
        # --------------------------------------------------------------
        market_q_all = self.market(
            market_states,
            market_masks,
        )

        market_q = market_q_all.gather(
            1,
            market_actions.unsqueeze(1),
        ).squeeze(1)

        # --------------------------------------------------------------
        # Double-DQN next action:
        #
        # action = argmax Q_online(s', a)
        #
        # value  = Q_target(s', action)
        # --------------------------------------------------------------
        with torch.no_grad():

            online_next_market_q = self.market(
                next_market_states,
                next_market_masks,
            )

            next_market_actions = self._masked_argmax_batch(
                online_next_market_q,
                next_market_masks,
            )

            target_next_market_q = self.market_target(
                next_market_states,
                next_market_masks,
            )

            next_market_q = target_next_market_q.gather(
                1,
                next_market_actions.unsqueeze(1),
            ).squeeze(1)

            # If a malformed all-false mask slipped through, the helper
            # handles it. Terminal transitions must never bootstrap.
            market_target = (
                rewards
                + self.cfg.gamma
                * (1.0 - dones)
                * next_market_q
            )

        market_td = market_q - market_target

        market_loss_per_sample = F.smooth_l1_loss(
            market_q,
            market_target,
            reduction="none",
        )

        market_loss = (
            market_loss_per_sample * weights_t
        ).mean()

        # ==============================================================
        # PER PRIORITIES
        # ==============================================================

        # CRITICAL:
        #
        # We need one priority for EACH transition in the replay batch.
        #
        # Market TD-error is already [batch_size], so start there.
        #
        # We DO NOT use:
        #
        #     np.maximum.reduce(td_errors)
        #
        # across differently sized arrays.
        priorities = (
            torch.abs(market_td)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # ==============================================================
        # UNIT BRANCH
        # ==============================================================

        max_slots = int(batch[0].unit_states.shape[0])

        unit_losses: List[torch.Tensor] = []

        for slot in range(max_slots):

            # ----------------------------------------------------------
            # Active transitions for this unit slot
            #
            # -1 means this slot was inactive.
            # ----------------------------------------------------------
            active_np = np.asarray(
                [
                    int(b.unit_actions[slot]) >= 0
                    for b in batch
                ],
                dtype=bool,
            )

            if not active_np.any():
                continue

            active_indices = np.flatnonzero(active_np)

            # ----------------------------------------------------------
            # Gather only active examples for this slot.
            # ----------------------------------------------------------
            current_states = torch.as_tensor(
                np.stack(
                    [
                        batch[i].unit_states[slot]
                        for i in active_indices
                    ],
                    axis=0,
                ),
                dtype=torch.float32,
                device=device,
            )

            next_states = torch.as_tensor(
                np.stack(
                    [
                        batch[i].next_unit_states[slot]
                        for i in active_indices
                    ],
                    axis=0,
                ),
                dtype=torch.float32,
                device=device,
            )

            current_masks = torch.as_tensor(
                np.stack(
                    [
                        batch[i].unit_masks[slot]
                        for i in active_indices
                    ],
                    axis=0,
                ),
                dtype=torch.bool,
                device=device,
            )

            next_masks = torch.as_tensor(
                np.stack(
                    [
                        batch[i].next_unit_masks[slot]
                        for i in active_indices
                    ],
                    axis=0,
                ),
                dtype=torch.bool,
                device=device,
            )

            unit_actions = torch.as_tensor(
                np.asarray(
                    [
                        batch[i].unit_actions[slot]
                        for i in active_indices
                    ],
                    dtype=np.int64,
                ),
                dtype=torch.long,
                device=device,
            )

            unit_weights = weights_t[
                torch.as_tensor(
                    active_indices,
                    dtype=torch.long,
                    device=device,
                )
            ]

            unit_rewards = rewards[
                torch.as_tensor(
                    active_indices,
                    dtype=torch.long,
                    device=device,
                )
            ]

            unit_dones = dones[
                torch.as_tensor(
                    active_indices,
                    dtype=torch.long,
                    device=device,
                )
            ]

            # ----------------------------------------------------------
            # Q(s,a)
            # ----------------------------------------------------------
            unit_q_all = self.unit(
                current_states,
                current_masks,
            )

            unit_q = unit_q_all.gather(
                1,
                unit_actions.unsqueeze(1),
            ).squeeze(1)

            # ----------------------------------------------------------
            # Double-DQN next Q
            # ----------------------------------------------------------
            with torch.no_grad():

                online_next_q = self.unit(
                    next_states,
                    next_masks,
                )

                next_actions = self._masked_argmax_batch(
                    online_next_q,
                    next_masks,
                )

                target_next_q_all = self.unit_target(
                    next_states,
                    next_masks,
                )

                next_q = target_next_q_all.gather(
                    1,
                    next_actions.unsqueeze(1),
                ).squeeze(1)

                # No bootstrap for terminal states.
                unit_target = (
                    unit_rewards
                    + self.cfg.gamma
                    * (1.0 - unit_dones)
                    * next_q
                )

            unit_td = unit_q - unit_target

            # ----------------------------------------------------------
            # Weighted Huber loss
            # ----------------------------------------------------------
            unit_loss_per_sample = F.smooth_l1_loss(
                unit_q,
                unit_target,
                reduction="none",
            )

            weighted_unit_loss = (
                unit_loss_per_sample * unit_weights
            ).mean()

            unit_losses.append(weighted_unit_loss)

            # ----------------------------------------------------------
            # IMPORTANT PER FIX
            #
            # unit_td is only defined for active examples in this slot.
            #
            # Merge its abs(TD) back into the corresponding positions
            # in the full batch.
            # ----------------------------------------------------------
            unit_priorities = (
                torch.abs(unit_td)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            priorities[active_indices] = np.maximum(
                priorities[active_indices],
                unit_priorities,
            )

        # --------------------------------------------------------------
        # Combine losses
        # --------------------------------------------------------------
        if unit_losses:
            unit_loss = torch.stack(unit_losses).mean()
        else:
            unit_loss = torch.zeros(
                (),
                dtype=torch.float32,
                device=device,
            )

        total_loss = unit_loss + market_loss

        # --------------------------------------------------------------
        # Backprop
        # --------------------------------------------------------------
        self.optimizer.zero_grad(set_to_none=True)

        total_loss.backward()

        # Gradient clipping is important for long-horizon DQN.
        torch.nn.utils.clip_grad_norm_(
            list(self.unit.parameters())
            + list(self.market.parameters()),
            self.cfg.grad_clip,
        )

        self.optimizer.step()

        # Number of optimizer updates.
        self.steps += 1

        # --------------------------------------------------------------
        # PER numerical safety
        # --------------------------------------------------------------
        priorities = np.nan_to_num(
            priorities,
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )

        priorities = np.maximum(
            priorities,
            float(self.cfg.per_eps),
        ).astype(np.float32)

        return float(total_loss.item()), priorities

    # ==================================================================
    # Saving
    # ==================================================================

    def save(self, path: str | Path) -> None:
        """
        Save all networks, target networks, optimizer and training state.
        """
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

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

    # ==================================================================
    # Loading
    # ==================================================================

    def load(
        self,
        path: str | Path,
        load_optimizer: bool = False,
    ) -> None:
        """
        Load a checkpoint.

        Supports checkpoints containing target networks and also older
        checkpoints containing only online networks.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        payload = torch.load(
            path,
            map_location=self.device,
        )

        # --------------------------------------------------------------
        # Online networks
        # --------------------------------------------------------------
        if "unit" not in payload:
            raise KeyError(
                "Checkpoint does not contain 'unit' network"
            )

        if "market" not in payload:
            raise KeyError(
                "Checkpoint does not contain 'market' network"
            )

        self.unit.load_state_dict(
            payload["unit"]
        )

        self.market.load_state_dict(
            payload["market"]
        )

        # --------------------------------------------------------------
        # Target unit network
        # --------------------------------------------------------------
        if "unit_target" in payload:
            self.unit_target.load_state_dict(
                payload["unit_target"]
            )
        else:
            # Backward compatibility.
            self.unit_target.load_state_dict(
                payload["unit"]
            )

        # --------------------------------------------------------------
        # Target market network
        # --------------------------------------------------------------
        if "market_target" in payload:
            self.market_target.load_state_dict(
                payload["market_target"]
            )
        else:
            # Backward compatibility.
            self.market_target.load_state_dict(
                payload["market"]
            )

        # --------------------------------------------------------------
        # Optimizer
        # --------------------------------------------------------------
        if load_optimizer and "optimizer" in payload:
            try:
                self.optimizer.load_state_dict(
                    payload["optimizer"]
                )
            except (RuntimeError, ValueError) as exc:
                # A checkpoint created with a different optimizer
                # configuration should not prevent model inference.
                print(
                    "Warning: could not load optimizer state:",
                    exc,
                )

        # --------------------------------------------------------------
        # Training step counter
        # --------------------------------------------------------------
        self.steps = int(
            payload.get("steps", 0)
        )