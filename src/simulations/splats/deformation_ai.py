from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from animation import JOINT_INDEX


@dataclass
class DeformationStats:
    rms: float
    max_abs: float


class DeformationMLP(nn.Module):
    """Tiny conditioning MLP for realtime per-particle additive offsets."""

    def __init__(self, context_dim: int, joint_count: int):
        super().__init__()
        self.joint_embed = nn.Embedding(joint_count, 4)
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 24),
            nn.SiLU(),
        )
        self.particle_net = nn.Sequential(
            nn.Linear(3 + 3 + 1 + 4 + 24, 40),
            nn.SiLU(),
            nn.Linear(40, 24),
            nn.SiLU(),
            nn.Linear(24, 3),
            nn.Tanh(),
        )

    def forward(
        self,
        context: torch.Tensor,
        particle_rel: torch.Tensor,
        joint_vel: torch.Tensor,
        dom_weight: torch.Tensor,
        dom_joint_idx: torch.Tensor,
    ) -> torch.Tensor:
        ctx = self.context_net(context)
        ctx = ctx.expand(particle_rel.shape[0], -1)
        emb = self.joint_embed(dom_joint_idx)
        feats = torch.cat([particle_rel, joint_vel, dom_weight, emb, ctx], dim=1)
        return self.particle_net(feats)


class NeuralDeformationController:
    """
    Additive neural deformation on top of skeletal skinning.

    Input features:
    - joint rotations (flattened matrix state)
    - joint velocities
    - pose state vector

    Output:
    - per-particle XYZ offsets, intended to be added after skinning.
    """

    def __init__(
        self,
        local_positions: np.ndarray,
        bind_world_positions: np.ndarray,
        bone_indices: np.ndarray,
        bone_weights: np.ndarray,
        rest_joint_positions: np.ndarray,
        joint_count: int,
        pose_state_dim: int,
        strength: float = 1.0,
        inference_hz: float = 30.0,
        seed: int = 37,
    ) -> None:
        self.joint_count = int(joint_count)
        self.strength = float(np.clip(strength, 0.0, 2.5))
        self.inference_hz = float(max(8.0, inference_hz))
        self.interval = 1.0 / self.inference_hz

        self.local_positions = local_positions.astype(np.float32, copy=False)
        self.bind_world_positions = bind_world_positions.astype(np.float32, copy=False)
        self.rest_joint_positions = rest_joint_positions.astype(np.float32, copy=False)

        w = bone_weights.astype(np.float32, copy=False)
        dominant_slot = np.argmax(w, axis=1)
        self.dominant_joint = bone_indices[np.arange(bone_indices.shape[0]), dominant_slot].astype(np.int64)
        self.dominant_weight = w[np.arange(w.shape[0]), dominant_slot][:, None].astype(np.float32)

        self.rest_particle_rel = (
            self.bind_world_positions - self.rest_joint_positions[self.dominant_joint]
        ).astype(np.float32)

        self.prev_joint_world = np.zeros((self.joint_count, 3), dtype=np.float32)
        self.joint_vel = np.zeros((self.joint_count, 3), dtype=np.float32)
        self.prev_offsets = np.zeros_like(self.local_positions)
        self.cached_offsets = np.zeros_like(self.local_positions)
        self.time_accum = 0.0

        self._last_stats = DeformationStats(rms=0.0, max_abs=0.0)

        self._device = torch.device("cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)

        context_dim = self.joint_count * 9 + self.joint_count * 3 + pose_state_dim
        self.model = DeformationMLP(context_dim=context_dim, joint_count=self.joint_count).to(self._device)
        self.model.eval()

        self._dom_joint_t = torch.from_numpy(self.dominant_joint).long().to(self._device)
        self._dom_weight_t = torch.from_numpy(self.dominant_weight).float().to(self._device)
        self._rest_rel_t = torch.from_numpy(self.rest_particle_rel).float().to(self._device)

    @property
    def stats(self) -> DeformationStats:
        return self._last_stats

    @staticmethod
    def _safe_normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (n + 1e-8)

    @staticmethod
    def _chain_bend(world_pos: np.ndarray, a: int, b: int, c: int) -> float:
        v1 = world_pos[a] - world_pos[b]
        v2 = world_pos[c] - world_pos[b]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-7 or n2 < 1e-7:
            return 0.0
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        return float(np.clip((np.pi - ang) / np.pi, 0.0, 1.0))

    def _pose_prior(self, world_rot: np.ndarray, world_pos: np.ndarray) -> np.ndarray:
        # Anatomical priors guide the tiny MLP toward plausible soft-tissue motion.
        prior = np.zeros_like(self.local_positions)

        rel = self.rest_particle_rel
        rel_n = self._safe_normalize(rel)

        elbow_l = self._chain_bend(
            world_pos,
            JOINT_INDEX["shoulder_l"],
            JOINT_INDEX["elbow_l"],
            JOINT_INDEX["hand_l"],
        )
        elbow_r = self._chain_bend(
            world_pos,
            JOINT_INDEX["shoulder_r"],
            JOINT_INDEX["elbow_r"],
            JOINT_INDEX["hand_r"],
        )

        left_elbow_mask = self.dominant_joint == JOINT_INDEX["elbow_l"]
        right_elbow_mask = self.dominant_joint == JOINT_INDEX["elbow_r"]

        prior[left_elbow_mask] += (-0.016 * elbow_l) * rel_n[left_elbow_mask]
        prior[right_elbow_mask] += (-0.016 * elbow_r) * rel_n[right_elbow_mask]

        shoulder_speed_l = float(np.linalg.norm(self.joint_vel[JOINT_INDEX["shoulder_l"]]))
        shoulder_speed_r = float(np.linalg.norm(self.joint_vel[JOINT_INDEX["shoulder_r"]]))
        shoulder_l_mask = self.dominant_joint == JOINT_INDEX["shoulder_l"]
        shoulder_r_mask = self.dominant_joint == JOINT_INDEX["shoulder_r"]
        prior[shoulder_l_mask, 2] += -0.010 * np.clip(shoulder_speed_l, 0.0, 2.0)
        prior[shoulder_r_mask, 2] += -0.010 * np.clip(shoulder_speed_r, 0.0, 2.0)

        spine_twist = float(world_rot[JOINT_INDEX["chest"], 0, 2] - world_rot[JOINT_INDEX["spine"], 0, 2])
        torso_mask = np.logical_or(
            self.dominant_joint == JOINT_INDEX["spine"],
            self.dominant_joint == JOINT_INDEX["chest"],
        )
        prior[torso_mask, 0] += 0.014 * spine_twist * rel[torso_mask, 2]

        neck_speed = float(np.linalg.norm(self.joint_vel[JOINT_INDEX["neck"]]))
        neck_mask = np.logical_or(
            self.dominant_joint == JOINT_INDEX["neck"],
            self.dominant_joint == JOINT_INDEX["head"],
        )
        prior[neck_mask, 1] += 0.008 * np.clip(neck_speed, 0.0, 1.8)

        return prior.astype(np.float32)

    def predict_offsets(
        self,
        world_rot: np.ndarray,
        world_pos: np.ndarray,
        pose_state: np.ndarray,
        dt: float,
        paused: bool,
    ) -> np.ndarray:
        if self.strength <= 1e-5:
            self.cached_offsets.fill(0.0)
            self.prev_offsets.fill(0.0)
            self._last_stats = DeformationStats(rms=0.0, max_abs=0.0)
            return self.cached_offsets

        dts = max(1e-4, float(dt))
        self.time_accum += dts

        if not paused:
            vel = (world_pos - self.prev_joint_world) / dts
            self.joint_vel = (0.78 * self.joint_vel + 0.22 * vel).astype(np.float32)
            self.prev_joint_world = world_pos.astype(np.float32, copy=True)

        if self.time_accum < self.interval:
            return self.cached_offsets
        self.time_accum = 0.0

        rot_flat = world_rot.astype(np.float32, copy=False).reshape(-1)
        vel_flat = self.joint_vel.astype(np.float32, copy=False).reshape(-1)
        pose_vec = pose_state.astype(np.float32, copy=False).reshape(-1)
        context = np.concatenate([rot_flat, vel_flat, pose_vec], axis=0)[None, :]

        world_vel = self.joint_vel[self.dominant_joint].astype(np.float32, copy=False)

        context_t = torch.from_numpy(context).float().to(self._device)
        world_vel_t = torch.from_numpy(world_vel).float().to(self._device)

        with torch.no_grad():
            base = self.model(
                context=context_t,
                particle_rel=self._rest_rel_t,
                joint_vel=world_vel_t,
                dom_weight=self._dom_weight_t,
                dom_joint_idx=self._dom_joint_t,
            )

        mlp_offsets = base.cpu().numpy().astype(np.float32) * 0.018
        prior_offsets = self._pose_prior(world_rot=world_rot, world_pos=world_pos)

        raw = (0.62 * mlp_offsets + 0.38 * prior_offsets) * self.strength

        alpha = 0.20
        smooth = ((1.0 - alpha) * self.prev_offsets + alpha * raw).astype(np.float32)

        max_norm = 0.045
        norms = np.linalg.norm(smooth, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_norm / (norms + 1e-8))
        smooth *= scale

        self.prev_offsets = smooth
        self.cached_offsets = smooth

        self._last_stats = DeformationStats(
            rms=float(np.sqrt(np.mean(smooth**2))),
            max_abs=float(np.max(np.abs(smooth))),
        )

        return self.cached_offsets
