from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from animation import JOINT_INDEX, JOINT_NAMES, Pose

PROMPT_TOKENS = {
    "idle": 0,
    "walk": 1,
    "wave": 2,
    "dance": 3,
    "point": 4,
}


@dataclass
class MotionCommand:
    prompt: str
    intensity: float = 1.0


@dataclass
class MotionAISample:
    prompt: str
    intensity: float
    pose: Pose
    sequence_blend: float
    locomotion_delta: np.ndarray


class MotionPromptTransformer(nn.Module):
    """Small autoregressive transformer used as a latent motion prior."""

    def __init__(self, latent_dim: int = 64, d_model: int = 64, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.command_embed = nn.Embedding(len(PROMPT_TOKENS), d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.out = nn.Linear(d_model, latent_dim)

    def forward(self, command_ids: torch.Tensor, phase: torch.Tensor, prev_latent: torch.Tensor) -> torch.Tensor:
        cmd = self.command_embed(command_ids)
        tvec = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
        tfeat = self.time_mlp(tvec)
        lfeat = self.latent_mlp(prev_latent)

        x = (cmd + tfeat + lfeat).unsqueeze(1)
        x = self.encoder(x)
        return torch.tanh(self.out(x[:, 0, :]))


class MotionAIController:
    """
    Prompt-driven motion generator that controls only skeleton joint poses.

    Text input format examples:
    - walk
    - dance:0.9
    - point 0.6
    """

    def __init__(self, fps: float = 60.0, seed: int = 23, sequence_len: int = 72):
        self.fps = float(max(24.0, fps))
        self.sequence_len = int(max(12, sequence_len))
        self.latent_dim = 64

        torch.manual_seed(seed)
        np.random.seed(seed)

        self._device = torch.device("cpu")
        self.model = MotionPromptTransformer(latent_dim=self.latent_dim).to(self._device)
        self.model.eval()

        self.motion_queue: deque[MotionCommand] = deque()
        self.current_prompt = "idle"
        self.current_intensity = 1.0

        self.sequence: list[Pose] = []
        self.sequence_cursor = 0
        self.sequence_timer = 0.0

        self.latent_state = np.zeros((self.latent_dim,), dtype=np.float32)
        self.prev_euler = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)
        self.prev_root = np.zeros((3,), dtype=np.float32)

        self.transition_blend = 1.0
        self.command_age = 0.0

    @staticmethod
    def _clamp_intensity(x: float) -> float:
        return float(np.clip(x, 0.1, 2.0))

    def enqueue_prompt(self, text_command: str, intensity: float | None = None) -> MotionCommand:
        prompt, parsed_intensity = self._parse_command(text_command)
        final_intensity = self._clamp_intensity(parsed_intensity if intensity is None else intensity)
        cmd = MotionCommand(prompt=prompt, intensity=final_intensity)
        self.motion_queue.append(cmd)
        return cmd

    def _parse_command(self, raw: str) -> tuple[str, float]:
        text = (raw or "").strip().lower()
        if not text:
            return "idle", 1.0

        intensity = 1.0
        if ":" in text:
            p, maybe = text.split(":", 1)
            text = p.strip()
            try:
                intensity = float(maybe.strip())
            except ValueError:
                intensity = 1.0
        else:
            parts = text.split()
            if len(parts) >= 2:
                text = parts[0]
                try:
                    intensity = float(parts[1])
                except ValueError:
                    intensity = 1.0

        if text not in PROMPT_TOKENS:
            text = "idle"

        return text, self._clamp_intensity(intensity)

    def _rot_x(self, theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)

    def _rot_y(self, theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)

    def _rot_z(self, theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    def _joint_rot_from_euler_xyz(self, xyz: np.ndarray) -> np.ndarray:
        return self._rot_z(float(xyz[2])) @ self._rot_y(float(xyz[1])) @ self._rot_x(float(xyz[0]))

    def _sample_latent(self, prompt: str, phase: float) -> np.ndarray:
        cmd = torch.tensor([PROMPT_TOKENS[prompt]], dtype=torch.long, device=self._device)
        p = torch.tensor([phase], dtype=torch.float32, device=self._device)
        prev = torch.from_numpy(self.latent_state[None, :]).to(self._device)

        with torch.no_grad():
            latent = self.model(command_ids=cmd, phase=p, prev_latent=prev)[0].cpu().numpy().astype(np.float32)

        self.latent_state = (0.85 * self.latent_state + 0.15 * latent).astype(np.float32)
        return self.latent_state

    def _base_motion_euler(self, prompt: str, t: float, intensity: float) -> tuple[np.ndarray, np.ndarray]:
        euler = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)
        root = np.zeros((3,), dtype=np.float32)

        if prompt == "idle":
            breath = np.sin(2.0 * np.pi * 0.28 * t)
            sway = np.sin(2.0 * np.pi * 0.19 * t + 0.55)
            # Idle stance bias: slight knee bend, open shoulders, and soft S-curve.
            euler[JOINT_INDEX["pelvis"], 2] = -0.03
            euler[JOINT_INDEX["spine"], 0] = 0.06 * breath * intensity
            euler[JOINT_INDEX["chest"], 0] = 0.10 * breath * intensity
            euler[JOINT_INDEX["head"], 0] = 0.07 * breath * intensity
            euler[JOINT_INDEX["chest"], 2] = 0.045 * sway * intensity
            euler[JOINT_INDEX["spine"], 2] = -0.030 + 0.020 * sway * intensity
            euler[JOINT_INDEX["shoulder_l"], 0] = -0.12
            euler[JOINT_INDEX["shoulder_r"], 0] = -0.12
            euler[JOINT_INDEX["shoulder_l"], 2] = 0.18
            euler[JOINT_INDEX["shoulder_r"], 2] = -0.18
            euler[JOINT_INDEX["elbow_l"], 0] = 0.18
            euler[JOINT_INDEX["elbow_r"], 0] = 0.18
            euler[JOINT_INDEX["knee_l"], 0] = 0.18
            euler[JOINT_INDEX["knee_r"], 0] = 0.18
            euler[JOINT_INDEX["foot_l"], 0] = -0.08
            euler[JOINT_INDEX["foot_r"], 0] = -0.08
            root[1] = 0.020 * breath
            root[0] = 0.010 * sway
            return euler, root

        if prompt == "walk":
            w = 2.0 * np.pi * (1.25 + 0.35 * intensity)
            s = np.sin(w * t)
            s2 = np.sin(2.0 * w * t + 0.4)
            euler[JOINT_INDEX["shoulder_l"], 0] = 0.95 * s * intensity
            euler[JOINT_INDEX["shoulder_r"], 0] = -0.95 * s * intensity
            euler[JOINT_INDEX["hip_l"], 0] = -1.02 * s * intensity
            euler[JOINT_INDEX["hip_r"], 0] = 1.02 * s * intensity
            euler[JOINT_INDEX["knee_l"], 0] = 0.28 + 1.15 * max(0.0, s) * intensity
            euler[JOINT_INDEX["knee_r"], 0] = 0.28 + 1.15 * max(0.0, -s) * intensity
            euler[JOINT_INDEX["chest"], 2] = -0.10 * s
            euler[JOINT_INDEX["spine"], 2] = 0.09 * s
            root[1] = 0.032 * s2
            root[2] = 0.100 * np.sin(w * t + np.pi / 2.0) * intensity
            return euler, root

        if prompt == "wave":
            a = np.sin(2.0 * np.pi * (2.0 + 0.5 * intensity) * t)
            b = np.sin(2.0 * np.pi * (4.0 + 0.5 * intensity) * t + 0.3)
            euler[JOINT_INDEX["chest"], 2] = 0.10
            euler[JOINT_INDEX["neck"], 1] = 0.12
            euler[JOINT_INDEX["head"], 1] = 0.14
            euler[JOINT_INDEX["shoulder_r"], 2] = -1.00
            euler[JOINT_INDEX["shoulder_r"], 0] = -0.30
            euler[JOINT_INDEX["elbow_r"], 0] = 1.15 + 0.22 * a * intensity
            euler[JOINT_INDEX["hand_r"], 1] = 0.90 * a * intensity
            euler[JOINT_INDEX["hand_r"], 2] = 0.22 * b
            root[1] = 0.008 * np.sin(2.0 * np.pi * 0.35 * t)
            return euler, root

        if prompt == "dance":
            w = 2.0 * np.pi * (1.6 + 0.9 * intensity)
            s = np.sin(w * t)
            c = np.cos(w * t)
            euler[JOINT_INDEX["pelvis"], 1] = 0.42 * s
            euler[JOINT_INDEX["chest"], 2] = 0.34 * c
            euler[JOINT_INDEX["head"], 1] = 0.30 * s
            euler[JOINT_INDEX["shoulder_l"], 0] = 1.00 * s
            euler[JOINT_INDEX["shoulder_r"], 0] = -1.00 * s
            euler[JOINT_INDEX["hip_l"], 0] = -0.68 * c
            euler[JOINT_INDEX["hip_r"], 0] = 0.68 * c
            euler[JOINT_INDEX["knee_l"], 0] = 0.30 + 0.65 * max(0.0, s)
            euler[JOINT_INDEX["knee_r"], 0] = 0.30 + 0.65 * max(0.0, -s)
            root[1] = 0.050 * np.sin(2.0 * w * t)
            root[2] = 0.048 * np.sin(w * t + 0.4)
            return euler, root

        # point
        euler[JOINT_INDEX["chest"], 1] = 0.12 * intensity
        euler[JOINT_INDEX["neck"], 1] = 0.15 * intensity
        euler[JOINT_INDEX["head"], 1] = 0.18 * intensity
        euler[JOINT_INDEX["shoulder_r"], 2] = -0.42
        euler[JOINT_INDEX["shoulder_r"], 1] = -0.15
        euler[JOINT_INDEX["elbow_r"], 0] = 0.28
        euler[JOINT_INDEX["hand_r"], 1] = -0.32
        root[1] = 0.004 * np.sin(2.0 * np.pi * 0.55 * t)
        return euler, root

    def _latent_to_pose(self, prompt: str, t: float, latent: np.ndarray, intensity: float) -> Pose:
        euler, root = self._base_motion_euler(prompt=prompt, t=t, intensity=intensity)

        # Latent channels modulate specific regions to add non-repeating micro-variation.
        l = latent
        amp = 0.30 * intensity
        euler[JOINT_INDEX["spine"], 2] += amp * l[0] * 0.18
        euler[JOINT_INDEX["chest"], 1] += amp * l[1] * 0.22
        euler[JOINT_INDEX["head"], 0] += amp * l[2] * 0.18
        euler[JOINT_INDEX["shoulder_l"], 0] += amp * l[3] * 0.35
        euler[JOINT_INDEX["shoulder_r"], 0] -= amp * l[4] * 0.35
        euler[JOINT_INDEX["elbow_l"], 0] += amp * l[5] * 0.25
        euler[JOINT_INDEX["elbow_r"], 0] += amp * l[6] * 0.25
        euler[JOINT_INDEX["hip_l"], 0] += amp * l[7] * 0.30
        euler[JOINT_INDEX["hip_r"], 0] -= amp * l[8] * 0.30
        euler[JOINT_INDEX["knee_l"], 0] += amp * l[9] * 0.22
        euler[JOINT_INDEX["knee_r"], 0] += amp * l[10] * 0.22

        root += np.array([0.0, 0.016 * l[11], 0.020 * l[12] * intensity], dtype=np.float32)

        # Movement smoothing in latent-decoded pose space.
        smooth = 0.24
        euler = ((1.0 - smooth) * self.prev_euler + smooth * euler).astype(np.float32)
        root = ((1.0 - smooth) * self.prev_root + smooth * root).astype(np.float32)
        self.prev_euler = euler
        self.prev_root = root

        local_rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], len(JOINT_NAMES), axis=0)
        for j in range(len(JOINT_NAMES)):
            local_rot[j] = self._joint_rot_from_euler_xyz(euler[j])

        return Pose(local_rotations=local_rot, root_offset=root)

    def _generate_sequence(self, prompt: str, intensity: float, command_age: float) -> list[Pose]:
        seq: list[Pose] = []
        for i in range(self.sequence_len):
            t = command_age + (i / self.fps)
            phase = np.float32(2.0 * np.pi * t)
            latent = self._sample_latent(prompt=prompt, phase=float(phase))
            seq.append(self._latent_to_pose(prompt=prompt, t=t, latent=latent, intensity=intensity))
        return seq

    def _ensure_command(self) -> None:
        if self.sequence_cursor < len(self.sequence):
            return

        if self.motion_queue:
            cmd = self.motion_queue.popleft()
            self.current_prompt = cmd.prompt
            self.current_intensity = cmd.intensity
            self.command_age = 0.0
            self.transition_blend = 0.0
        else:
            self.current_prompt = "idle"
            self.current_intensity = 1.0

        self.sequence = self._generate_sequence(
            prompt=self.current_prompt,
            intensity=self.current_intensity,
            command_age=self.command_age,
        )
        self.sequence_cursor = 0

    def update(self, dt: float, paused: bool) -> MotionAISample:
        if paused:
            pose = self.sequence[self.sequence_cursor - 1] if self.sequence_cursor > 0 else self._latent_to_pose(
                prompt="idle",
                t=self.command_age,
                latent=self.latent_state,
                intensity=1.0,
            )
            return MotionAISample(
                prompt=self.current_prompt,
                intensity=self.current_intensity,
                pose=pose,
                sequence_blend=self.transition_blend,
                locomotion_delta=np.zeros((3,), dtype=np.float32),
            )

        self._ensure_command()

        self.command_age += dt
        self.sequence_timer += dt

        frame_advance = int(self.sequence_timer * self.fps)
        if frame_advance > 0:
            self.sequence_timer -= frame_advance / self.fps
            self.sequence_cursor = min(len(self.sequence), self.sequence_cursor + frame_advance)
            if self.sequence_cursor >= len(self.sequence):
                self._ensure_command()

        frame_idx = min(self.sequence_cursor, len(self.sequence) - 1)
        pose = self.sequence[frame_idx]

        self.transition_blend = min(1.0, self.transition_blend + dt * 4.0)

        locomotion = np.zeros((3,), dtype=np.float32)
        if self.current_prompt == "walk":
            locomotion[2] = -0.88 * self.current_intensity * dt
        elif self.current_prompt == "dance":
            locomotion[2] = -0.24 * self.current_intensity * dt

        return MotionAISample(
            prompt=self.current_prompt,
            intensity=self.current_intensity,
            pose=pose,
            sequence_blend=self.transition_blend,
            locomotion_delta=locomotion,
        )
