from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame


@dataclass
class MotionParams:
    """Neural-network-derived motion parameters fed into the skeletal animator."""

    freq_scale: float = 1.0
    arm_amp: float = 1.0
    leg_amp: float = 1.0
    breath_amp: float = 1.0
    torso_twist: float = 1.0
    bob_amp: float = 1.0
    forward_amp: float = 1.0
    head_nod_amp: float = 1.0
    turn_bias: float = 0.0


class TinyMotionNet:
    """Small MLP implemented in NumPy for realtime control-rate inference."""

    def __init__(self, seed: int = 123):
        rng = np.random.default_rng(seed)
        self.w1 = (rng.normal(0.0, 0.8, size=(9, 24))).astype(np.float32)
        self.b1 = (rng.normal(0.0, 0.15, size=(24,))).astype(np.float32)
        self.w2 = (rng.normal(0.0, 0.7, size=(24, 16))).astype(np.float32)
        self.b2 = (rng.normal(0.0, 0.12, size=(16,))).astype(np.float32)
        self.w3 = (rng.normal(0.0, 0.6, size=(16, 8))).astype(np.float32)
        self.b3 = np.zeros((8,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        return np.tanh(h2 @ self.w3 + self.b3)


class NeuralMotionController:
    """
    Drives gait style through a tiny neural net.

    Keys:
    - I/K: increase/decrease speed
    - J/L: left/right turn bias
    - U/O: decrease/increase arm swing style
    - N/M: decrease/increase stride length
    - B/V: decrease/increase vertical bounce
    - R: reset controls
    """

    def __init__(self) -> None:
        self.net = TinyMotionNet(seed=11)
        self.phase = 0.0

        self.speed = 0.55
        self.turn = 0.0
        self.arm_style = 0.0
        self.stride = 0.45
        self.bounce = 0.35

    @staticmethod
    def _approach(x: float, target: float, rate: float, dt: float) -> float:
        delta = target - x
        max_step = rate * dt
        if delta > max_step:
            return x + max_step
        if delta < -max_step:
            return x - max_step
        return target

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.speed = 0.55
            self.turn = 0.0
            self.arm_style = 0.0
            self.stride = 0.45
            self.bounce = 0.35

    def update(self, dt: float, keys: pygame.key.ScancodeWrapper, paused: bool) -> MotionParams:
        # Continuous keyboard control over high-level motion intent.
        speed_target = self.speed + (0.6 if keys[pygame.K_i] else 0.0) - (0.6 if keys[pygame.K_k] else 0.0)
        turn_target = self.turn + (0.8 if keys[pygame.K_l] else 0.0) - (0.8 if keys[pygame.K_j] else 0.0)
        arm_target = self.arm_style + (0.75 if keys[pygame.K_o] else 0.0) - (0.75 if keys[pygame.K_u] else 0.0)
        stride_target = self.stride + (0.75 if keys[pygame.K_m] else 0.0) - (0.75 if keys[pygame.K_n] else 0.0)
        bounce_target = self.bounce + (0.75 if keys[pygame.K_v] else 0.0) - (0.75 if keys[pygame.K_b] else 0.0)

        self.speed = float(np.clip(self._approach(self.speed, speed_target, 1.6, dt), 0.0, 1.2))
        self.turn = float(np.clip(self._approach(self.turn, turn_target, 2.2, dt), -1.0, 1.0))
        self.arm_style = float(np.clip(self._approach(self.arm_style, arm_target, 1.8, dt), -1.0, 1.0))
        self.stride = float(np.clip(self._approach(self.stride, stride_target, 1.8, dt), 0.0, 1.2))
        self.bounce = float(np.clip(self._approach(self.bounce, bounce_target, 1.8, dt), 0.0, 1.2))

        if not paused:
            self.phase += dt * (0.8 + 1.8 * self.speed)

        x = np.array(
            [
                self.speed,
                self.turn,
                self.arm_style,
                self.stride,
                self.bounce,
                np.sin(self.phase),
                np.cos(self.phase),
                self.speed * self.turn,
                self.stride * self.bounce,
            ],
            dtype=np.float32,
        )

        out = self.net.forward(x)

        return MotionParams(
            freq_scale=float(np.clip(1.00 + 0.45 * out[0] + 0.35 * self.speed, 0.45, 2.20)),
            arm_amp=float(np.clip(1.00 + 0.55 * out[1] + 0.45 * self.arm_style, 0.35, 2.10)),
            leg_amp=float(np.clip(0.95 + 0.55 * out[2] + 0.55 * self.stride, 0.30, 2.20)),
            breath_amp=float(np.clip(1.00 + 0.40 * out[3] + 0.22 * (1.0 - self.speed), 0.25, 1.80)),
            torso_twist=float(np.clip(1.00 + 0.55 * out[4] + 0.40 * abs(self.turn), 0.30, 2.20)),
            bob_amp=float(np.clip(0.90 + 0.65 * out[5] + 0.55 * self.bounce, 0.20, 2.40)),
            forward_amp=float(np.clip(0.90 + 0.60 * out[6] + 0.50 * self.stride, 0.10, 2.30)),
            head_nod_amp=float(np.clip(0.90 + 0.50 * out[7] + 0.20 * self.arm_style, 0.20, 1.90)),
            turn_bias=float(np.clip(0.65 * self.turn + 0.25 * out[4], -1.0, 1.0)),
        )
