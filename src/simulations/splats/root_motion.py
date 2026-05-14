from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class MovementState(str, Enum):
    IDLE = "IDLE"
    WALK = "WALK"
    RUN = "RUN"


@dataclass
class RootMotionSample:
    delta_position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    state: MovementState
    state_blend: float
    heading_yaw: float
    turn_speed: float


class RootMotionController:
    """Momentum-based root movement with inertial turning and state transitions."""

    def __init__(
        self,
        walk_speed: float = 0.95,
        run_speed: float = 1.85,
        inertia: float = 1.45,
        friction: float = 2.20,
        accel_gain: float = 6.8,
        turn_inertia: float = 2.8,
        turn_damping: float = 3.4,
        dir_smoothing: float = 7.5,
    ) -> None:
        self.walk_speed = float(max(0.05, walk_speed))
        self.run_speed = float(max(self.walk_speed, run_speed))

        self.inertia = float(max(0.05, inertia))
        self.friction = float(max(0.0, friction))
        self.accel_gain = float(max(0.0, accel_gain))

        self.turn_inertia = float(max(0.05, turn_inertia))
        self.turn_damping = float(max(0.0, turn_damping))
        self.dir_smoothing = float(max(0.1, dir_smoothing))

        self.position = np.zeros((3,), dtype=np.float32)
        self.velocity = np.zeros((3,), dtype=np.float32)
        self.acceleration = np.zeros((3,), dtype=np.float32)

        self.heading_yaw = 0.0
        self.turn_speed = 0.0
        self._desired_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        self.state = MovementState.IDLE
        self.prev_state = MovementState.IDLE
        self.state_time = 0.0
        self.state_transition = 0.0
        self.state_transition_duration = 0.28

    @staticmethod
    def _smoothstep(x: float) -> float:
        t = float(np.clip(x, 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = b - a
        while d > np.pi:
            d -= 2.0 * np.pi
        while d < -np.pi:
            d += 2.0 * np.pi
        return float(d)

    @staticmethod
    def _safe_dir(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return (v / n).astype(np.float32)

    def _update_state(self, speed_target: float, dt: float) -> None:
        self.state_time += dt

        if speed_target < 0.08:
            target_state = MovementState.IDLE
        elif speed_target < 0.62 * self.run_speed:
            target_state = MovementState.WALK
        else:
            target_state = MovementState.RUN

        if target_state != self.state:
            self.prev_state = self.state
            self.state = target_state
            self.state_time = 0.0
            self.state_transition = 0.0
            if self.prev_state == MovementState.IDLE or self.state == MovementState.IDLE:
                self.state_transition_duration = 0.34
            else:
                self.state_transition_duration = 0.24

        self.state_transition = min(self.state_transition_duration, self.state_transition + dt)

    def _state_speed_scale(self, desired_speed: float) -> float:
        if self.state == MovementState.IDLE:
            return 0.0
        if self.state == MovementState.WALK:
            return min(desired_speed, self.walk_speed)
        return min(desired_speed, self.run_speed)

    def update(self, desired_velocity: np.ndarray, dt: float, paused: bool) -> RootMotionSample:
        dts = max(1e-4, float(dt))
        if paused:
            return RootMotionSample(
                delta_position=np.zeros((3,), dtype=np.float32),
                velocity=self.velocity.copy(),
                acceleration=self.acceleration.copy(),
                state=self.state,
                state_blend=1.0,
                heading_yaw=float(self.heading_yaw),
                turn_speed=float(self.turn_speed),
            )

        desired_v = desired_velocity.astype(np.float32, copy=False)
        desired_speed = float(np.linalg.norm(desired_v[[0, 2]]))

        self._update_state(speed_target=desired_speed, dt=dts)

        if desired_speed > 1e-5:
            target_dir = self._safe_dir(np.array([desired_v[0], 0.0, desired_v[2]], dtype=np.float32))
            alpha_dir = 1.0 - np.exp(-self.dir_smoothing * dts)
            self._desired_dir = ((1.0 - alpha_dir) * self._desired_dir + alpha_dir * target_dir).astype(np.float32)
            self._desired_dir = self._safe_dir(self._desired_dir)

        speed_scale = self._state_speed_scale(desired_speed)

        # Movement state transition blending for idle<->walk/run smoothing.
        trans_t = 1.0 if self.state_transition_duration <= 1e-6 else self.state_transition / self.state_transition_duration
        state_blend = self._smoothstep(trans_t)

        prev_scale = 0.0
        if self.prev_state == MovementState.WALK:
            prev_scale = min(desired_speed, self.walk_speed)
        elif self.prev_state == MovementState.RUN:
            prev_scale = min(desired_speed, self.run_speed)

        blended_speed = (1.0 - state_blend) * prev_scale + state_blend * speed_scale
        target_vel = (self._desired_dir * blended_speed).astype(np.float32)

        # Inertial acceleration with friction damping.
        force = self.accel_gain * (target_vel - self.velocity) - self.friction * self.velocity
        accel_new = force / self.inertia
        alpha_acc = 1.0 - np.exp(-7.0 * dts)
        self.acceleration = ((1.0 - alpha_acc) * self.acceleration + alpha_acc * accel_new).astype(np.float32)

        self.velocity = (self.velocity + self.acceleration * dts).astype(np.float32)

        # Turn inertia based on desired direction changes.
        desired_yaw = float(np.arctan2(self._desired_dir[0], -self._desired_dir[2]))
        yaw_error = self._angle_diff(self.heading_yaw, desired_yaw)
        turn_accel = (9.5 * yaw_error - self.turn_damping * self.turn_speed) / self.turn_inertia
        self.turn_speed = float(self.turn_speed + turn_accel * dts)

        # Directional smoothing: reduce snap when reversing hard.
        self.turn_speed *= float(np.exp(-1.25 * dts))
        self.heading_yaw = float(self.heading_yaw + self.turn_speed * dts)

        delta = (self.velocity * dts).astype(np.float32)
        self.position += delta

        return RootMotionSample(
            delta_position=delta,
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            state=self.state,
            state_blend=float(state_blend),
            heading_yaw=float(self.heading_yaw),
            turn_speed=float(self.turn_speed),
        )
