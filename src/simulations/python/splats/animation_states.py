from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pygame

from animation import JOINT_INDEX, JOINT_NAMES, MotionParams, Pose


class AnimationState(str, Enum):
    IDLE = "IDLE"
    WALK = "WALK"
    RUN = "RUN"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    WAVE = "WAVE"


@dataclass
class AnimationSample:
    state: AnimationState
    motion: MotionParams
    blend_factor: float
    anim_time: float
    layer_pose: Pose | None
    layer_blend: float
    layer_joint_indices: np.ndarray | None


class AnimationController:
    """State-machine-based controller with smooth transitions and layered upper-body wave."""

    def __init__(self) -> None:
        self.current_state = AnimationState.IDLE
        self.prev_state = AnimationState.IDLE
        self.transition_duration = 0.22
        self.transition_elapsed = 1.0

        self.current_time = 0.0
        self.prev_time = 0.0

        self.animation_speed = 1.0

        self.wave_phase = 0.0
        self.wave_blend = 0.0

        self.upper_body_joint_indices = np.array(
            [
                JOINT_INDEX["chest"],
                JOINT_INDEX["neck"],
                JOINT_INDEX["head"],
                JOINT_INDEX["shoulder_l"],
                JOINT_INDEX["elbow_l"],
                JOINT_INDEX["hand_l"],
                JOINT_INDEX["shoulder_r"],
                JOINT_INDEX["elbow_r"],
                JOINT_INDEX["hand_r"],
            ],
            dtype=np.int32,
        )

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _approach(current: float, target: float, rate: float, dt: float) -> float:
        step = rate * dt
        if current < target:
            return min(current + step, target)
        return max(current - step, target)

    @staticmethod
    def _lerp_motion(a: MotionParams, b: MotionParams, t: float) -> MotionParams:
        return MotionParams(
            freq_scale=(1.0 - t) * a.freq_scale + t * b.freq_scale,
            arm_amp=(1.0 - t) * a.arm_amp + t * b.arm_amp,
            leg_amp=(1.0 - t) * a.leg_amp + t * b.leg_amp,
            breath_amp=(1.0 - t) * a.breath_amp + t * b.breath_amp,
            torso_twist=(1.0 - t) * a.torso_twist + t * b.torso_twist,
            bob_amp=(1.0 - t) * a.bob_amp + t * b.bob_amp,
            forward_amp=(1.0 - t) * a.forward_amp + t * b.forward_amp,
            head_nod_amp=(1.0 - t) * a.head_nod_amp + t * b.head_nod_amp,
            turn_bias=(1.0 - t) * a.turn_bias + t * b.turn_bias,
        )

    def _state_motion(self, state: AnimationState, move_x: float, move_y: float, running: bool) -> MotionParams:
        speed_mag = min(1.0, np.hypot(move_x, move_y))
        direction = 1.0 if move_y >= 0.0 else -1.0

        if state == AnimationState.IDLE:
            return MotionParams(
                freq_scale=0.55,
                arm_amp=0.15,
                leg_amp=0.08,
                breath_amp=1.15,
                torso_twist=0.35,
                bob_amp=0.25,
                forward_amp=0.05,
                head_nod_amp=0.25,
                turn_bias=0.0,
            )

        if state == AnimationState.WALK:
            g = direction * (0.55 + 0.85 * speed_mag)
            return MotionParams(
                freq_scale=0.95 + 0.45 * speed_mag,
                arm_amp=g,
                leg_amp=g,
                breath_amp=0.85,
                torso_twist=0.90,
                bob_amp=0.75,
                forward_amp=g,
                head_nod_amp=0.65,
                turn_bias=0.25 * move_x,
            )

        if state == AnimationState.RUN:
            g = direction * (1.35 + 0.85 * speed_mag)
            return MotionParams(
                freq_scale=1.65 + 0.55 * speed_mag,
                arm_amp=g,
                leg_amp=g,
                breath_amp=0.55,
                torso_twist=1.45,
                bob_amp=1.35,
                forward_amp=g,
                head_nod_amp=0.95,
                turn_bias=0.35 * move_x,
            )

        if state == AnimationState.TURN_LEFT:
            return MotionParams(
                freq_scale=0.82,
                arm_amp=0.32,
                leg_amp=0.30,
                breath_amp=1.00,
                torso_twist=1.25,
                bob_amp=0.40,
                forward_amp=0.15,
                head_nod_amp=0.50,
                turn_bias=-0.95,
            )

        if state == AnimationState.TURN_RIGHT:
            return MotionParams(
                freq_scale=0.82,
                arm_amp=0.32,
                leg_amp=0.30,
                breath_amp=1.00,
                torso_twist=1.25,
                bob_amp=0.40,
                forward_amp=0.15,
                head_nod_amp=0.50,
                turn_bias=0.95,
            )

        # WAVE base locomotion behaves like idle while upper body is layered.
        return self._state_motion(AnimationState.IDLE, move_x, move_y, running)

    def _select_state(self, move_x: float, move_y: float, running: bool) -> AnimationState:
        move_mag = np.hypot(move_x, move_y)
        if move_mag < 1e-3:
            if move_x < -0.35:
                return AnimationState.TURN_LEFT
            if move_x > 0.35:
                return AnimationState.TURN_RIGHT
            return AnimationState.IDLE

        if running:
            return AnimationState.RUN
        return AnimationState.WALK

    def _wave_layer_pose(self, t: float) -> Pose:
        local_rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], len(JOINT_NAMES), axis=0)

        # Right-hand wave with subtle chest/neck counter motion.
        a = np.sin(2.0 * np.pi * 2.1 * t)
        b = np.sin(2.0 * np.pi * 4.2 * t + 0.4)

        def rot_x(theta: float) -> np.ndarray:
            c = np.cos(theta)
            s = np.sin(theta)
            return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)

        def rot_y(theta: float) -> np.ndarray:
            c = np.cos(theta)
            s = np.sin(theta)
            return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)

        def rot_z(theta: float) -> np.ndarray:
            c = np.cos(theta)
            s = np.sin(theta)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        local_rot[JOINT_INDEX["chest"]] = rot_z(0.08)
        local_rot[JOINT_INDEX["neck"]] = rot_y(0.10)
        local_rot[JOINT_INDEX["head"]] = rot_y(0.16)

        local_rot[JOINT_INDEX["shoulder_r"]] = rot_z(-0.95) @ rot_x(-0.35)
        local_rot[JOINT_INDEX["elbow_r"]] = rot_x(1.20 + 0.20 * a)
        local_rot[JOINT_INDEX["hand_r"]] = rot_y(0.20 + 0.85 * a) @ rot_z(0.20 * b)

        # Keep left arm quieter for asymmetry.
        local_rot[JOINT_INDEX["shoulder_l"]] = rot_x(0.18)
        local_rot[JOINT_INDEX["elbow_l"]] = rot_x(0.30)

        return Pose(local_rotations=local_rot, root_offset=np.zeros(3, dtype=np.float32))

    def update(self, dt: float, keys: pygame.key.ScancodeWrapper) -> AnimationSample:
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            self.animation_speed = min(2.5, self.animation_speed + dt * 0.8)
        if keys[pygame.K_MINUS] or keys[pygame.K_UNDERSCORE]:
            self.animation_speed = max(0.35, self.animation_speed - dt * 0.8)

        move_x = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
        move_y = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
        running = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
        waving = bool(keys[pygame.K_SPACE])

        target_state = self._select_state(move_x=move_x, move_y=move_y, running=running)
        if target_state != self.current_state:
            self.prev_state = self.current_state
            self.current_state = target_state
            self.prev_time = self.current_time
            self.current_time = 0.0
            self.transition_elapsed = 0.0
            self.transition_duration = 0.28 if target_state in (AnimationState.RUN, AnimationState.WALK) else 0.22

        self.transition_elapsed = min(self.transition_duration, self.transition_elapsed + dt)
        self.current_time += dt * self.animation_speed

        t_raw = 1.0 if self.transition_duration <= 1e-6 else self.transition_elapsed / self.transition_duration
        blend = self._smoothstep(t_raw)

        prev_motion = self._state_motion(self.prev_state, move_x=move_x, move_y=move_y, running=running)
        curr_motion = self._state_motion(self.current_state, move_x=move_x, move_y=move_y, running=running)
        motion = self._lerp_motion(prev_motion, curr_motion, blend)

        wave_target = 1.0 if waving else 0.0
        self.wave_blend = self._approach(self.wave_blend, wave_target, rate=4.0, dt=dt)
        if self.wave_blend > 1e-4:
            self.wave_phase += dt * self.animation_speed

        layer_pose = self._wave_layer_pose(self.wave_phase) if self.wave_blend > 1e-4 else None

        display_state = AnimationState.WAVE if self.wave_blend > 0.2 else self.current_state

        return AnimationSample(
            state=display_state,
            motion=motion,
            blend_factor=blend,
            anim_time=self.current_time,
            layer_pose=layer_pose,
            layer_blend=self.wave_blend,
            layer_joint_indices=self.upper_body_joint_indices if layer_pose is not None else None,
        )
