from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from animation import JOINT_INDEX


class FootContactState(str, Enum):
    PLANTED = "PLANTED"
    RELEASING = "RELEASING"
    SWING = "SWING"


@dataclass
class FootPhaseSample:
    side: str
    state: FootContactState
    phase: float
    planted: bool


class FootPlantingController:
    """Grounded-foot controller with lock/release/swing phases to prevent sliding."""

    def __init__(self) -> None:
        self.foot_joint = {"l": JOINT_INDEX["foot_l"], "r": JOINT_INDEX["foot_r"]}

        self.state = {"l": FootContactState.SWING, "r": FootContactState.SWING}
        self.phase = {"l": 0.0, "r": 0.0}

        self.plant_pos = {
            "l": np.zeros((3,), dtype=np.float32),
            "r": np.zeros((3,), dtype=np.float32),
        }
        self.prev_pos = {
            "l": np.zeros((3,), dtype=np.float32),
            "r": np.zeros((3,), dtype=np.float32),
        }
        self.vel = {
            "l": np.zeros((3,), dtype=np.float32),
            "r": np.zeros((3,), dtype=np.float32),
        }

        self.release_alpha = {"l": 0.0, "r": 0.0}
        self.swing_time = {"l": 0.0, "r": 0.0}

        self.step_count = {"l": 0, "r": 0}
        self.just_transitioned = {"l": False, "r": False}

        self._initialized = False

    @staticmethod
    def _smoothstep(x: float) -> float:
        t = float(np.clip(x, 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def _horizontal_dist(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.sqrt(d[0] * d[0] + d[2] * d[2]))

    @staticmethod
    def _with_ground(p: np.ndarray, ground_y: float) -> np.ndarray:
        out = p.astype(np.float32, copy=True)
        out[1] = np.float32(ground_y)
        return out

    def _expected_swing_time(self, root_speed: float) -> float:
        # Faster movement gives shorter swing phase.
        return float(np.clip(0.38 - 0.14 * root_speed, 0.16, 0.40))

    def _transition(self, side: str, next_state: FootContactState) -> None:
        if self.state[side] == next_state:
            return
        self.state[side] = next_state
        self.just_transitioned[side] = True

        if next_state == FootContactState.SWING:
            self.swing_time[side] = 0.0
        elif next_state == FootContactState.RELEASING:
            self.release_alpha[side] = 0.0
        elif next_state == FootContactState.PLANTED:
            self.phase[side] = 0.0
            self.step_count[side] += 1

    def update(self, world_pos: np.ndarray, dt: float, ground_y: float, root_speed: float) -> tuple[FootPhaseSample, FootPhaseSample]:
        dts = max(1e-4, float(dt))

        if not self._initialized:
            for side in ("l", "r"):
                p = world_pos[self.foot_joint[side]].astype(np.float32)
                self.prev_pos[side] = p
                self.vel[side][:] = 0.0
                self.plant_pos[side] = self._with_ground(p, ground_y)
                self.state[side] = FootContactState.PLANTED
                self.phase[side] = 0.0
            self._initialized = True

        for side in ("l", "r"):
            self.just_transitioned[side] = False

        # 1) Kinematics and candidate transitions per foot.
        for side in ("l", "r"):
            idx = self.foot_joint[side]
            p = world_pos[idx].astype(np.float32)
            vel_raw = (p - self.prev_pos[side]) / dts
            self.vel[side] = (0.72 * self.vel[side] + 0.28 * vel_raw).astype(np.float32)
            self.prev_pos[side] = p

            speed = float(np.linalg.norm(self.vel[side]))
            near_ground = bool(p[1] <= ground_y + 0.055)
            horizontal_slip = self._horizontal_dist(p, self.plant_pos[side])

            if self.state[side] == FootContactState.PLANTED:
                self.plant_pos[side][1] = np.float32(ground_y)

                should_release = (
                    (not near_ground and speed > 0.24)
                    or horizontal_slip > 0.23
                    or (root_speed > 0.35 and speed > 0.45)
                )
                if should_release:
                    self._transition(side, FootContactState.RELEASING)

            elif self.state[side] == FootContactState.RELEASING:
                self.release_alpha[side] = min(1.0, self.release_alpha[side] + dts / 0.11)
                self.phase[side] = 1.0 - self.release_alpha[side]
                if self.release_alpha[side] >= 1.0:
                    self._transition(side, FootContactState.SWING)

            else:  # SWING
                self.swing_time[side] += dts
                expected = self._expected_swing_time(root_speed)
                self.phase[side] = float(np.clip(self.swing_time[side] / expected, 0.0, 1.0))

                contact_ready = near_ground and speed < 0.20
                other = "r" if side == "l" else "l"
                other_planted = self.state[other] == FootContactState.PLANTED

                if contact_ready and (other_planted or root_speed < 0.25):
                    self.plant_pos[side] = self._with_ground(p, ground_y)
                    self._transition(side, FootContactState.PLANTED)

        # 2) Movement synchronization safeguard: avoid both feet swinging under load.
        if root_speed > 0.2 and self.state["l"] == FootContactState.SWING and self.state["r"] == FootContactState.SWING:
            l_pos = world_pos[self.foot_joint["l"]]
            r_pos = world_pos[self.foot_joint["r"]]
            l_speed = float(np.linalg.norm(self.vel["l"]))
            r_speed = float(np.linalg.norm(self.vel["r"]))

            # Favor the lower/slower foot for immediate plant sync.
            score_l = float(l_pos[1]) + 0.12 * l_speed
            score_r = float(r_pos[1]) + 0.12 * r_speed
            keep = "l" if score_l <= score_r else "r"
            self.plant_pos[keep] = self._with_ground(world_pos[self.foot_joint[keep]], ground_y)
            self._transition(keep, FootContactState.PLANTED)

        return (
            FootPhaseSample(
                side="l",
                state=self.state["l"],
                phase=float(self.phase["l"]),
                planted=self.state["l"] == FootContactState.PLANTED,
            ),
            FootPhaseSample(
                side="r",
                state=self.state["r"],
                phase=float(self.phase["r"]),
                planted=self.state["r"] == FootContactState.PLANTED,
            ),
        )

    def target_for(self, side: str, current_foot: np.ndarray, ground_y: float) -> np.ndarray:
        curr = self._with_ground(current_foot, ground_y)

        if self.state[side] == FootContactState.PLANTED:
            t = self.plant_pos[side].copy()
            t[1] = np.float32(ground_y)
            return t.astype(np.float32)

        if self.state[side] == FootContactState.RELEASING:
            a = self._smoothstep(self.release_alpha[side])
            t = (1.0 - a) * self.plant_pos[side] + a * curr
            t[1] = np.float32(ground_y)
            return t.astype(np.float32)

        return curr.astype(np.float32)
