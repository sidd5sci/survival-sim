from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from animation import JOINT_INDEX, JOINT_NAMES, PARENTS, Pose, REST_OFFSETS


def _rot_x(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rot_y(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _rot_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


@dataclass
class BalanceSample:
    com: np.ndarray
    support_center: np.ndarray
    body_tilt: np.ndarray
    com_error: float


class BalanceSystemController:
    """Realtime balance controller using COM/support estimation and upper-body compensation."""

    def __init__(self, strength: float = 1.0) -> None:
        self.strength = float(np.clip(strength, 0.0, 2.0))
        self.time = 0.0

        self.pelvis_comp = np.zeros((3,), dtype=np.float32)
        self.torso_corr = np.zeros((2,), dtype=np.float32)
        self.upper_stab = np.zeros((2,), dtype=np.float32)

        self._com = np.zeros((3,), dtype=np.float32)
        self._support_center = np.zeros((3,), dtype=np.float32)
        self._body_tilt = np.zeros((2,), dtype=np.float32)
        self._com_error = 0.0

        self.joint_mass = np.ones((len(JOINT_NAMES),), dtype=np.float32)
        self._setup_masses()

    @property
    def sample(self) -> BalanceSample:
        return BalanceSample(
            com=self._com.copy(),
            support_center=self._support_center.copy(),
            body_tilt=self._body_tilt.copy(),
            com_error=float(self._com_error),
        )

    def _setup_masses(self) -> None:
        self.joint_mass[JOINT_INDEX["pelvis"]] = 7.0
        self.joint_mass[JOINT_INDEX["spine"]] = 5.3
        self.joint_mass[JOINT_INDEX["chest"]] = 4.8
        self.joint_mass[JOINT_INDEX["neck"]] = 1.2
        self.joint_mass[JOINT_INDEX["head"]] = 1.9

        for side in ("l", "r"):
            self.joint_mass[JOINT_INDEX[f"shoulder_{side}"]] = 1.4
            self.joint_mass[JOINT_INDEX[f"elbow_{side}"]] = 1.1
            self.joint_mass[JOINT_INDEX[f"hand_{side}"]] = 0.8
            self.joint_mass[JOINT_INDEX[f"hip_{side}"]] = 2.5
            self.joint_mass[JOINT_INDEX[f"knee_{side}"]] = 2.2
            self.joint_mass[JOINT_INDEX[f"foot_{side}"]] = 1.8

    @staticmethod
    def _smooth(curr: np.ndarray, target: np.ndarray, rate: float, dt: float) -> np.ndarray:
        alpha = 1.0 - np.exp(-rate * dt)
        return (curr + alpha * (target - curr)).astype(np.float32)

    @staticmethod
    def _smooth_scalar(curr: float, target: float, rate: float, dt: float) -> float:
        alpha = 1.0 - np.exp(-rate * dt)
        return float(curr + alpha * (target - curr))

    def _fk(self, pose: Pose) -> np.ndarray:
        joint_count = len(JOINT_NAMES)
        world_rot = np.zeros((joint_count, 3, 3), dtype=np.float32)
        world_pos = np.zeros((joint_count, 3), dtype=np.float32)

        for j in range(joint_count):
            p = PARENTS[j]
            if p < 0:
                world_rot[j] = pose.local_rotations[j]
                world_pos[j] = REST_OFFSETS[j] + pose.root_offset
            else:
                world_rot[j] = world_rot[p] @ pose.local_rotations[j]
                world_pos[j] = world_pos[p] + world_rot[p] @ REST_OFFSETS[j]

        return world_pos

    def _estimate_com(self, world_pos: np.ndarray) -> np.ndarray:
        m = self.joint_mass[:, None]
        total = float(np.sum(m))
        return (np.sum(world_pos * m, axis=0) / max(1e-6, total)).astype(np.float32)

    def _estimate_support_polygon(self, world_pos: np.ndarray, ground_y: float) -> tuple[np.ndarray, np.ndarray]:
        foot_l = world_pos[JOINT_INDEX["foot_l"]].copy()
        foot_r = world_pos[JOINT_INDEX["foot_r"]].copy()
        foot_l[1] = ground_y
        foot_r[1] = ground_y

        # Approximate support polygon from two planted feet rectangles.
        width = 0.06
        length = 0.11

        poly = np.array(
            [
                [foot_l[0] - width, ground_y, foot_l[2] - length],
                [foot_l[0] + width, ground_y, foot_l[2] - length],
                [foot_l[0] + width, ground_y, foot_l[2] + length],
                [foot_l[0] - width, ground_y, foot_l[2] + length],
                [foot_r[0] - width, ground_y, foot_r[2] - length],
                [foot_r[0] + width, ground_y, foot_r[2] - length],
                [foot_r[0] + width, ground_y, foot_r[2] + length],
                [foot_r[0] - width, ground_y, foot_r[2] + length],
            ],
            dtype=np.float32,
        )
        center = np.mean(poly, axis=0).astype(np.float32)
        return poly, center

    def apply(
        self,
        pose: Pose,
        root_velocity: np.ndarray,
        root_acceleration: np.ndarray,
        turn_speed: float,
        dt: float,
        paused: bool,
        ground_y: float = 0.0,
    ) -> Pose:
        if self.strength <= 1e-5 or paused:
            return pose

        dts = max(1e-4, float(dt))
        self.time += dts

        out = Pose(local_rotations=pose.local_rotations.copy(), root_offset=pose.root_offset.copy())
        world_pos = self._fk(out)

        com = self._estimate_com(world_pos)
        _, support_center = self._estimate_support_polygon(world_pos, ground_y=ground_y)

        error = com - support_center
        error_xz = np.array([error[0], error[2]], dtype=np.float32)
        err_mag = float(np.linalg.norm(error_xz))

        root_speed = float(np.linalg.norm(root_velocity[[0, 2]]))
        lateral_vel = float(root_velocity[0])
        forward_vel = float(root_velocity[2])
        lateral_accel = float(root_acceleration[0])
        forward_accel = float(root_acceleration[2])

        # Movement lean + turning balance.
        lean_target = np.array(
            [
                -0.05 * forward_accel - 0.020 * forward_vel,
                -0.05 * lateral_accel - 0.018 * lateral_vel - 0.03 * turn_speed,
            ],
            dtype=np.float32,
        )

        # Pelvis compensation keeps COM over support center.
        pelvis_target = np.array(
            [
                -0.22 * error[0] + 0.04 * lateral_vel,
                0.0,
                -0.25 * error[2] + 0.05 * forward_vel,
            ],
            dtype=np.float32,
        )

        # Balance recovery boost when COM drifts too far.
        if err_mag > 0.09:
            recover = min(0.20, (err_mag - 0.09) * 0.55)
            pelvis_target[0] += -recover * np.sign(error[0])
            pelvis_target[2] += -recover * np.sign(error[2])

        # Idle sway when almost stationary.
        idle_sway = np.zeros((2,), dtype=np.float32)
        if root_speed < 0.12:
            idle_sway[0] = 0.010 * np.sin(2.0 * np.pi * 0.45 * self.time)
            idle_sway[1] = 0.013 * np.sin(2.0 * np.pi * 0.31 * self.time + 0.7)

        torso_target = np.array(
            [
                0.34 * (lean_target[0] + idle_sway[0]),
                0.34 * (lean_target[1] + idle_sway[1]),
            ],
            dtype=np.float32,
        )

        upper_target = np.array(
            [
                -0.58 * torso_target[0],
                -0.62 * torso_target[1],
            ],
            dtype=np.float32,
        )

        self.pelvis_comp = self._smooth(self.pelvis_comp, pelvis_target, rate=6.5, dt=dts)
        self.torso_corr = self._smooth(self.torso_corr, torso_target, rate=7.2, dt=dts)
        self.upper_stab = self._smooth(self.upper_stab, upper_target, rate=8.5, dt=dts)

        self.pelvis_comp = np.clip(self.pelvis_comp, -0.12, 0.12)
        self.torso_corr = np.clip(self.torso_corr, -0.24, 0.24)
        self.upper_stab = np.clip(self.upper_stab, -0.22, 0.22)

        gain = self.strength

        # Apply pelvis compensation.
        out.root_offset[0] += gain * self.pelvis_comp[0]
        out.root_offset[2] += gain * self.pelvis_comp[2]

        spine = JOINT_INDEX["spine"]
        chest = JOINT_INDEX["chest"]
        neck = JOINT_INDEX["neck"]
        head = JOINT_INDEX["head"]

        # Torso correction and balance lean.
        out.local_rotations[spine] = _rot_x(gain * self.torso_corr[0] * 0.60) @ _rot_z(gain * self.torso_corr[1] * 0.50) @ out.local_rotations[spine]
        out.local_rotations[chest] = _rot_x(gain * self.torso_corr[0]) @ _rot_z(gain * self.torso_corr[1]) @ _rot_y(gain * 0.06 * turn_speed) @ out.local_rotations[chest]

        # Upper body stabilization counters torso motion for believable balance.
        out.local_rotations[neck] = _rot_x(gain * self.upper_stab[0] * 0.75) @ _rot_y(gain * self.upper_stab[1] * 0.85) @ out.local_rotations[neck]
        out.local_rotations[head] = _rot_x(gain * self.upper_stab[0]) @ _rot_y(gain * self.upper_stab[1]) @ out.local_rotations[head]

        self._com = com
        self._support_center = support_center
        self._body_tilt = torso_target
        self._com_error = err_mag

        return out
