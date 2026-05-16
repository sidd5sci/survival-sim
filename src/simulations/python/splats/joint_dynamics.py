from __future__ import annotations

import numpy as np

from animation import JOINT_INDEX


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float32,
    )


def _exp_so3(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-6:
        return (np.eye(3, dtype=np.float32) + _skew(w)).astype(np.float32)

    k = w / theta
    kx = _skew(k)
    c = np.cos(theta)
    s = np.sin(theta)
    return (np.eye(3, dtype=np.float32) + s * kx + (1.0 - c) * (kx @ kx)).astype(np.float32)


def _log_so3(r: np.ndarray) -> np.ndarray:
    tr = float(np.trace(r))
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)

    denom = 2.0 * np.sin(theta)
    axis = np.array(
        [
            (r[2, 1] - r[1, 2]) / denom,
            (r[0, 2] - r[2, 0]) / denom,
            (r[1, 0] - r[0, 1]) / denom,
        ],
        dtype=np.float32,
    )
    return axis * theta


class JointDynamics:
    """Realtime local-joint spring-damper dynamics in SO(3)."""

    def __init__(
        self,
        joint_count: int,
        damping: float = 6.0,
        stiffness: float = 17.5,
        inertia: float = 1.0,
        max_velocity: float = 10.5,
        max_acceleration: float = 120.0,
        velocity_smoothing: float = 0.35,
        critical_damping: bool = False,
        damping_ratio: float = 1.0,
    ) -> None:
        self.joint_count = int(joint_count)
        eye = np.eye(3, dtype=np.float32)

        self.current_rotations = np.repeat(eye[None, :, :], self.joint_count, axis=0)
        self.target_rotations = np.repeat(eye[None, :, :], self.joint_count, axis=0)
        self.angular_velocity = np.zeros((self.joint_count, 3), dtype=np.float32)
        self.angular_acceleration = np.zeros((self.joint_count, 3), dtype=np.float32)

        self.base_damping = float(max(0.0, damping))
        self.base_stiffness = float(max(0.0, stiffness))
        self.base_inertia = float(max(0.01, inertia))
        self.max_velocity = float(max(0.01, max_velocity))
        self.max_acceleration = float(max(1.0, max_acceleration))
        self.velocity_smoothing = float(np.clip(velocity_smoothing, 0.0, 0.99))
        self.critical_damping = bool(critical_damping)
        self.damping_ratio = float(max(0.05, damping_ratio))

        self.joint_stiffness = np.full((self.joint_count,), self.base_stiffness, dtype=np.float32)
        self.joint_damping = np.full((self.joint_count,), self.base_damping, dtype=np.float32)
        self.joint_inertia = np.full((self.joint_count,), self.base_inertia, dtype=np.float32)

        self._configure_spring_profiles()

        self._initialized = False

    def _set_joint_profile(
        self,
        joint_name: str,
        stiffness: float | None = None,
        damping: float | None = None,
        inertia: float | None = None,
    ) -> None:
        j = JOINT_INDEX[joint_name]
        if stiffness is not None:
            self.joint_stiffness[j] = float(max(0.0, stiffness))
        if damping is not None:
            self.joint_damping[j] = float(max(0.0, damping))
        if inertia is not None:
            self.joint_inertia[j] = float(max(0.01, inertia))

    def _configure_spring_profiles(self) -> None:
        # Default profile already set from constructor; now tune key regions.
        # Spine and chest: soft follow-through with settle.
        self._set_joint_profile("spine", stiffness=14.2, damping=3.6, inertia=1.75)
        self._set_joint_profile("chest", stiffness=13.5, damping=3.3, inertia=1.85)

        # Neck/head: stabilize but still permit subtle lag.
        self._set_joint_profile("neck", stiffness=13.6, damping=5.0, inertia=1.1)
        self._set_joint_profile("head", stiffness=14.2, damping=5.2, inertia=1.05)

        # Shoulders and arms: more inertia and lower damping for overshoot/settle.
        for side in ("l", "r"):
            self._set_joint_profile(f"shoulder_{side}", stiffness=11.5, damping=2.7, inertia=2.25)
            self._set_joint_profile(f"elbow_{side}", stiffness=12.5, damping=3.1, inertia=1.95)
            self._set_joint_profile(f"hand_{side}", stiffness=13.2, damping=3.6, inertia=1.6)

    def set_config(
        self,
        damping: float | None = None,
        stiffness: float | None = None,
        inertia: float | None = None,
        max_velocity: float | None = None,
        max_acceleration: float | None = None,
        critical_damping: bool | None = None,
        damping_ratio: float | None = None,
    ) -> None:
        if damping is not None:
            self.base_damping = float(max(0.0, damping))
            self.joint_damping[:] = self.base_damping
        if stiffness is not None:
            self.base_stiffness = float(max(0.0, stiffness))
            self.joint_stiffness[:] = self.base_stiffness
        if inertia is not None:
            self.base_inertia = float(max(0.01, inertia))
            self.joint_inertia[:] = self.base_inertia
        if max_velocity is not None:
            self.max_velocity = float(max(0.01, max_velocity))
        if max_acceleration is not None:
            self.max_acceleration = float(max(1.0, max_acceleration))
        if critical_damping is not None:
            self.critical_damping = bool(critical_damping)
        if damping_ratio is not None:
            self.damping_ratio = float(max(0.05, damping_ratio))

        self._configure_spring_profiles()

    def set_target_rotations(self, target_rotations: np.ndarray) -> None:
        self.target_rotations[:] = target_rotations.astype(np.float32, copy=False)
        if not self._initialized:
            self.current_rotations[:] = self.target_rotations
            self._initialized = True

    def update(self, dt: float, paused: bool) -> np.ndarray:
        if not self._initialized:
            return self.current_rotations

        dts = float(np.clip(dt, 1e-4, 1.0 / 20.0))
        if paused:
            self.angular_velocity *= np.exp(-self.base_damping * dts)
            self.angular_acceleration *= 0.0
            return self.current_rotations

        for j in range(self.joint_count):
            current = self.current_rotations[j]
            target = self.target_rotations[j]

            r_err = target @ current.T
            err = _log_so3(r_err)

            k = float(self.joint_stiffness[j])
            c = float(self.joint_damping[j])
            inertia = float(self.joint_inertia[j])
            if self.critical_damping:
                c = 2.0 * self.damping_ratio * np.sqrt(max(1e-6, k * inertia))

            # Second-order spring-damper with inertia in local rotation-space.
            raw_accel = (k * err - c * self.angular_velocity[j]) / inertia
            raw_accel = np.clip(raw_accel, -self.max_acceleration, self.max_acceleration)
            self.angular_acceleration[j] = (
                (1.0 - self.velocity_smoothing) * self.angular_acceleration[j]
                + self.velocity_smoothing * raw_accel
            ).astype(np.float32)

            vel_prev = self.angular_velocity[j].copy()
            vel_raw = vel_prev + self.angular_acceleration[j] * dts
            vel = (1.0 - self.velocity_smoothing) * vel_prev + self.velocity_smoothing * vel_raw

            speed = float(np.linalg.norm(vel))
            if speed > self.max_velocity:
                vel *= self.max_velocity / speed
            self.angular_velocity[j] = vel.astype(np.float32)

            delta_r = _exp_so3(self.angular_velocity[j] * dts)
            r_new = delta_r @ current

            # Keep matrices orthonormal for long-run stability.
            u, _, vt = np.linalg.svd(r_new)
            self.current_rotations[j] = (u @ vt).astype(np.float32)

        return self.current_rotations
