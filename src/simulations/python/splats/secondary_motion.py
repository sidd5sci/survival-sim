from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from animation import JOINT_INDEX, Pose


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
class SecondaryMotionState:
    root_velocity: np.ndarray
    root_acceleration: np.ndarray
    turn_speed: float


class SecondaryMotionController:
    """Upper-body secondary motion layer driven by velocity, acceleration, and turn speed."""

    def __init__(self, response: float = 8.0) -> None:
        self.response = float(max(1.0, response))

        self.prev_root_offset = np.zeros((3,), dtype=np.float32)
        self.root_velocity = np.zeros((3,), dtype=np.float32)
        self.root_acceleration = np.zeros((3,), dtype=np.float32)

        self.prev_pelvis_yaw = 0.0
        self.turn_speed = 0.0

        self.shoulder_lag = np.zeros((2,), dtype=np.float32)
        self.spine_drag = np.zeros((2,), dtype=np.float32)
        self.head_stabilize = np.zeros((2,), dtype=np.float32)
        self.arm_follow = np.zeros((4,), dtype=np.float32)

        self._initialized = False

    @staticmethod
    def _yaw_from_rotation(r: np.ndarray) -> float:
        # Approximate yaw from rotation matrix.
        return float(np.arctan2(r[0, 2], r[2, 2]))

    @staticmethod
    def _approach_vec(curr: np.ndarray, target: np.ndarray, rate: float, dt: float) -> np.ndarray:
        alpha = 1.0 - np.exp(-rate * dt)
        return (curr + alpha * (target - curr)).astype(np.float32)

    @staticmethod
    def _approach_scalar(curr: float, target: float, rate: float, dt: float) -> float:
        alpha = 1.0 - np.exp(-rate * dt)
        return float(curr + alpha * (target - curr))

    def _update_kinematics(self, pose: Pose, dt: float, paused: bool) -> SecondaryMotionState:
        dts = max(1e-4, float(dt))

        if not self._initialized:
            self.prev_root_offset[:] = pose.root_offset
            self.prev_pelvis_yaw = self._yaw_from_rotation(pose.local_rotations[JOINT_INDEX["pelvis"]])
            self._initialized = True

        if paused:
            return SecondaryMotionState(
                root_velocity=self.root_velocity.copy(),
                root_acceleration=self.root_acceleration.copy(),
                turn_speed=self.turn_speed,
            )

        vel_raw = (pose.root_offset - self.prev_root_offset) / dts
        vel = self._approach_vec(self.root_velocity, vel_raw.astype(np.float32), rate=10.0, dt=dts)
        accel_raw = (vel - self.root_velocity) / dts
        accel = self._approach_vec(self.root_acceleration, accel_raw.astype(np.float32), rate=8.0, dt=dts)

        pelvis_yaw = self._yaw_from_rotation(pose.local_rotations[JOINT_INDEX["pelvis"]])
        dyaw = pelvis_yaw - self.prev_pelvis_yaw
        while dyaw > np.pi:
            dyaw -= 2.0 * np.pi
        while dyaw < -np.pi:
            dyaw += 2.0 * np.pi
        turn_raw = float(dyaw / dts)
        turn_speed = self._approach_scalar(self.turn_speed, turn_raw, rate=10.0, dt=dts)

        self.prev_root_offset[:] = pose.root_offset
        self.root_velocity[:] = vel
        self.root_acceleration[:] = accel
        self.prev_pelvis_yaw = pelvis_yaw
        self.turn_speed = turn_speed

        return SecondaryMotionState(
            root_velocity=self.root_velocity.copy(),
            root_acceleration=self.root_acceleration.copy(),
            turn_speed=self.turn_speed,
        )

    def apply(self, pose: Pose, dt: float, paused: bool) -> Pose:
        state = self._update_kinematics(pose=pose, dt=dt, paused=paused)
        if paused:
            return pose

        dts = max(1e-4, float(dt))
        out = Pose(local_rotations=pose.local_rotations.copy(), root_offset=pose.root_offset.copy())

        speed = float(np.linalg.norm(state.root_velocity))
        accel_mag = float(np.linalg.norm(state.root_acceleration))
        lateral_vel = float(state.root_velocity[0])
        forward_vel = float(state.root_velocity[2])
        forward_accel = float(state.root_acceleration[2])
        lateral_accel = float(state.root_acceleration[0])
        turn = float(state.turn_speed)

        # Shoulder lag: delayed counter-rotation to sudden movement and turning.
        shoulder_target = np.array(
            [
                -0.18 * lateral_accel - 0.05 * turn,
                -0.12 * turn - 0.04 * lateral_vel,
            ],
            dtype=np.float32,
        )
        self.shoulder_lag = self._approach_vec(self.shoulder_lag, shoulder_target, rate=self.response * 0.75, dt=dts)

        # Spine drag: torso trails momentum and catches up naturally.
        spine_target = np.array(
            [
                -0.12 * forward_accel - 0.06 * forward_vel,
                -0.16 * lateral_accel - 0.07 * turn,
            ],
            dtype=np.float32,
        )
        self.spine_drag = self._approach_vec(self.spine_drag, spine_target, rate=self.response * 0.62, dt=dts)

        # Head stabilization: counter chest/body perturbations.
        head_target = np.array(
            [
                0.10 * forward_accel + 0.03 * speed,
                0.10 * turn + 0.03 * lateral_accel,
            ],
            dtype=np.float32,
        )
        self.head_stabilize = self._approach_vec(self.head_stabilize, head_target, rate=self.response * 0.90, dt=dts)

        # Arm follow-through: delayed swing with overshoot tendency during turns/speed changes.
        arm_target = np.array(
            [
                -0.28 * forward_accel - 0.16 * turn,
                0.16 * lateral_accel,
                -0.28 * forward_accel + 0.16 * turn,
                0.16 * lateral_accel,
            ],
            dtype=np.float32,
        )
        self.arm_follow = self._approach_vec(self.arm_follow, arm_target, rate=self.response * 0.55, dt=dts)

        # Clamp amplitudes to keep secondary motion stable and believable.
        self.shoulder_lag = np.clip(self.shoulder_lag, -0.48, 0.48)
        self.spine_drag = np.clip(self.spine_drag, -0.38, 0.38)
        self.head_stabilize = np.clip(self.head_stabilize, -0.30, 0.30)
        self.arm_follow = np.clip(self.arm_follow, -0.66, 0.66)

        spine = JOINT_INDEX["spine"]
        chest = JOINT_INDEX["chest"]
        neck = JOINT_INDEX["neck"]
        head = JOINT_INDEX["head"]
        shoulder_l = JOINT_INDEX["shoulder_l"]
        shoulder_r = JOINT_INDEX["shoulder_r"]
        elbow_l = JOINT_INDEX["elbow_l"]
        elbow_r = JOINT_INDEX["elbow_r"]

        # Apply spine drag across spine/chest for upper-body softness.
        out.local_rotations[spine] = _rot_x(self.spine_drag[0] * 0.9) @ _rot_z(self.spine_drag[1] * 0.75) @ out.local_rotations[spine]
        out.local_rotations[chest] = _rot_x(self.spine_drag[0]) @ _rot_z(self.spine_drag[1]) @ out.local_rotations[chest]

        # Shoulder lag and arm follow-through.
        out.local_rotations[shoulder_l] = _rot_x(self.shoulder_lag[0] + self.arm_follow[0]) @ _rot_y(self.shoulder_lag[1]) @ out.local_rotations[shoulder_l]
        out.local_rotations[shoulder_r] = _rot_x(self.shoulder_lag[0] + self.arm_follow[2]) @ _rot_y(-self.shoulder_lag[1]) @ out.local_rotations[shoulder_r]

        out.local_rotations[elbow_l] = _rot_x(self.arm_follow[0] * 0.58) @ _rot_z(self.arm_follow[1] * 0.52) @ out.local_rotations[elbow_l]
        out.local_rotations[elbow_r] = _rot_x(self.arm_follow[2] * 0.58) @ _rot_z(-self.arm_follow[3] * 0.52) @ out.local_rotations[elbow_r]

        # Head stabilization uses neck + head counter offsets.
        out.local_rotations[neck] = _rot_x(-self.head_stabilize[0] * 0.85) @ _rot_y(-self.head_stabilize[1] * 0.95) @ out.local_rotations[neck]
        out.local_rotations[head] = _rot_x(-self.head_stabilize[0]) @ _rot_y(-self.head_stabilize[1]) @ out.local_rotations[head]

        return out
