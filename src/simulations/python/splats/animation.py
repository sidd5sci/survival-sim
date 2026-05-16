from __future__ import annotations

from dataclasses import dataclass

import numpy as np

JOINT_NAMES = [
    "pelvis",
    "spine",
    "chest",
    "neck",
    "head",
    "shoulder_l",
    "elbow_l",
    "hand_l",
    "shoulder_r",
    "elbow_r",
    "hand_r",
    "hip_l",
    "knee_l",
    "foot_l",
    "hip_r",
    "knee_r",
    "foot_r",
]

JOINT_INDEX = {name: idx for idx, name in enumerate(JOINT_NAMES)}

PARENTS = np.array(
    [
        -1,  # pelvis
        0,  # spine
        1,  # chest
        2,  # neck
        3,  # head
        2,  # shoulder_l
        5,  # elbow_l
        6,  # hand_l
        2,  # shoulder_r
        8,  # elbow_r
        9,  # hand_r
        0,  # hip_l
        11,  # knee_l
        12,  # foot_l
        0,  # hip_r
        14,  # knee_r
        15,  # foot_r
    ],
    dtype=np.int32,
)

REST_OFFSETS = np.array(
    [
        [0.00, 0.92, 0.00],  # pelvis world origin for bind pose
        [0.00, 0.22, 0.01],
        [0.00, 0.22, 0.02],
        [0.00, 0.15, 0.02],
        [0.00, 0.15, 0.01],
        [-0.25, 0.12, 0.02],
        [0.00, -0.30, 0.00],
        [0.00, -0.24, 0.02],
        [0.25, 0.12, 0.02],
        [0.00, -0.30, 0.00],
        [0.00, -0.24, 0.02],
        [-0.15, -0.06, 0.00],
        [-0.01, -0.44, 0.03],
        [-0.02, -0.43, 0.09],
        [0.15, -0.06, 0.00],
        [0.01, -0.44, 0.03],
        [0.02, -0.43, 0.09],
    ],
    dtype=np.float32,
)


@dataclass
class Pose:
    local_rotations: np.ndarray
    root_offset: np.ndarray


@dataclass
class MotionParams:
    freq_scale: float = 1.0
    arm_amp: float = 1.0
    leg_amp: float = 1.0
    breath_amp: float = 1.0
    torso_twist: float = 1.0
    bob_amp: float = 1.0
    forward_amp: float = 1.0
    head_nod_amp: float = 1.0
    turn_bias: float = 0.0


class ParticleSkinner:
    """Vectorized CPU skinning that scales well to large particle counts."""

    def __init__(self, joint_indices: np.ndarray, joint_count: int):
        self.groups = [np.where(joint_indices == i)[0] for i in range(joint_count)]

    def deform(
        self,
        local_positions: np.ndarray,
        world_rot: np.ndarray,
        world_pos: np.ndarray,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        if out is None:
            out = np.empty_like(local_positions)

        for joint_id, idx in enumerate(self.groups):
            if idx.size == 0:
                continue
            out[idx] = local_positions[idx] @ world_rot[joint_id].T + world_pos[joint_id]
        return out


class SkeletalAnimator:
    """
    Lightweight procedural skeletal animation.

    Includes a pose interpolation hook so future authored/keyframed poses can be
    blended with the procedural walk cycle.
    """

    def __init__(self, walk_frequency_hz: float = 1.6):
        self.time = 0.0
        self.walk_frequency_hz = walk_frequency_hz
        self.joint_count = len(JOINT_NAMES)

    def step(self, dt: float, paused: bool) -> None:
        if not paused:
            self.time += dt

    @staticmethod
    def _rot_x(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)

    @staticmethod
    def _rot_y(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)

    @staticmethod
    def _rot_z(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    def sample_pose(self, t: float, motion: MotionParams | None = None) -> Pose:
        if motion is None:
            motion = MotionParams()

        local_rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], self.joint_count, axis=0)

        w = np.float32(2.0 * np.pi * self.walk_frequency_hz * motion.freq_scale)
        walk = np.sin(w * t)
        walk2 = np.sin(2.0 * w * t + 0.6)
        breathing = np.sin(2.0 * np.pi * 0.28 * t)

        # Pelvis root motion and subtle idle breathing.
        root_offset = np.array(
            [
                0.0,
                (0.02 * walk2) * motion.bob_amp + (0.010 * breathing) * motion.breath_amp,
                (0.05 * np.sin(w * t + np.pi / 2.0)) * motion.forward_amp,
            ],
            dtype=np.float32,
        )
        local_rot[JOINT_INDEX["pelvis"]] = self._rot_y(0.08 * walk + 0.22 * motion.turn_bias)

        # Spine/chest breathing and gentle counter-twist.
        local_rot[JOINT_INDEX["spine"]] = self._rot_z(0.035 * walk * motion.torso_twist) @ self._rot_x(
            0.03 * breathing * motion.breath_amp
        )
        local_rot[JOINT_INDEX["chest"]] = self._rot_z(-0.045 * walk * motion.torso_twist) @ self._rot_x(
            0.05 * breathing * motion.breath_amp
        )
        local_rot[JOINT_INDEX["neck"]] = self._rot_y(0.05 * walk + 0.10 * motion.turn_bias)
        local_rot[JOINT_INDEX["head"]] = self._rot_x(0.03 * breathing * motion.head_nod_amp) @ self._rot_y(
            0.09 * walk + 0.14 * motion.turn_bias
        )

        # Arms swing opposite legs for a natural walk rhythm.
        left_arm_swing = 0.60 * walk * motion.arm_amp
        right_arm_swing = -0.60 * walk * motion.arm_amp
        local_rot[JOINT_INDEX["shoulder_l"]] = self._rot_x(left_arm_swing)
        local_rot[JOINT_INDEX["shoulder_r"]] = self._rot_x(right_arm_swing)

        local_rot[JOINT_INDEX["elbow_l"]] = self._rot_x(0.20 + 0.18 * np.maximum(0.0, -walk) * motion.arm_amp)
        local_rot[JOINT_INDEX["elbow_r"]] = self._rot_x(0.20 + 0.18 * np.maximum(0.0, walk) * motion.arm_amp)

        # Legs with stronger knee articulation for clearer bend.
        leg_amp_signed = motion.leg_amp
        leg_amp_mag = max(0.25, abs(motion.leg_amp))

        left_leg = -0.78 * walk * leg_amp_signed
        right_leg = 0.78 * walk * leg_amp_signed
        local_rot[JOINT_INDEX["hip_l"]] = self._rot_x(left_leg)
        local_rot[JOINT_INDEX["hip_r"]] = self._rot_x(right_leg)

        # Use magnitude so knees keep flexing even when walking backward.
        knee_l = 0.18 + 0.90 * np.maximum(0.0, walk) * leg_amp_mag
        knee_r = 0.18 + 0.90 * np.maximum(0.0, -walk) * leg_amp_mag
        local_rot[JOINT_INDEX["knee_l"]] = self._rot_x(knee_l)
        local_rot[JOINT_INDEX["knee_r"]] = self._rot_x(knee_r)

        local_rot[JOINT_INDEX["foot_l"]] = self._rot_x(-0.28 * np.maximum(0.0, -walk) * leg_amp_mag)
        local_rot[JOINT_INDEX["foot_r"]] = self._rot_x(-0.28 * np.maximum(0.0, walk) * leg_amp_mag)

        return Pose(local_rotations=local_rot, root_offset=root_offset)

    @staticmethod
    def interpolate_pose(a: Pose, b: Pose, t: float) -> Pose:
        """Linear matrix blend hook for future pose interpolation systems."""
        t = np.clip(t, 0.0, 1.0)
        rot = ((1.0 - t) * a.local_rotations + t * b.local_rotations).astype(np.float32)
        root = ((1.0 - t) * a.root_offset + t * b.root_offset).astype(np.float32)
        return Pose(local_rotations=rot, root_offset=root)

    def get_joint_world_transforms(
        self,
        dt: float,
        paused: bool,
        target_pose: Pose | None = None,
        blend: float = 0.0,
        motion: MotionParams | None = None,
        layer_pose: Pose | None = None,
        layer_blend: float = 0.0,
        layer_joint_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.step(dt, paused)
        base_pose = self.sample_pose(self.time, motion=motion)

        pose = base_pose
        if target_pose is not None:
            pose = self.interpolate_pose(base_pose, target_pose, blend)

        if layer_pose is not None and layer_blend > 1e-4:
            pose = Pose(local_rotations=pose.local_rotations.copy(), root_offset=pose.root_offset)
            lb = float(np.clip(layer_blend, 0.0, 1.0))
            if layer_joint_indices is None:
                layer_joint_indices = np.arange(self.joint_count, dtype=np.int32)
            for j in layer_joint_indices:
                pose.local_rotations[j] = (
                    (1.0 - lb) * pose.local_rotations[j] + lb * layer_pose.local_rotations[j]
                ).astype(np.float32)

        world_rot = np.zeros((self.joint_count, 3, 3), dtype=np.float32)
        world_pos = np.zeros((self.joint_count, 3), dtype=np.float32)

        for j in range(self.joint_count):
            parent = PARENTS[j]
            if parent == -1:
                world_rot[j] = pose.local_rotations[j]
                world_pos[j] = REST_OFFSETS[j] + pose.root_offset
            else:
                world_rot[j] = world_rot[parent] @ pose.local_rotations[j]
                world_pos[j] = world_pos[parent] + world_rot[parent] @ REST_OFFSETS[j]

        return world_rot, world_pos
