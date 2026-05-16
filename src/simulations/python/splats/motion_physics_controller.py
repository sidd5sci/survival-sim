from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from animation import JOINT_INDEX, JOINT_NAMES, PARENTS, Pose, REST_OFFSETS
from balance_system import BalanceSystemController
from foot_planting import FootContactState, FootPhaseSample, FootPlantingController
from joint_dynamics import JointDynamics
from root_motion import MovementState, RootMotionController, RootMotionSample
from secondary_motion import SecondaryMotionController


def _rot_lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    r = ((1.0 - t) * a + t * b).astype(np.float32)
    u, _, vt = np.linalg.svd(r)
    return (u @ vt).astype(np.float32)


@dataclass
class MotionPhysicsOutput:
    pose: Pose
    root_delta: np.ndarray
    root_sample: RootMotionSample
    foot_left: FootPhaseSample
    foot_right: FootPhaseSample
    balance_error: float


class MotionPhysicsController:
    """Unified realtime motion-physics stack for skeleton control and grounded contacts."""

    def __init__(
        self,
        joint_dynamics: JointDynamics,
        secondary_motion: SecondaryMotionController,
        root_motion: RootMotionController,
        foot_planting: FootPlantingController,
        balance_system: BalanceSystemController,
    ) -> None:
        self.joint_dynamics = joint_dynamics
        self.secondary_motion = secondary_motion
        self.root_motion = root_motion
        self.foot_planting = foot_planting
        self.balance_system = balance_system

        self._vel_target_rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], len(JOINT_NAMES), axis=0)
        self._vel_angular = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)

        self._debug_world_pos = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)
        self._debug_prev_world_pos = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)
        self._debug_world_vel = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)

        self._last_left = FootPhaseSample(side="l", state=FootContactState.PLANTED, phase=0.0, planted=True)
        self._last_right = FootPhaseSample(side="r", state=FootContactState.PLANTED, phase=0.0, planted=True)
        self._initialized = False

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

    def _velocity_dynamics_stage(self, target_pose: Pose, dt: float, paused: bool) -> Pose:
        dts = max(1e-4, float(dt))

        if not self._initialized:
            self._vel_target_rot[:] = target_pose.local_rotations
            self._initialized = True

        if paused:
            return Pose(local_rotations=self._vel_target_rot.copy(), root_offset=target_pose.root_offset)

        for j in range(len(JOINT_NAMES)):
            curr = self._vel_target_rot[j]
            tgt = target_pose.local_rotations[j]

            # Velocity-based delayed target tracking before spring stage.
            diff = tgt - curr
            accel = 11.0 * diff - 4.0 * self._vel_angular[j][:, None]
            accel_vec = np.array([np.mean(accel[:, 0]), np.mean(accel[:, 1]), np.mean(accel[:, 2])], dtype=np.float32)
            self._vel_angular[j] = (0.72 * self._vel_angular[j] + 0.28 * accel_vec).astype(np.float32)

            speed = float(np.linalg.norm(self._vel_angular[j]))
            if speed > 6.5:
                self._vel_angular[j] *= 6.5 / speed

            t = float(np.clip(np.linalg.norm(self._vel_angular[j]) * dts * 0.22, 0.04, 0.35))
            self._vel_target_rot[j] = _rot_lerp(curr, tgt, t)

        return Pose(local_rotations=self._vel_target_rot.copy(), root_offset=target_pose.root_offset)

    def update(
        self,
        animation_target_pose: Pose,
        desired_root_velocity: np.ndarray,
        character_offset: np.ndarray,
        dt: float,
        paused: bool,
        ground_y: float,
    ) -> MotionPhysicsOutput:
        # 1) Velocity dynamics stage.
        velocity_pose = self._velocity_dynamics_stage(animation_target_pose, dt=dt, paused=paused)

        # 2) Spring dynamics stage.
        self.joint_dynamics.set_target_rotations(velocity_pose.local_rotations)
        spring_rot = self.joint_dynamics.update(dt=dt, paused=paused)
        spring_pose = Pose(local_rotations=spring_rot.copy(), root_offset=velocity_pose.root_offset)

        # 3) Secondary motion stage.
        secondary_pose = self.secondary_motion.apply(spring_pose, dt=dt, paused=paused)

        # 4) Root motion stage.
        root_sample = self.root_motion.update(desired_velocity=desired_root_velocity, dt=dt, paused=paused)

        # 5) Foot planting stage (evaluate on pre-balance pose with predicted world offset).
        predicted_offset = character_offset + root_sample.delta_position
        world_pre_balance = self._fk(secondary_pose) + predicted_offset[None, :]
        left_phase, right_phase = self.foot_planting.update(
            world_pos=world_pre_balance,
            dt=dt,
            ground_y=ground_y,
            root_speed=float(np.linalg.norm(root_sample.velocity[[0, 2]])),
        )

        # 6) Balance correction stage.
        balanced_pose = self.balance_system.apply(
            pose=secondary_pose,
            root_velocity=root_sample.velocity,
            root_acceleration=root_sample.acceleration,
            turn_speed=root_sample.turn_speed,
            dt=dt,
            paused=paused,
            ground_y=ground_y,
        )

        # Debug world-space kinematics use final pose.
        world_final = self._fk(balanced_pose) + predicted_offset[None, :]
        dts = max(1e-4, float(dt))
        vel = (world_final - self._debug_prev_world_pos) / dts
        self._debug_world_vel = (0.72 * self._debug_world_vel + 0.28 * vel).astype(np.float32)
        self._debug_prev_world_pos = world_final.copy()
        self._debug_world_pos = world_final

        self._last_left = left_phase
        self._last_right = right_phase

        return MotionPhysicsOutput(
            pose=balanced_pose,
            root_delta=root_sample.delta_position,
            root_sample=root_sample,
            foot_left=left_phase,
            foot_right=right_phase,
            balance_error=self.balance_system.sample.com_error,
        )

    def foot_target(self, side: str, current_foot: np.ndarray, ground_y: float) -> np.ndarray:
        return self.foot_planting.target_for(side=side, current_foot=current_foot, ground_y=ground_y)

    def build_debug_overlay_vertices(self) -> list[float]:
        verts: list[float] = []

        # COM visualization.
        b = self.balance_system.sample
        com = b.com
        s = 0.06
        c_com = (1.0, 0.92, 0.22)
        verts.extend([com[0] - s, com[1], com[2], *c_com, com[0] + s, com[1], com[2], *c_com])
        verts.extend([com[0], com[1] - s, com[2], *c_com, com[0], com[1] + s, com[2], *c_com])
        verts.extend([com[0], com[1], com[2] - s, *c_com, com[0], com[1], com[2] + s, *c_com])

        # Body tilt vector (support center -> COM).
        sc = b.support_center
        c_tilt = (0.34, 0.88, 0.98)
        verts.extend([sc[0], sc[1] + 0.01, sc[2], *c_tilt, com[0], com[1], com[2], *c_tilt])

        # Planted feet markers.
        for side, phase in (("l", self._last_left), ("r", self._last_right)):
            if not phase.planted:
                continue
            p = self.foot_planting.plant_pos[side]
            s2 = 0.05
            c_plant = (0.26, 0.98, 0.45)
            verts.extend([p[0] - s2, p[1], p[2], *c_plant, p[0] + s2, p[1], p[2], *c_plant])
            verts.extend([p[0], p[1], p[2] - s2, *c_plant, p[0], p[1], p[2] + s2, *c_plant])

        # Motion vectors (selected joints).
        c_vel = (0.95, 0.38, 0.30)
        for j in (
            JOINT_INDEX["pelvis"],
            JOINT_INDEX["chest"],
            JOINT_INDEX["head"],
            JOINT_INDEX["hand_l"],
            JOINT_INDEX["hand_r"],
            JOINT_INDEX["foot_l"],
            JOINT_INDEX["foot_r"],
        ):
            a = self._debug_world_pos[j]
            bpt = a + self._debug_world_vel[j] * 0.08
            verts.extend([a[0], a[1], a[2], *c_vel, bpt[0], bpt[1], bpt[2], *c_vel])

        # Spring force proxies from angular acceleration.
        c_force = (0.45, 0.95, 0.35)
        for j in (
            JOINT_INDEX["spine"],
            JOINT_INDEX["chest"],
            JOINT_INDEX["shoulder_l"],
            JOINT_INDEX["shoulder_r"],
            JOINT_INDEX["elbow_l"],
            JOINT_INDEX["elbow_r"],
        ):
            a = self._debug_world_pos[j]
            f = self.joint_dynamics.angular_acceleration[j] * 0.010
            bpt = a + f
            verts.extend([a[0], a[1], a[2], *c_force, bpt[0], bpt[1], bpt[2], *c_force])

        return verts

    def debug_status(self) -> dict[str, str | float]:
        rs = self.root_motion
        return {
            "move_state": rs.state.value,
            "root_speed": float(np.linalg.norm(rs.velocity[[0, 2]])),
            "root_accel": float(np.linalg.norm(rs.acceleration[[0, 2]])),
            "foot_l": self._last_left.state.value,
            "foot_r": self._last_right.state.value,
            "balance_error": float(self.balance_system.sample.com_error),
        }
