from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from animation import JOINT_INDEX, JOINT_NAMES, PARENTS, Pose, REST_OFFSETS


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
class JointDynamics:
    stiffness: float
    damping: float
    angular_damping: float
    inertia: float
    follow_through: float


class MotionPhysicsController:
    """
    Physics-based motion layer that smooths target skeletal pose using spring dynamics.

    Pipeline:
    target_pose -> spring dynamics -> smoothed physical pose -> final skeleton pose
    """

    def __init__(self) -> None:
        self.joint_count = len(JOINT_NAMES)

        eye = np.eye(3, dtype=np.float32)
        self.current_rot = np.repeat(eye[None, :, :], self.joint_count, axis=0)
        self.angular_vel = np.zeros((self.joint_count, 3), dtype=np.float32)
        self.spring_force = np.zeros((self.joint_count, 3), dtype=np.float32)

        self.root_offset = np.zeros(3, dtype=np.float32)
        self.root_vel = np.zeros(3, dtype=np.float32)
        self.root_accel = np.zeros(3, dtype=np.float32)
        self.root_motion_vel = np.zeros(3, dtype=np.float32)

        self.prev_world_pos = np.zeros((self.joint_count, 3), dtype=np.float32)
        self.world_vel = np.zeros((self.joint_count, 3), dtype=np.float32)

        self.target_world_pos = np.zeros((self.joint_count, 3), dtype=np.float32)
        self.current_world_pos = np.zeros((self.joint_count, 3), dtype=np.float32)
        self.com = np.zeros(3, dtype=np.float32)

        self.plant_state = {"l": False, "r": False}
        self.plant_world = {
            "l": np.zeros(3, dtype=np.float32),
            "r": np.zeros(3, dtype=np.float32),
        }
        self.prev_foot_world = {
            "l": np.zeros(3, dtype=np.float32),
            "r": np.zeros(3, dtype=np.float32),
        }

        self.masses = np.ones((self.joint_count,), dtype=np.float32)
        self._configure_joint_masses()

        self.dynamics = [
            JointDynamics(
                stiffness=24.0,
                damping=8.2,
                angular_damping=4.0,
                inertia=1.0,
                follow_through=0.12,
            )
            for _ in range(self.joint_count)
        ]
        self._configure_joint_dynamics()

        self._initialized = False

    def _configure_joint_masses(self) -> None:
        self.masses[JOINT_INDEX["pelvis"]] = 6.2
        self.masses[JOINT_INDEX["spine"]] = 4.8
        self.masses[JOINT_INDEX["chest"]] = 4.2
        self.masses[JOINT_INDEX["neck"]] = 1.2
        self.masses[JOINT_INDEX["head"]] = 1.8

        for side in ("l", "r"):
            self.masses[JOINT_INDEX[f"shoulder_{side}"]] = 1.3
            self.masses[JOINT_INDEX[f"elbow_{side}"]] = 1.1
            self.masses[JOINT_INDEX[f"hand_{side}"]] = 0.8
            self.masses[JOINT_INDEX[f"hip_{side}"]] = 2.5
            self.masses[JOINT_INDEX[f"knee_{side}"]] = 2.1
            self.masses[JOINT_INDEX[f"foot_{side}"]] = 1.7

    def _set_joint(self, name: str, **kwargs: float) -> None:
        j = JOINT_INDEX[name]
        d = self.dynamics[j]
        self.dynamics[j] = JointDynamics(
            stiffness=kwargs.get("stiffness", d.stiffness),
            damping=kwargs.get("damping", d.damping),
            angular_damping=kwargs.get("angular_damping", d.angular_damping),
            inertia=kwargs.get("inertia", d.inertia),
            follow_through=kwargs.get("follow_through", d.follow_through),
        )

    def _configure_joint_dynamics(self) -> None:
        # Pelvis stabilization and body inertia core.
        self._set_joint("pelvis", stiffness=18.0, damping=9.5, angular_damping=5.2, inertia=1.5, follow_through=0.07)
        self._set_joint("spine", stiffness=16.0, damping=7.6, angular_damping=4.1, inertia=1.35, follow_through=0.22)
        self._set_joint("chest", stiffness=15.0, damping=7.2, angular_damping=3.9, inertia=1.35, follow_through=0.26)

        # Head stabilization.
        self._set_joint("neck", stiffness=20.0, damping=10.5, angular_damping=5.5, inertia=0.85, follow_through=0.05)
        self._set_joint("head", stiffness=21.0, damping=11.0, angular_damping=6.0, inertia=0.80, follow_through=0.04)

        # Shoulder lag + arm overswing bias.
        for side in ("l", "r"):
            self._set_joint(f"shoulder_{side}", stiffness=11.0, damping=4.8, angular_damping=2.8, inertia=1.7, follow_through=0.42)
            self._set_joint(f"elbow_{side}", stiffness=14.0, damping=5.5, angular_damping=3.0, inertia=1.5, follow_through=0.35)
            self._set_joint(f"hand_{side}", stiffness=14.0, damping=6.0, angular_damping=3.2, inertia=1.35, follow_through=0.28)

        # Legs stay firmer to preserve support.
        for side in ("l", "r"):
            self._set_joint(f"hip_{side}", stiffness=20.0, damping=8.0, angular_damping=4.1, inertia=1.25, follow_through=0.12)
            self._set_joint(f"knee_{side}", stiffness=22.0, damping=8.6, angular_damping=4.2, inertia=1.15, follow_through=0.10)
            self._set_joint(f"foot_{side}", stiffness=24.0, damping=9.4, angular_damping=4.5, inertia=1.05, follow_through=0.08)

    def _forward_kinematics(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        world_rot = np.zeros((self.joint_count, 3, 3), dtype=np.float32)
        world_pos = np.zeros((self.joint_count, 3), dtype=np.float32)
        for j in range(self.joint_count):
            p = PARENTS[j]
            if p < 0:
                world_rot[j] = pose.local_rotations[j]
                world_pos[j] = REST_OFFSETS[j] + pose.root_offset
            else:
                world_rot[j] = world_rot[p] @ pose.local_rotations[j]
                world_pos[j] = world_pos[p] + world_rot[p] @ REST_OFFSETS[j]
        return world_rot, world_pos

    def update_root_motion(self, desired_velocity: np.ndarray, dt: float, paused: bool) -> np.ndarray:
        if paused:
            return np.zeros(3, dtype=np.float32)

        dts = max(1e-4, float(dt))
        target_v = desired_velocity.astype(np.float32, copy=False)

        # Smooth acceleration/deceleration with momentum.
        accel = 13.5 * (target_v - self.root_motion_vel) - 3.8 * self.root_motion_vel
        self.root_accel = (0.70 * self.root_accel + 0.30 * accel).astype(np.float32)
        self.root_motion_vel = (self.root_motion_vel + self.root_accel * dts).astype(np.float32)

        # Motion damping.
        self.root_motion_vel *= np.exp(-1.6 * dts)
        return (self.root_motion_vel * dts).astype(np.float32)

    def _spring_joint(self, j: int, target_rot: np.ndarray, parent_ang_vel: np.ndarray, dt: float) -> np.ndarray:
        d = self.dynamics[j]
        current = self.current_rot[j]

        r_err = target_rot @ current.T
        err = _log_so3(r_err)

        follow = d.follow_through * parent_ang_vel
        force = d.stiffness * (err + follow) - d.damping * self.angular_vel[j]

        # Arm overswing from shoulder momentum.
        if j in (JOINT_INDEX["elbow_l"], JOINT_INDEX["elbow_r"]):
            shoulder = JOINT_INDEX["shoulder_l"] if j == JOINT_INDEX["elbow_l"] else JOINT_INDEX["shoulder_r"]
            force += 0.18 * self.angular_vel[shoulder]

        self.spring_force[j] = force.astype(np.float32)

        ang_acc = force / max(0.05, d.inertia)
        self.angular_vel[j] = (self.angular_vel[j] + ang_acc * dt).astype(np.float32)

        # Angular damping.
        self.angular_vel[j] *= np.exp(-d.angular_damping * dt)

        delta_r = _exp_so3(self.angular_vel[j] * dt)
        new_r = delta_r @ current

        # Re-orthonormalize for stability.
        u, _, vt = np.linalg.svd(new_r)
        self.current_rot[j] = (u @ vt).astype(np.float32)
        return self.current_rot[j]

    def _estimate_com(self, world_pos: np.ndarray) -> np.ndarray:
        m = self.masses[:, None]
        total = float(np.sum(m))
        return (np.sum(world_pos * m, axis=0) / max(1e-6, total)).astype(np.float32)

    def _support_point(self, world_pos: np.ndarray) -> np.ndarray:
        if self.plant_state["l"] and self.plant_state["r"]:
            return ((self.plant_world["l"] + self.plant_world["r"]) * 0.5).astype(np.float32)
        if self.plant_state["l"]:
            return self.plant_world["l"].astype(np.float32)
        if self.plant_state["r"]:
            return self.plant_world["r"].astype(np.float32)

        foot_l = world_pos[JOINT_INDEX["foot_l"]]
        foot_r = world_pos[JOINT_INDEX["foot_r"]]
        return ((foot_l + foot_r) * 0.5).astype(np.float32)

    def _apply_balance(self, pose: Pose, world_pos: np.ndarray) -> Pose:
        com = self._estimate_com(world_pos)
        self.com = com

        support = self._support_point(world_pos)
        error = com - support

        corrected = Pose(local_rotations=pose.local_rotations.copy(), root_offset=pose.root_offset.copy())

        # Pelvis compensation in horizontal plane.
        corrected.root_offset[0] -= 0.22 * error[0]
        corrected.root_offset[2] -= 0.24 * error[2]

        # Upper body stabilization counters drift.
        chest = JOINT_INDEX["chest"]
        spine = JOINT_INDEX["spine"]
        corrected.local_rotations[spine] = _rot_z(0.08 * error[0]) @ corrected.local_rotations[spine]
        corrected.local_rotations[chest] = _rot_z(0.12 * error[0]) @ _rot_x(-0.06 * error[2]) @ corrected.local_rotations[chest]
        return corrected

    def update_foot_plants(self, world_pos: np.ndarray, root_speed: float, dt: float, ground_y: float) -> None:
        for side, foot_idx in (("l", JOINT_INDEX["foot_l"]), ("r", JOINT_INDEX["foot_r"])):
            curr = world_pos[foot_idx].astype(np.float32)
            prev = self.prev_foot_world[side]
            foot_vel = float(np.linalg.norm((curr - prev) / max(1e-4, dt)))
            self.prev_foot_world[side] = curr

            near_ground = curr[1] <= ground_y + 0.055
            low_vel = foot_vel < 0.17
            moving = root_speed > 0.22

            if self.plant_state[side]:
                release = (not near_ground) or (foot_vel > 0.40)
                if release:
                    self.plant_state[side] = False
            else:
                if near_ground and low_vel and moving:
                    self.plant_state[side] = True
                    self.plant_world[side] = np.array([curr[0], ground_y, curr[2]], dtype=np.float32)

    def foot_target(self, side: str, current_foot: np.ndarray, ground_y: float) -> np.ndarray:
        if self.plant_state[side]:
            t = self.plant_world[side].copy()
            t[1] = ground_y
            return t.astype(np.float32)
        t = current_foot.copy()
        t[1] = ground_y
        return t.astype(np.float32)

    def apply(self, target_pose: Pose, dt: float, paused: bool) -> Pose:
        dts = max(1e-4, float(dt))
        if not self._initialized:
            self.current_rot[:] = target_pose.local_rotations
            self.root_offset[:] = target_pose.root_offset
            self._initialized = True

        if paused:
            return Pose(local_rotations=self.current_rot.copy(), root_offset=self.root_offset.copy())

        out_rot = self.current_rot.copy()

        # Root inertia and stabilization.
        root_target = target_pose.root_offset.astype(np.float32, copy=False)
        root_err = root_target - self.root_offset
        root_force = 21.0 * root_err - 9.0 * self.root_vel
        self.root_vel = (self.root_vel + root_force * dts).astype(np.float32)
        self.root_vel *= np.exp(-3.0 * dts)
        self.root_offset = (self.root_offset + self.root_vel * dts).astype(np.float32)

        parent_ang = np.zeros((self.joint_count, 3), dtype=np.float32)

        for j in range(self.joint_count):
            p = PARENTS[j]
            if p >= 0:
                parent_ang[j] = self.angular_vel[p]
            out_rot[j] = self._spring_joint(j, target_pose.local_rotations[j], parent_ang[j], dts)

        # Movement anticipation based on root acceleration.
        fwd_accel = float(self.root_accel[2])
        side_accel = float(self.root_accel[0])
        out_rot[JOINT_INDEX["spine"]] = _rot_x(-0.030 * fwd_accel) @ _rot_z(0.020 * side_accel) @ out_rot[JOINT_INDEX["spine"]]
        out_rot[JOINT_INDEX["chest"]] = _rot_x(-0.042 * fwd_accel) @ _rot_z(0.024 * side_accel) @ out_rot[JOINT_INDEX["chest"]]

        # Torso drag against movement velocity.
        torso_drag = float(np.clip(np.linalg.norm(self.root_motion_vel), 0.0, 2.0))
        out_rot[JOINT_INDEX["chest"]] = _rot_y(0.030 * torso_drag) @ out_rot[JOINT_INDEX["chest"]]

        pose = Pose(local_rotations=out_rot, root_offset=self.root_offset.copy())

        # Balance pass based on COM and support point.
        _, world_pos = self._forward_kinematics(pose)
        pose = self._apply_balance(pose, world_pos)

        # Head stabilization counteracts chest over-rotation.
        pose.local_rotations[JOINT_INDEX["head"]] = _rot_x(-0.14 * self.angular_vel[JOINT_INDEX["chest"]][0]) @ pose.local_rotations[JOINT_INDEX["head"]]
        pose.local_rotations[JOINT_INDEX["head"]] = _rot_z(-0.10 * self.angular_vel[JOINT_INDEX["chest"]][2]) @ pose.local_rotations[JOINT_INDEX["head"]]

        target_world_rot, target_world_pos = self._forward_kinematics(target_pose)
        curr_world_rot, curr_world_pos = self._forward_kinematics(pose)
        _ = target_world_rot, curr_world_rot

        self.target_world_pos = target_world_pos
        self.current_world_pos = curr_world_pos

        vel = (curr_world_pos - self.prev_world_pos) / dts
        self.world_vel = (0.72 * self.world_vel + 0.28 * vel).astype(np.float32)
        self.prev_world_pos = curr_world_pos.copy()

        return pose

    def build_debug_vertices(self) -> list[float]:
        verts: list[float] = []

        # Velocity vectors.
        debug_joints = [
            JOINT_INDEX["pelvis"],
            JOINT_INDEX["chest"],
            JOINT_INDEX["head"],
            JOINT_INDEX["hand_l"],
            JOINT_INDEX["hand_r"],
            JOINT_INDEX["foot_l"],
            JOINT_INDEX["foot_r"],
        ]
        for j in debug_joints:
            a = self.current_world_pos[j]
            v = self.world_vel[j] * 0.08
            b = a + v
            c = (0.95, 0.35, 0.25)
            verts.extend([float(a[0]), float(a[1]), float(a[2]), *c])
            verts.extend([float(b[0]), float(b[1]), float(b[2]), *c])

        # Joint lag lines.
        for j in debug_joints:
            a = self.current_world_pos[j]
            b = self.target_world_pos[j]
            c = (0.30, 0.85, 0.95)
            verts.extend([float(a[0]), float(a[1]), float(a[2]), *c])
            verts.extend([float(b[0]), float(b[1]), float(b[2]), *c])

        # Spring force vectors.
        for j in (JOINT_INDEX["shoulder_l"], JOINT_INDEX["shoulder_r"], JOINT_INDEX["elbow_l"], JOINT_INDEX["elbow_r"], JOINT_INDEX["spine"], JOINT_INDEX["chest"]):
            a = self.current_world_pos[j]
            f = self.spring_force[j] * 0.010
            b = a + f
            c = (0.45, 0.95, 0.35)
            verts.extend([float(a[0]), float(a[1]), float(a[2]), *c])
            verts.extend([float(b[0]), float(b[1]), float(b[2]), *c])

        # COM marker (small cross).
        com = self.com
        cs = 0.06
        c = (1.0, 0.92, 0.22)
        verts.extend([com[0] - cs, com[1], com[2], *c, com[0] + cs, com[1], com[2], *c])
        verts.extend([com[0], com[1] - cs, com[2], *c, com[0], com[1] + cs, com[2], *c])
        verts.extend([com[0], com[1], com[2] - cs, *c, com[0], com[1], com[2] + cs, *c])

        # Planted feet markers.
        for side in ("l", "r"):
            if not self.plant_state[side]:
                continue
            p = self.plant_world[side]
            c2 = (0.26, 0.98, 0.45)
            s = 0.05
            verts.extend([p[0] - s, p[1], p[2], *c2, p[0] + s, p[1], p[2], *c2])
            verts.extend([p[0], p[1], p[2] - s, *c2, p[0], p[1], p[2] + s, *c2])

        return verts
