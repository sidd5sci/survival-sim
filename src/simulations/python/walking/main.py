#!/usr/bin/env python3
"""
Genetic Walking Simulation — PyBullet (stability-fixed rebuild)

Main fixes compared with the earlier version:
  - PyBullet POSITION_CONTROL gains are dimensionless. Huge values like 400/40
    make the motors explode. This version uses sane gains.
  - NN actions are updated at 30 Hz, not 240 Hz, and targets are smoothed.
  - Walkers die on pitch fall, roll fall, excessive height, or long airtime.
  - Reward only pays meaningful forward progress while grounded/upright.
  - Fitness penalizes flying/flipping exploits.
  - Contact tracking, airtime, and target smoothing were added.
  - In-world HUD/debug labels were removed for speed.
  - Lightweight status is shown in PyBullet's right-side debug panel.
  - GUI keyboard/camera controls keep working in idle/hold mode after finite runs.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pybullet as p
import pybullet_data


# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

POP_SIZE   = 20
SIM_TIME   = 15.0
PHYS_DT    = 1.0 / 240.0
GRAVITY    = 9.81

# NN still matches your original genome size.
N_IN, N_H1, N_H2, N_OUT = 20, 64, 64, 6
N_W = N_IN*N_H1 + N_H1 + N_H1*N_H2 + N_H2 + N_H2*N_OUT + N_OUT

ELITE_N    = 5
TOURN_K    = 5
MUT_RATE   = 0.08
MUT_STD    = 0.15
CROSS_RATE = 0.5

# IMPORTANT:
# PyBullet POSITION_CONTROL gains are not physical Kp/Kd values.
# Values like 400/40 create explosive motors. Keep these around 0.05..1.0.
MOTOR_POS_GAIN = 0.38
MOTOR_VEL_GAIN = 0.08
MOTOR_TMAX     = 170.0

ARM_POS_GAIN   = 0.25
ARM_VEL_GAIN   = 0.05
ARM_TMAX       = 35.0

# NN action-rate limiting.
ACTION_HZ      = 30.0
ACTION_EVERY   = max(1, int(round((1.0 / ACTION_HZ) / PHYS_DT)))
TARGET_SMOOTH  = 0.18   # higher = faster changes; lower = smoother

FALL_Z       = 0.55
MAX_Z        = 1.55     # catches launch/flying exploit
FALL_ROLL    = 0.95
FALL_PITCH   = 0.95
MAX_AIR_TIME = 0.45

GUI_FAST       = True
GUI_SPEED      = 24.0
STATUS_INTERVAL = 1.00  # terminal-only status throttle; no PyBullet UI updates
POST_GEN_IDLE  = 0.0    # set >0.0 if you want a visible pause between generations
INIT_YAW       = 40.0
INIT_PITCH     = -20.0
INIT_DIST      = 11.0
INIT_TARGET    = [4.0, 0.0, 0.8]
DEST_X         = 10.0
STEP_YAW       = 1.4
STEP_PITCH     = 1.0
STEP_DIST      = 0.25
FOLLOW_DEFAULT = False
SAVE_FILE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_fixed.npz")

JOINT_NAMES  = ["l_hip", "l_knee", "l_ankle", "r_hip", "r_knee", "r_ankle"]
JOINT_LIMITS = {
    "l_hip":   (-0.55,  0.85),
    "l_knee":  ( 0.05,  1.45),
    "l_ankle": (-0.35,  0.32),
    "r_hip":   (-0.55,  0.85),
    "r_knee":  ( 0.05,  1.45),
    "r_ankle": (-0.35,  0.32),
}
ARM_LIMITS = {
    "l_shoulder": (-0.7, 1.0),
    "l_elbow":    ( 0.0, 1.5),
    "r_shoulder": (-0.7, 1.0),
    "r_elbow":    ( 0.0, 1.5),
}

ALL_JOINTS = ["l_hip","l_knee","l_ankle","r_hip","r_knee","r_ankle",
              "chest","neck","head","l_shoulder","l_elbow","r_shoulder","r_elbow"]
FIXED_JOINT_IDX = {6, 7, 8}


# ──────────────────────────────────────────────────────────
# Neural network
# ──────────────────────────────────────────────────────────

class Net:
    def __init__(self, genome: np.ndarray) -> None:
        idx = 0
        def _m(r, c):
            nonlocal idx
            w = genome[idx:idx + r*c].reshape(r, c)
            idx += r*c
            return w
        def _v(n):
            nonlocal idx
            v = genome[idx:idx + n]
            idx += n
            return v
        self.W1 = _m(N_IN, N_H1);  self.b1 = _v(N_H1)
        self.W2 = _m(N_H1, N_H2);  self.b2 = _v(N_H2)
        self.W3 = _m(N_H2, N_OUT); self.b3 = _v(N_OUT)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        return np.tanh(h @ self.W3 + self.b3)


# ──────────────────────────────────────────────────────────
# Genetic Algorithm
# ──────────────────────────────────────────────────────────

def _xavier(fan_in: int, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(size).astype(np.float32) * math.sqrt(2.0 / fan_in)

def random_genome(rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([
        _xavier(N_IN,  N_IN*N_H1,  rng), np.zeros(N_H1,  np.float32),
        _xavier(N_H1,  N_H1*N_H2,  rng), np.zeros(N_H2,  np.float32),
        _xavier(N_H2,  N_H2*N_OUT, rng), np.zeros(N_OUT, np.float32),
    ])

class GA:
    def __init__(self) -> None:
        self.rng      = np.random.default_rng()
        self.gen      = 0
        self.pop      = [random_genome(self.rng) for _ in range(POP_SIZE)]
        self.fits     = [0.0] * POP_SIZE
        self.best_fit = 0.0
        self.best_w   = self.pop[0].copy()

    def record(self, idx: int, fit: float) -> None:
        self.fits[idx] = fit

    def evolve(self) -> None:
        fits    = np.array(self.fits, np.float32)
        best_i  = int(np.argmax(fits))
        best_v  = float(fits[best_i])
        avg_v   = float(np.mean(fits))
        if best_v > self.best_fit:
            self.best_fit = best_v
            self.best_w   = self.pop[best_i].copy()
        print(f"Gen {self.gen:4d} | best {best_v:.2f}m | avg {avg_v:.2f}m | all-time {self.best_fit:.2f}m")

        self.gen += 1
        elite   = list(np.argsort(fits)[-ELITE_N:])
        new_pop = [self.pop[i].copy() for i in elite]

        while len(new_pop) < POP_SIZE:
            def _tourn() -> np.ndarray:
                c = self.rng.choice(POP_SIZE, TOURN_K, replace=False)
                return self.pop[c[int(np.argmax(fits[c]))]]
            pa, pb = _tourn(), _tourn()
            mask  = self.rng.random(N_W) < CROSS_RATE
            child = np.where(mask, pa, pb).astype(np.float32)
            mut   = self.rng.random(N_W) < MUT_RATE
            if mut.any():
                child[mut] += self.rng.standard_normal(int(mut.sum())).astype(np.float32) * MUT_STD
            new_pop.append(child)

        self.pop  = new_pop
        self.fits = [0.0] * POP_SIZE


# ──────────────────────────────────────────────────────────
# Walker body
# ──────────────────────────────────────────────────────────

def _build_shapes(client: int) -> dict:
    def col(geom, **kw):
        return p.createCollisionShape(geom, physicsClientId=client, **kw)
    def vis(geom, color, **kw):
        return p.createVisualShape(geom, rgbaColor=color, physicsClientId=client, **kw)

    LT = [0.90,0.30,0.28,1.0]
    LS = [0.92,0.50,0.28,1.0]
    LF = [0.94,0.75,0.35,1.0]
    LA = [0.88,0.30,0.28,1.0]
    LFA= [0.90,0.50,0.28,1.0]
    RT = [0.25,0.48,0.92,1.0]
    RS = [0.28,0.68,0.92,1.0]
    RF = [0.35,0.85,0.95,1.0]
    RA = [0.25,0.48,0.90,1.0]
    RFA= [0.28,0.68,0.90,1.0]
    SP = [0.76,0.82,0.96,1.0]
    SN = [0.94,0.82,0.70,1.0]

    return {
        "base_col":    col(p.GEOM_BOX,      halfExtents=[0.09, 0.12, 0.095]),
        "base_vis":    vis(p.GEOM_BOX,      SP, halfExtents=[0.09,0.12,0.095]),
        "thigh_col":   col(p.GEOM_CYLINDER, radius=0.045, height=0.38),
        "thigh_vis_l": vis(p.GEOM_CYLINDER, LT, radius=0.045, length=0.38),
        "thigh_vis_r": vis(p.GEOM_CYLINDER, RT, radius=0.045, length=0.38),
        "shin_col":    col(p.GEOM_CYLINDER, radius=0.036, height=0.35),
        "shin_vis_l":  vis(p.GEOM_CYLINDER, LS, radius=0.036, length=0.35),
        "shin_vis_r":  vis(p.GEOM_CYLINDER, RS, radius=0.036, length=0.35),
        "foot_col":    col(p.GEOM_BOX,      halfExtents=[0.13, 0.05, 0.038]),
        "foot_vis_l":  vis(p.GEOM_BOX,      LF, halfExtents=[0.13,0.05,0.038]),
        "foot_vis_r":  vis(p.GEOM_BOX,      RF, halfExtents=[0.13,0.05,0.038]),
        "chest_col":   col(p.GEOM_BOX,      halfExtents=[0.09, 0.14, 0.155]),
        "chest_vis":   vis(p.GEOM_BOX,      SP, halfExtents=[0.09,0.14,0.155]),
        "neck_col":    col(p.GEOM_CYLINDER, radius=0.040, height=0.10),
        "neck_vis":    vis(p.GEOM_CYLINDER, SN, radius=0.040, length=0.10),
        "head_col":    col(p.GEOM_SPHERE,   radius=0.105),
        "head_vis":    vis(p.GEOM_SPHERE,   SN, radius=0.105),
        "uarm_col":    col(p.GEOM_CYLINDER, radius=0.030, height=0.25),
        "uarm_vis_l":  vis(p.GEOM_CYLINDER, LA,  radius=0.030, length=0.25),
        "uarm_vis_r":  vis(p.GEOM_CYLINDER, RA,  radius=0.030, length=0.25),
        "farm_col":    col(p.GEOM_CYLINDER, radius=0.024, height=0.22),
        "farm_vis_l":  vis(p.GEOM_CYLINDER, LFA, radius=0.024, length=0.22),
        "farm_vis_r":  vis(p.GEOM_CYLINDER, RFA, radius=0.024, length=0.22),
    }


@dataclass
class Walker:
    client: int
    genome: np.ndarray
    lane_y: float
    shapes: dict

    net: Net = field(init=False)
    phase: float = field(default=0.0, init=False)
    alive: bool = field(default=True, init=False)
    fit: float = field(default=0.0, init=False)
    reward_acc: float = field(default=0.0, init=False)
    last_x: float = field(default=0.0, init=False)
    air_time: float = field(default=0.0, init=False)
    lived_time: float = field(default=0.0, init=False)
    step_i: int = field(default=0, init=False)
    target_cache: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.3, 0.0, 0.0, 0.3, 0.0], dtype=np.float32), init=False)
    start_pos: tuple = field(default=None, init=False)
    body: int = field(default=-1, init=False)
    jids: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.net       = Net(self.genome)
        self.start_pos = (0.0, self.lane_y, 0.92)
        self.body      = self._build()
        self.jids      = {n: i for i, n in enumerate(ALL_JOINTS)}

    def _build(self) -> int:
        s = self.shapes
        body = p.createMultiBody(
            baseMass=12.0,
            baseCollisionShapeIndex=s["base_col"],
            baseVisualShapeIndex=s["base_vis"],
            basePosition=[0.0, self.lane_y, 0.92],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            linkMasses=[
                7.0, 3.5, 1.2,
                7.0, 3.5, 1.2,
                22.0, 1.0, 5.0,
                2.2, 1.8,
                2.2, 1.8,
            ],
            linkCollisionShapeIndices=[
                s["thigh_col"], s["shin_col"],  s["foot_col"],
                s["thigh_col"], s["shin_col"],  s["foot_col"],
                s["chest_col"], s["neck_col"],  s["head_col"],
                s["uarm_col"],  s["farm_col"],
                s["uarm_col"],  s["farm_col"],
            ],
            linkVisualShapeIndices=[
                s["thigh_vis_l"], s["shin_vis_l"], s["foot_vis_l"],
                s["thigh_vis_r"], s["shin_vis_r"], s["foot_vis_r"],
                s["chest_vis"],   s["neck_vis"],   s["head_vis"],
                s["uarm_vis_l"],  s["farm_vis_l"],
                s["uarm_vis_r"],  s["farm_vis_r"],
            ],
            linkPositions=[
                [ 0.00, +0.11, -0.095],
                [ 0.00,  0.00, -0.380],
                [ 0.00,  0.00, -0.350],
                [ 0.00, -0.11, -0.095],
                [ 0.00,  0.00, -0.380],
                [ 0.00,  0.00, -0.350],
                [ 0.00,  0.00, +0.095],
                [ 0.00,  0.00, +0.310],
                [ 0.00,  0.00, +0.100],
                [ 0.00, +0.14, +0.140],
                [ 0.00,  0.00, -0.250],
                [ 0.00, -0.14, +0.140],
                [ 0.00,  0.00, -0.250],
            ],
            linkOrientations=[[0,0,0,1]]*13,
            linkInertialFramePositions=[
                [ 0.00,  0.00, -0.190],
                [ 0.00,  0.00, -0.175],
                [ 0.09,  0.00, -0.038],
                [ 0.00,  0.00, -0.190],
                [ 0.00,  0.00, -0.175],
                [ 0.09,  0.00, -0.038],
                [ 0.00,  0.00, +0.155],
                [ 0.00,  0.00, +0.050],
                [ 0.00,  0.00, +0.105],
                [ 0.00,  0.00, -0.125],
                [ 0.00,  0.00, -0.110],
                [ 0.00,  0.00, -0.125],
                [ 0.00,  0.00, -0.110],
            ],
            linkInertialFrameOrientations=[[0,0,0,1]]*13,
            linkParentIndices=[
                0, 1, 2, 0, 4, 5,
                0, 7, 8,
                7, 10, 7, 12,
            ],
            linkJointTypes=(
                [p.JOINT_REVOLUTE]*6
                + [p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED]
                + [p.JOINT_REVOLUTE]*4
            ),
            linkJointAxis=[[0,1,0]]*6 + [[1,0,0]]*3 + [[0,1,0]]*4,
            physicsClientId=self.client,
        )

        p.changeDynamics(body, -1,
                         lateralFriction=1.8, restitution=0.0,
                         linearDamping=0.06, angularDamping=0.08,
                         physicsClientId=self.client)
        for j in range(13):
            p.changeDynamics(body, j,
                             lateralFriction=1.8,
                             restitution=0.0,
                             linearDamping=0.06,
                             angularDamping=0.08,
                             maxJointVelocity=6.0,
                             physicsClientId=self.client)
            if j not in FIXED_JOINT_IDX:
                p.setJointMotorControl2(body, j, p.VELOCITY_CONTROL,
                                        targetVelocity=0.0, force=0.0,
                                        physicsClientId=self.client)

        for jidx, lo, hi in [
            (0,  *JOINT_LIMITS["l_hip"]),    (1,  *JOINT_LIMITS["l_knee"]),
            (2,  *JOINT_LIMITS["l_ankle"]),  (3,  *JOINT_LIMITS["r_hip"]),
            (4,  *JOINT_LIMITS["r_knee"]),   (5,  *JOINT_LIMITS["r_ankle"]),
            (9,  *ARM_LIMITS["l_shoulder"]), (10, *ARM_LIMITS["l_elbow"]),
            (11, *ARM_LIMITS["r_shoulder"]), (12, *ARM_LIMITS["r_elbow"]),
        ]:
            p.changeDynamics(body, jidx,
                             jointLowerLimit=lo, jointUpperLimit=hi,
                             physicsClientId=self.client)

        for jidx, angle in [(1, 0.3), (4, 0.3), (10, 0.3), (12, 0.3)]:
            p.resetJointState(body, jidx, angle, physicsClientId=self.client)

        return body

    def highlight(self, on: bool) -> None:
        if on:
            base_c  = [1.0, 0.95, 0.20, 1.0]
            link_cs = [[1.0,0.68,0.20,1.0],[1.0,0.72,0.30,1.0],[1.0,0.78,0.40,1.0],
                       [1.0,0.68,0.20,1.0],[1.0,0.72,0.30,1.0],[1.0,0.78,0.40,1.0],
                       [1.0,0.95,0.20,1.0],[1.0,0.90,0.55,1.0],[1.0,0.88,0.60,1.0],
                       [1.0,0.68,0.20,1.0],[1.0,0.72,0.30,1.0],
                       [1.0,0.68,0.20,1.0],[1.0,0.72,0.30,1.0]]
        else:
            base_c  = [0.76, 0.82, 0.96, 1.0]
            link_cs = [[0.90,0.30,0.28,1.0],[0.92,0.50,0.28,1.0],[0.94,0.75,0.35,1.0],
                       [0.25,0.48,0.92,1.0],[0.28,0.68,0.92,1.0],[0.35,0.85,0.95,1.0],
                       [0.76,0.82,0.96,1.0],[0.94,0.82,0.70,1.0],[0.94,0.82,0.70,1.0],
                       [0.88,0.30,0.28,1.0],[0.90,0.50,0.28,1.0],
                       [0.25,0.48,0.90,1.0],[0.28,0.68,0.90,1.0]]
        p.changeVisualShape(self.body, -1, rgbaColor=base_c, physicsClientId=self.client)
        for i, c in enumerate(link_cs):
            p.changeVisualShape(self.body, i, rgbaColor=c, physicsClientId=self.client)

    def _contacts(self) -> tuple[float, float]:
        # Foot links are l_ankle=2 and r_ankle=5 because those joints create the foot links.
        lc = 1.0 if p.getContactPoints(bodyA=self.body, linkIndexA=self.jids["l_ankle"],
                                        physicsClientId=self.client) else 0.0
        rc = 1.0 if p.getContactPoints(bodyA=self.body, linkIndexA=self.jids["r_ankle"],
                                        physicsClientId=self.client) else 0.0
        return lc, rc

    def step(self) -> None:
        if not self.alive:
            return

        obs, m = self._observe()
        self.step_i += 1

        # Update the neural action at 30 Hz and smooth it.
        if self.step_i % ACTION_EVERY == 1:
            raw = self.net.forward(obs)
            new_targets = np.array([
                lo + 0.5*(float(raw[i])+1.0)*(hi-lo)
                for i, (lo, hi) in enumerate(JOINT_LIMITS[n] for n in JOINT_NAMES)
            ], dtype=np.float32)
            self.target_cache = (
                (1.0 - TARGET_SMOOTH) * self.target_cache
                + TARGET_SMOOTH * new_targets
            )

        p.setJointMotorControlArray(
            self.body, list(range(6)),
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.target_cache.tolist(),
            forces=[MOTOR_TMAX]*6,
            positionGains=[MOTOR_POS_GAIN]*6,
            velocityGains=[MOTOR_VEL_GAIN]*6,
            physicsClientId=self.client,
        )

        # Passive arm swing. It is deliberately weak so it does not throw the body.
        sw = 0.32 * math.sin(self.phase)
        l_sw = max(ARM_LIMITS["l_shoulder"][0], min(ARM_LIMITS["l_shoulder"][1],  sw))
        r_sw = max(ARM_LIMITS["r_shoulder"][0], min(ARM_LIMITS["r_shoulder"][1], -sw))
        p.setJointMotorControlArray(
            self.body,
            [self.jids["l_shoulder"], self.jids["l_elbow"],
             self.jids["r_shoulder"], self.jids["r_elbow"]],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[l_sw, 0.35, r_sw, 0.35],
            forces=[ARM_TMAX]*4,
            positionGains=[ARM_POS_GAIN]*4,
            velocityGains=[ARM_VEL_GAIN]*4,
            physicsClientId=self.client,
        )

        self._accumulate_reward(m)

    def _observe(self) -> tuple[np.ndarray, dict]:
        pos, quat = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        lin, ang  = p.getBaseVelocity(self.body, physicsClientId=self.client)
        roll, pitch, _ = p.getEulerFromQuaternion(quat)
        angs, vels = [], []
        for n in JOINT_NAMES:
            st = p.getJointState(self.body, self.jids[n], physicsClientId=self.client)
            angs.append(float(st[0]))
            vels.append(float(st[1]))
        lc, rc = self._contacts()

        grounded = (lc > 0.0 or rc > 0.0)
        self.air_time = 0.0 if grounded else self.air_time + PHYS_DT
        self.phase += PHYS_DT * 1.8

        # Keep input count at 20.
        obs = np.array([float(pos[2]), float(pitch), float(lin[0]), float(lin[2]),
                        *angs, *vels, lc, rc, math.sin(self.phase), math.cos(self.phase)],
                       dtype=np.float32)
        return obs, {
            "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "roll": float(roll), "pitch": float(pitch),
            "vx": float(lin[0]), "vz": float(lin[2]),
            "ang_speed": float(math.sqrt(ang[0]*ang[0] + ang[1]*ang[1] + ang[2]*ang[2])),
            "angs": angs, "vels": vels,
            "lc": lc, "rc": rc,
            "grounded": 1.0 if grounded else 0.0,
        }

    def _accumulate_reward(self, m: dict) -> None:
        dx = m["x"] - self.last_x
        self.last_x = m["x"]

        # Do not let launch/explosion earn huge progress in one step.
        forward = max(-0.02, min(0.025, dx))

        upright = max(0.0, 1.0 - (abs(m["roll"]) + abs(m["pitch"])) / 1.5)
        h_stab  = max(0.0, 1.0 - abs(m["z"] - 0.87) * 2.5)
        grounded = m["grounded"]

        a, v = m["angs"], m["vels"]
        sym_pen = abs(a[0] + a[3]) + 0.4*abs(a[1] - a[4]) + 0.3*abs(a[2] - a[5])
        energy = sum(abs(vi) for vi in v)

        # Movement should come from joint usage, but only while grounded and upright.
        hip_swing  = (abs(v[0]) + abs(v[3])) * 0.5
        knee_swing = (abs(v[1]) + abs(v[4])) * 0.5
        useful_joint_work = min(4.0, 0.7 * hip_swing + 0.3 * knee_swing)

        # Prefer one or two feet on ground. Penalize airtime instead of rewarding jumps.
        contact_bonus = 1.0 if grounded else 0.0
        air_pen = min(1.0, self.air_time / MAX_AIR_TIME)

        self.reward_acc += (
            8.0  * max(0.0, forward) * grounded * upright
            + 0.020 * upright
            + 0.012 * h_stab
            + 0.010 * contact_bonus
            + 0.025 * useful_joint_work * max(0.0, forward) * grounded
            - 0.006 * abs(m["y"] - self.lane_y)
            - 0.003 * energy
            - 0.006 * sym_pen
            - 0.012 * m["ang_speed"]
            - 0.050 * air_pen
        )

    def check_alive(self, t: float) -> None:
        if not self.alive:
            return
        self.lived_time = t
        pos, quat = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        roll, pitch, _ = p.getEulerFromQuaternion(quat)

        fallen = (
            pos[2] < FALL_Z
            or pos[2] > MAX_Z
            or abs(roll) > FALL_ROLL
            or abs(pitch) > FALL_PITCH
            or self.air_time > MAX_AIR_TIME
        )
        if fallen or t >= SIM_TIME:
            self.alive = False
            self.fit = self._fitness(fell=fallen)

    def _fitness(self, fell: bool = False) -> float:
        pos, quat = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        roll, pitch, _ = p.getEulerFromQuaternion(quat)
        upright = max(0.0, 1.0 - (abs(roll) + abs(pitch)) / 1.5)
        distance = max(0.0, float(pos[0]))
        survival = self.lived_time / SIM_TIME
        fall_penalty = 1.5 if fell else 0.0

        return max(0.0, distance + 0.28 * max(0.0, self.reward_acc) + 0.5 * survival * upright - fall_penalty)

    def final_fitness(self) -> float:
        if self.alive:
            self.fit = self._fitness(fell=False)
        return self.fit

    def remove(self) -> None:
        p.removeBody(self.body, physicsClientId=self.client)


# ──────────────────────────────────────────────────────────
# Simulation
# ──────────────────────────────────────────────────────────

class Sim:
    def __init__(self, gui: bool, load: bool = True) -> None:
        self.gui    = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0.0, 0.0, -GRAVITY, physicsClientId=self.client)
        p.setTimeStep(PHYS_DT, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=120, numSubSteps=1,
                                    physicsClientId=self.client)

        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(self.plane, -1,
                         lateralFriction=2.2,
                         spinningFriction=0.4,
                         rollingFriction=0.1,
                         restitution=0.0,
                         physicsClientId=self.client)

        if gui:
            # Fast viewport-only mode. The PyBullet sidebar/debug UI is disabled
            # because debug sliders/text are expensive and slow rendering.
            for flag in (p.COV_ENABLE_GUI,
                         p.COV_ENABLE_SHADOWS,
                         p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                         p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                         p.COV_ENABLE_RGB_BUFFER_PREVIEW):
                p.configureDebugVisualizer(flag, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(
                cameraDistance=INIT_DIST,
                cameraYaw=INIT_YAW,
                cameraPitch=INIT_PITCH,
                cameraTargetPosition=INIT_TARGET,
                physicsClientId=self.client,
            )

        self.ga      = GA()
        self.shapes  = _build_shapes(self.client)
        self.walkers : list[Walker] = []

        self._cam_yaw   = INIT_YAW
        self._cam_pitch = INIT_PITCH
        self._cam_dist  = INIT_DIST
        self._cam_dirty = False

        self._sel    = 0
        self._follow = FOLLOW_DEFAULT

        # No debug text, debug lines, or sidebar sliders. Terminal-only status.
        self._last_status_print = 0.0

        self._spawn()
        if load:
            self._load_checkpoint()

    def _spawn(self) -> None:
        for w in self.walkers:
            w.remove()
        self.walkers.clear()
        spacing = 1.8
        offset  = 0.5 * (POP_SIZE - 1) * spacing
        for i, genome in enumerate(self.ga.pop):
            self.walkers.append(Walker(self.client, genome, i*spacing - offset, self.shapes))
        self._sel = 0
        self._refresh_highlights()

    def _refresh_highlights(self) -> None:
        for i, w in enumerate(self.walkers):
            w.highlight(i == self._sel)

    def _best_alive_idx(self) -> int:
        alive = [i for i, w in enumerate(self.walkers) if w.alive]
        pool  = alive if alive else list(range(len(self.walkers)))
        return max(pool, key=lambda i: self.walkers[i].last_x)

    def _save_checkpoint(self) -> None:
        np.savez(
            SAVE_FILE,
            gen=np.array(self.ga.gen),
            best_fit=np.array(self.ga.best_fit),
            best_w=self.ga.best_w,
            pop=np.array(self.ga.pop),
        )
        print(f"  [saved] gen={self.ga.gen} best={self.ga.best_fit:.2f}m -> {SAVE_FILE}")

    def _load_checkpoint(self) -> None:
        if not os.path.exists(SAVE_FILE):
            return
        try:
            data             = np.load(SAVE_FILE)
            self.ga.gen      = int(data["gen"])
            self.ga.best_fit = float(data["best_fit"])
            self.ga.best_w   = data["best_w"].copy()
            loaded_pop       = list(data["pop"])
            if len(loaded_pop) == POP_SIZE:
                self.ga.pop  = loaded_pop
            self.ga.fits     = [0.0] * POP_SIZE
            print(f"  [loaded] gen={self.ga.gen} best={self.ga.best_fit:.2f}m <- {SAVE_FILE}")
        except Exception as e:
            print(f"  [checkpoint] could not load '{SAVE_FILE}': {e}")

    def _poll_keys(self) -> None:
        ev = p.getKeyboardEvents(physicsClientId=self.client)
        if not ev:
            return

        cam_changed = False
        if p.B3G_LEFT_ARROW  in ev and ev[p.B3G_LEFT_ARROW]  & p.KEY_IS_DOWN:
            self._cam_yaw  -= STEP_YAW; cam_changed = True
        if p.B3G_RIGHT_ARROW in ev and ev[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            self._cam_yaw  += STEP_YAW; cam_changed = True
        if p.B3G_UP_ARROW    in ev and ev[p.B3G_UP_ARROW]    & p.KEY_IS_DOWN:
            self._cam_pitch  = min(5.0,  self._cam_pitch + STEP_PITCH); cam_changed = True
        if p.B3G_DOWN_ARROW  in ev and ev[p.B3G_DOWN_ARROW]  & p.KEY_IS_DOWN:
            self._cam_pitch  = max(-85.0,self._cam_pitch - STEP_PITCH); cam_changed = True

        # Fixed comment/behavior: Q zooms out, E zooms in.
        if ord('q') in ev and ev[ord('q')] & p.KEY_IS_DOWN:
            self._cam_dist   = min(60.0, self._cam_dist + STEP_DIST); cam_changed = True
        if ord('e') in ev and ev[ord('e')] & p.KEY_IS_DOWN:
            self._cam_dist   = max(3.5,  self._cam_dist - STEP_DIST); cam_changed = True
        if cam_changed:
            self._cam_dirty = True

        if ord('a') in ev and ev[ord('a')] & p.KEY_WAS_TRIGGERED:
            self._sel = (self._sel - 1) % len(self.walkers)
            self._refresh_highlights()
        if ord('d') in ev and ev[ord('d')] & p.KEY_WAS_TRIGGERED:
            self._sel = (self._sel + 1) % len(self.walkers)
            self._refresh_highlights()
        if ord('f') in ev and ev[ord('f')] & p.KEY_WAS_TRIGGERED:
            self._follow    = not self._follow
            self._cam_dirty = True

    def _update_camera(self) -> None:
        cam         = p.getDebugVisualizerCamera(physicsClientId=self.client)
        live_yaw    = float(cam[8])
        live_pitch  = float(cam[9])
        live_dist   = float(cam[10])
        live_target = [float(cam[11][0]), float(cam[11][1]), float(cam[11][2])]

        sel_pos, _ = p.getBasePositionAndOrientation(
            self.walkers[self._sel].body, physicsClientId=self.client)

        if self._cam_dirty:
            target = ([float(sel_pos[0]), float(sel_pos[1]), 0.9]
                      if self._follow else live_target)
            p.resetDebugVisualizerCamera(
                cameraDistance=self._cam_dist,
                cameraYaw=self._cam_yaw,
                cameraPitch=self._cam_pitch,
                cameraTargetPosition=target,
                physicsClientId=self.client,
            )
            self._cam_dirty = False

        elif self._follow:
            alpha = 0.06
            new_target = [
                live_target[0] + (float(sel_pos[0]) - live_target[0]) * alpha,
                live_target[1] + (float(sel_pos[1]) - live_target[1]) * alpha,
                0.9,
            ]
            p.resetDebugVisualizerCamera(
                cameraDistance=live_dist,
                cameraYaw=live_yaw,
                cameraPitch=live_pitch,
                cameraTargetPosition=new_target,
                physicsClientId=self.client,
            )

        self._cam_yaw, self._cam_pitch, self._cam_dist = live_yaw, live_pitch, live_dist

    def _print_status(self, t: float, force: bool = False) -> None:
        """Cheap terminal status only. No PyBullet debug items are created."""
        if not force:
            now = time.time()
            if now - self._last_status_print < STATUS_INTERVAL:
                return
            self._last_status_print = now
        alive = sum(1 for ww in self.walkers if ww.alive)
        live_best_x = max((ww.last_x for ww in self.walkers), default=0.0)
        sel_x = 0.0
        if self.walkers:
            pos, _ = p.getBasePositionAndOrientation(self.walkers[self._sel].body,
                                                     physicsClientId=self.client)
            sel_x = float(pos[0])
        print(
            f"\rGen {self.ga.gen} | t={t:4.1f}s | alive={alive:2d}/{POP_SIZE} | "
            f"sel={self._sel:02d} x={sel_x:6.2f}m | live_best_x={live_best_x:6.2f}m | "
            f"all_time={self.ga.best_fit:6.2f}m",
            end='',
            flush=True,
        )

    def run_generation(self) -> None:
        total = int(SIM_TIME / PHYS_DT)
        t     = 0.0

        for step in range(total):
            if self.gui:
                self._poll_keys()
                self._update_camera()

            for w in self.walkers:
                w.step()

            p.stepSimulation(physicsClientId=self.client)
            t = (step + 1) * PHYS_DT

            for w in self.walkers:
                w.check_alive(t)

            if self.gui and step % max(1, int(STATUS_INTERVAL / PHYS_DT)) == 0:
                if self._follow:
                    self._sel = self._best_alive_idx()
                    self._refresh_highlights()
                self._print_status(t)

            if not GUI_FAST and self.gui:
                time.sleep(PHYS_DT / max(1.0, GUI_SPEED))

            if not any(w.alive for w in self.walkers):
                break

        for i, w in enumerate(self.walkers):
            self.ga.record(i, w.final_fitness())

        if self.gui:
            self._print_status(t, force=True)
            print()
            if POST_GEN_IDLE > 0.0:
                self.idle_gui(POST_GEN_IDLE, t)

        self.ga.evolve()
        self._save_checkpoint()
        self._spawn()

    def idle_gui(self, seconds: float | None = None, t: float = 0.0) -> None:
        """Keep the PyBullet GUI responsive when physics is not advancing.

        PyBullet keyboard events only update when the app keeps polling them.
        This lets arrow/Q/E/A/D/F camera and selection controls work after a
        finite `--gens` run, or during an optional between-generation pause.

        Press Ctrl+C in the terminal to exit.
        """
        if not self.gui:
            return
        start = time.time()
        while seconds is None or (time.time() - start) < seconds:
            self._poll_keys()
            self._update_camera()
            time.sleep(1.0 / 60.0)

    def close(self) -> None:
        if self.client >= 0:
            p.disconnect(self.client)


def main() -> None:
    ap = argparse.ArgumentParser(description="Genetic Walker — PyBullet fast viewport")
    ap.add_argument("--direct",  action="store_true", help="Run headless (no GUI)")
    ap.add_argument("--gens",    type=int, default=0,  help="Generations to run (0=infinite)")
    ap.add_argument("--no-load", action="store_true",  help="Ignore checkpoint, start fresh")
    args = ap.parse_args()

    sim = Sim(gui=not args.direct, load=not args.no_load)
    print("Genetic Walker — PyBullet fast viewport")
    print(f"Pop={POP_SIZE} NN={N_IN}->{N_H1}->{N_H2}->{N_OUT} weights={N_W}")
    if args.direct:
        print("Mode: DIRECT/headless")
    else:
        print("Mode: GUI — mouse drag=rotate Shift+drag=translate scroll=zoom")

    try:
        gen = 0
        while args.gens == 0 or gen < args.gens:
            sim.run_generation()
            gen += 1

        # If the user asked for a finite number of GUI generations, keep the
        # window alive and keep polling keyboard/camera controls instead of
        # immediately disconnecting. This fixes controls only working while the
        # simulation loop is actively running.
        if not args.direct and args.gens > 0:
            print("Finished requested generations. GUI is in idle mode; press Ctrl+C in terminal to exit.")
            sim.idle_gui(None, 0.0)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()
