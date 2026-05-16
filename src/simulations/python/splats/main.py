from __future__ import annotations

import argparse
import time

import numpy as np
import pygame

from animation import JOINT_INDEX, JOINT_NAMES, PARENTS, SkeletalAnimator
from balance_system import BalanceSystemController
from deformation_ai import NeuralDeformationController
from foot_planting import FootPlantingController
from human_generator import attach_face_splats, generate_human_particles
from ik_solver import solve_limb_ik
from joint_dynamics import JointDynamics
from motion_physics_controller import MotionPhysicsController
from motion_ai import MotionAIController, PROMPT_TOKENS
from renderer import Camera, GaussianSplatRenderer, look_at, perspective
from root_motion import RootMotionController
from secondary_motion import SecondaryMotionController
from skinning import (
    apply_skinning,
    apply_skinning_matrix_palette,
    build_matrix_palette,
    dominant_bone_debug_colors,
    dominant_weight_debug_colors,
    get_rest_joint_world_positions,
)


# ---------------------------------------------------------------------------
# 3DGS PLY loader
# ---------------------------------------------------------------------------

def load_ply_splats(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a standard 3DGS PLY file (binary little-endian).

    Returns:
        positions  (N, 3) float32 – world-space Gaussian centres
        colors     (N, 3) float32 – RGB in [0, 1] from DC SH coefficients
        sizes      (N,)   float32 – max isotropic scale in world units
        opacities  (N,)   float32 – per-Gaussian opacity in [0, 1]
    """
    with open(path, "rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_verts = 0
        props: list[str] = []
        for line in header_lines:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                n_verts = int(parts[2])
            elif parts[:2] == ["property", "float"]:
                props.append(parts[2])

        raw = np.frombuffer(
            f.read(n_verts * len(props) * 4), dtype="<f4"
        ).reshape(n_verts, len(props))

    idx = {name: i for i, name in enumerate(props)}

    positions = raw[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    # 3DGS captures often use Y-down (OpenCV convention). Flip Y so the scene
    # sits upright in the renderer's Y-up world.
    positions[:, 1] *= -1.0

    log_scales = raw[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]]
    sizes = np.exp(log_scales).max(axis=1).astype(np.float32)

    # DC SH → RGB: color = 0.5 + (1/sqrt(4π)) * f_dc
    _C0 = 0.28209479177387814
    colors = np.clip(
        0.5 + _C0 * raw[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]],
        0.0,
        1.0,
    ).astype(np.float32)

    # Logit → sigmoid probability
    opacities = (1.0 / (1.0 + np.exp(-raw[:, idx["opacity"]]))).astype(np.float32)

    return positions, colors, sizes, opacities


def _fit_skeleton_to_ply(positions: np.ndarray) -> np.ndarray:
    """
    Detect body-part positions from a 3DGS point cloud and fit a humanoid
    skeleton.

    Pipeline
    --------
    1. Robust Y-bounds give height H.
    2. PCA on the XZ plane finds the TRUE lateral axis (resolves 90° rotation).
    3. Dense width profile (120 Y-slices, smoothed) is read to find:
         shoulder peak → neck valley → head → waist valley → chest → pelvis peak → spine
    4. Per-side arm tracing along the lateral axis finds elbow and hand.
    5. All 3-D positions are reconstructed via the (lat, dep) coordinate frame.
    """
    pts = positions.copy()

    # ── 1. Vertical extent ──────────────────────────────────────────────────
    y_lo = float(np.percentile(pts[:, 1], 2))
    y_hi = float(np.percentile(pts[:, 1], 98))
    H = max(y_hi - y_lo, 0.01)
    pts = pts[(pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi)]

    cx = float(np.median(pts[:, 0]))
    cz = float(np.median(pts[:, 2]))

    # ── 2. PCA lateral-axis detection ───────────────────────────────────────
    upper = pts[pts[:, 1] >= y_lo + 0.5 * H]
    xz = upper[:, [0, 2]] - np.array([[cx, cz]])
    cov = (xz.T @ xz) / max(len(xz) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    lat2d = eigvecs[:, np.argmax(eigvals)]   # shoulder-to-shoulder direction
    dep2d = eigvecs[:, np.argmin(eigvals)]   # front-back direction
    if lat2d[0] < 0:
        lat2d = -lat2d
    print(f"  Lateral axis (XZ): [{lat2d[0]:.3f}, {lat2d[1]:.3f}]")

    # ── 2b. Depth filter: remove background (keep ±2.2σ around body centre) ─
    all_deps = (pts[:, 0] - cx) * dep2d[0] + (pts[:, 2] - cz) * dep2d[1]
    dep_med  = float(np.median(all_deps))
    dep_std  = float(np.std(all_deps))
    pts = pts[np.abs(all_deps - dep_med) <= 2.2 * dep_std]
    print(f"  Depth filter: {len(pts)} pts kept")

    # ── 3. Projection / reconstruction helpers ───────────────────────────────
    def _lat(p: np.ndarray) -> np.ndarray:
        return (p[:, 0] - cx) * lat2d[0] + (p[:, 2] - cz) * lat2d[1]

    def _dep_mean(p: np.ndarray) -> float:
        return float(((p[:, 0] - cx) * dep2d[0] + (p[:, 2] - cz) * dep2d[1]).mean())

    def _to3d(lat: float, y: float, dep: float) -> np.ndarray:
        return np.array([cx + lat * lat2d[0] + dep * dep2d[0],
                         y,
                         cz + lat * lat2d[1] + dep * dep2d[1]], np.float32)

    # ── 4. Y-band helpers ────────────────────────────────────────────────────
    def _band(flo: float, fhi: float) -> np.ndarray:
        lo, hi = y_lo + flo * H, y_lo + fhi * H
        return pts[(pts[:, 1] >= lo) & (pts[:, 1] <= hi)]

    def _centroid_band(flo: float, fhi: float) -> np.ndarray:
        sub = _band(flo, fhi)
        if len(sub) < 3:
            return _to3d(0.0, y_lo + (flo + fhi) * 0.5 * H, 0.0)
        return sub.mean(0).astype(np.float32)

    def _bilateral(flo: float, fhi: float, side: str) -> np.ndarray:
        sub = _band(flo, fhi)
        mid_y = y_lo + (flo + fhi) * 0.5 * H
        if len(sub) < 3:
            offset = (-0.15 if side == "left" else 0.15) * H
            return _to3d(offset, mid_y, 0.0)
        lats = _lat(sub)
        med = float(np.median(lats))
        half = sub[lats <= med] if side == "left" else sub[lats > med]
        grp = half if len(half) >= 3 else sub
        return _to3d(float(_lat(grp).mean()), float(grp[:, 1].mean()), _dep_mean(grp))

    def _lateral_extreme(flo: float, fhi: float, side: str, pct: float = 14.0) -> np.ndarray:
        sub = _band(flo, fhi)
        mid_y = y_lo + (flo + fhi) * 0.5 * H
        if len(sub) < 5:
            offset = (-0.2 if side == "left" else 0.2) * H
            return _to3d(offset, mid_y, 0.0)
        lats = _lat(sub)
        if side == "left":
            thresh = float(np.percentile(lats, pct))
            cluster = sub[lats <= thresh]
        else:
            thresh = float(np.percentile(lats, 100.0 - pct))
            cluster = sub[lats >= thresh]
        grp = cluster if len(cluster) > 0 else sub
        return _to3d(float(_lat(grp).mean()), float(grp[:, 1].mean()), _dep_mean(grp))

    # ── 5. Dense width profile ───────────────────────────────────────────────
    N_PROF = 120
    prof_f = np.linspace(0.01, 0.99, N_PROF)   # Y-fraction for each slice
    half_df = 1.5 / N_PROF                       # half-slice height
    lat_w = np.zeros(N_PROF)                     # lateral width at each slice

    for i, f in enumerate(prof_f):
        sub = _band(f - half_df, f + half_df)
        if len(sub) < 10:
            continue
        lats = _lat(sub)
        lat_w[i] = float(np.percentile(lats, 88) - np.percentile(lats, 12))

    # Simple box-filter smoothing (no scipy needed)
    def _smooth(arr: np.ndarray, w: int = 7) -> np.ndarray:
        k = np.ones(w) / w
        return np.convolve(arr, k, mode="same")

    sw = _smooth(lat_w, w=7)

    # ── 6. Profile-based anatomical landmark detection ───────────────────────
    # Masked argmax/argmin: add large penalty outside region of interest
    def _masked_argmax(arr, mask):
        return int(np.argmax(arr * mask.astype(float)))

    def _masked_argmin(arr, mask):
        return int(np.argmin(arr + (~mask).astype(float) * 1e6))

    # Shoulder: maximum lateral width in upper body (55–85 % height)
    sh_mask = (prof_f >= 0.55) & (prof_f <= 0.85)
    sh_frac = float(prof_f[_masked_argmax(sw, sh_mask)])
    sh_frac = float(np.clip(sh_frac, 0.58, 0.82))

    # Neck: narrowest point in the anatomical range [sh+0.06, sh+0.22]
    # (neck can't be right at the shoulder or above 92% of height)
    nk_mask = (prof_f > sh_frac + 0.06) & (prof_f < min(sh_frac + 0.22, 0.92))
    if nk_mask.any():
        neck_frac = float(prof_f[_masked_argmin(sw, nk_mask)])
    else:
        neck_frac = sh_frac + 0.12   # proportional fallback

    # Head: WIDEST blob above neck (skull is wider than neck)
    hd_mask = (prof_f > neck_frac + 0.01) & (prof_f < 0.98)
    if hd_mask.any() and sw[hd_mask].max() > 0:
        head_frac = float(np.clip(float(prof_f[_masked_argmax(sw, hd_mask)]),
                                  neck_frac + 0.04, 0.97))
    else:
        head_frac = neck_frac + 0.09

    # Waist: narrowest point between shoulders and 0.30 H (below torso)
    wt_mask = (prof_f >= 0.28) & (prof_f < sh_frac - 0.04)
    if wt_mask.any():
        waist_frac = float(prof_f[_masked_argmin(sw, wt_mask)])
    else:
        waist_frac = sh_frac - 0.18

    # Chest: 60 % of the way from waist up to shoulder (upper torso bulk)
    chest_frac = float(waist_frac + (sh_frac - waist_frac) * 0.60)

    # Pelvis: widest point in mid-body (0.36 H → waist), hip bones bulge
    pv_mask = (prof_f >= 0.36) & (prof_f <= max(waist_frac + 0.04, 0.46))
    if pv_mask.any() and sw[pv_mask].max() > 0:
        pelvis_frac = float(prof_f[_masked_argmax(sw, pv_mask)])
    else:
        pelvis_frac = float(np.clip(waist_frac - 0.08, 0.36, 0.52))

    # Spine: midpoint between chest and pelvis
    spine_frac = (chest_frac + pelvis_frac) * 0.5

    print(f"  head={head_frac:.2f} neck={neck_frac:.2f} sh={sh_frac:.2f} "
          f"chest={chest_frac:.2f} waist={waist_frac:.2f} "
          f"pelvis={pelvis_frac:.2f}  (fracs of H)")

    # ── 7. Per-side arm tracing ──────────────────────────────────────────────
    def _trace_arm(side: str) -> tuple[float, float]:
        """
        Scan downward starting 8% BELOW shoulder to avoid treating the
        shoulder itself as the elbow. Elbow = Y where lateral distance
        from body centre is maximum.
        """
        ext_lats: list[float] = []
        y_fracs: list[float] = []
        n_steps = 40
        arm_start = sh_frac - 0.08          # skip shoulder region
        arm_end   = max(pelvis_frac - 0.06, 0.22)   # don't scan below hips
        for i in range(n_steps):
            frac = arm_start - i * (arm_start - arm_end) / n_steps
            if frac < arm_end:
                break
            sub = _band(frac, frac + (arm_start - arm_end) / n_steps)
            if len(sub) < 5:
                continue
            lats = _lat(sub)
            pct = 14.0
            if side == "left":
                thresh = float(np.percentile(lats, pct))
                ext = sub[lats <= thresh]
            else:
                thresh = float(np.percentile(lats, 100.0 - pct))
                ext = sub[lats >= thresh]
            if len(ext) < 2:
                continue
            ext_lats.append(float(_lat(ext).mean()))
            y_fracs.append(frac)

        if len(ext_lats) < 4:
            return max(sh_frac - 0.18, 0.35), max(sh_frac - 0.34, 0.18)

        dists = np.abs(np.array(ext_lats))
        raw_elbow = float(y_fracs[int(np.argmax(dists))])
        # Clamp elbow to anatomically valid range: [pelvis, sh-0.10]
        elbow_frac = float(np.clip(raw_elbow, pelvis_frac - 0.02, sh_frac - 0.10))
        arm_gap = max(sh_frac - elbow_frac, 0.10)
        hand_frac = float(np.clip(elbow_frac - arm_gap * 0.85, 0.04, elbow_frac - 0.08))
        return elbow_frac, hand_frac

    elbow_l, hand_l = _trace_arm("left")
    elbow_r, hand_r = _trace_arm("right")
    print(f"  elbow L={elbow_l:.2f} R={elbow_r:.2f} | hand L={hand_l:.2f} R={hand_r:.2f}  (fracs of H)")

    # ── 8. Build joint map ───────────────────────────────────────────────────
    dj = 0.03   # half-band for point joints
    J: dict[str, np.ndarray] = {
        "head":       _centroid_band(neck_frac + 0.005, min(head_frac + 0.04, 0.99)),
        "neck":       _centroid_band(max(neck_frac - 0.025, 0.0), neck_frac + 0.025),
        "chest":      _centroid_band(chest_frac - dj, chest_frac + dj),
        "spine":      _centroid_band(spine_frac - dj, spine_frac + dj),
        "pelvis":     _centroid_band(pelvis_frac - dj, pelvis_frac + dj),
        "shoulder_l": _lateral_extreme(sh_frac - dj, sh_frac + dj, "left",  12.0),
        "shoulder_r": _lateral_extreme(sh_frac - dj, sh_frac + dj, "right", 12.0),
        "elbow_l":    _lateral_extreme(elbow_l - dj, elbow_l + dj, "left",  14.0),
        "elbow_r":    _lateral_extreme(elbow_r - dj, elbow_r + dj, "right", 14.0),
        "hand_l":     _lateral_extreme(hand_l  - dj, hand_l  + dj, "left",  14.0),
        "hand_r":     _lateral_extreme(hand_r  - dj, hand_r  + dj, "right", 14.0),
        "hip_l":      _bilateral(pelvis_frac - dj, pelvis_frac + dj, "left"),
        "hip_r":      _bilateral(pelvis_frac - dj, pelvis_frac + dj, "right"),
        "knee_l":     _bilateral(0.22, 0.32, "left"),
        "knee_r":     _bilateral(0.22, 0.32, "right"),
        "foot_l":     _lateral_extreme(0.00, 0.08, "left",  28.0),
        "foot_r":     _lateral_extreme(0.00, 0.08, "right", 28.0),
    }

    # ── 9. Pack into JOINT_NAMES-ordered array ───────────────────────────────
    rest = get_rest_joint_world_positions()
    ref_lo = float(rest[:, 1].min())
    ref_hi = float(rest[:, 1].max())
    sc = H / max(ref_hi - ref_lo, 0.01)

    fitted = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)
    for name in JOINT_NAMES:
        i = JOINT_INDEX[name]
        if name in J:
            fitted[i] = J[name]
        else:
            fitted[i] = _to3d(rest[i, 0] * sc,
                               (rest[i, 1] - ref_lo) * sc + y_lo,
                               rest[i, 2] * sc)
    return fitted


def _run_ply_loop(
    renderer: GaussianSplatRenderer,
    camera: Camera,
    positions: np.ndarray,
    colors: np.ndarray,
    sizes: np.ndarray,
    opacities: np.ndarray,
    bone_positions: np.ndarray,
) -> None:
    """Render loop for a static 3DGS PLY scene with a walking bone overlay."""
    n = positions.shape[0]
    bone_indices = np.zeros((n, 4), dtype=np.float32)
    bone_weights = np.zeros((n, 4), dtype=np.float32)
    bone_weights[:, 0] = 1.0
    deform  = np.zeros((n, 3), dtype=np.float32)
    palette = np.eye(4, dtype=np.float32)[np.newaxis].repeat(renderer.max_bones, axis=0)

    # ── Locomotion constants ─────────────────────────────────────────────────
    WALK_SPEED      = 2.0    # world units / second
    TURN_SPEED      = 2.2    # radians  / second
    WALK_PHASE_RATE = 8.5    # radians  / second (gait cycle speed)

    # ── Skeleton root / local pose ───────────────────────────────────────────
    pelvis_idx  = JOINT_INDEX.get("pelvis", 0)
    skel_root   = bone_positions[pelvis_idx].astype(np.float32).copy()
    local_bones = (bone_positions - skel_root).astype(np.float32)
    skel_pos    = skel_root.copy()
    skel_yaw    = 0.0

    walk_phase        = 0.0   # 0 .. 2π  current phase in gait cycle
    walk_speed_factor = 0.0   # 0 = idle, 1 = full walk  (smoothly ramped)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _apply_transform(local: np.ndarray, pos: np.ndarray, yaw: float) -> np.ndarray:
        cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
        R = np.array([[cy,  0.0, sy],
                      [0.0, 1.0, 0.0],
                      [-sy, 0.0, cy]], dtype=np.float32)
        return (R @ local.T).T + pos

    def _animate_walk(bw: np.ndarray, yaw: float, phase: float, spd: float) -> np.ndarray:
        """
        Apply a procedural human walk cycle to world-space bone positions.

        bw    – (J,3) world-space rest positions at current skeleton pose
        yaw   – current facing angle (radians)
        phase – walk-cycle phase 0..2π
        spd   – blend factor 0=idle → 1=full walk
        """
        if spd < 0.001:
            return bw.copy()

        out = bw.copy()
        t   = float(phase)

        # Body-frame unit vectors
        fwd = np.array([np.sin(yaw), 0.0, np.cos(yaw)], np.float32)
        up  = np.array([0.0, 1.0, 0.0],                 np.float32)

        jx = JOINT_INDEX

        # Body height as reference scale
        head_y = float(bw[jx['head']][1])
        foot_y = float((bw[jx['foot_l']][1] + bw[jx['foot_r']][1]) * 0.5)
        H = max(head_y - foot_y, 0.5)

        # Animation amplitudes (all scaled by height × speed factor)
        STRIDE   = H * 0.28 * spd   # foot forward/back displacement
        LIFT     = H * 0.12 * spd   # foot lift height during swing
        BOB      = H * 0.020 * spd  # pelvis vertical bob per step
        ARM_SW   = H * 0.09 * spd   # hand forward/back swing
        KNEE_LED = H * 0.07 * spd   # knee forward lead when foot is swinging

        # Leg phases: right leg = t,  left leg = t + π  (alternating)
        rs = float(np.sin(t))            # +1 = right leg forward
        ls = float(np.sin(t + np.pi))   # +1 = left  leg forward

        # ── Pelvis + spine chain: vertical bob twice per stride ──────────────
        bob_v = up * (-abs(np.sin(t)) * BOB)
        for jname in ('pelvis', 'spine', 'chest', 'neck', 'head'):
            if jname in jx:
                out[jx[jname]] = bw[jx[jname]] + bob_v

        # ── Legs ─────────────────────────────────────────────────────────────
        for side, swing in (('l', ls), ('r', rs)):
            hi = jx[f'hip_{side}']
            ki = jx[f'knee_{side}']
            fi = jx[f'foot_{side}']

            # Hip: slight forward lean with the swinging leg
            out[hi] = bw[hi] + fwd * (swing * STRIDE * 0.20) + bob_v

            # Foot: swings forward/backward; lifts only on forward (swing > 0)
            foot_lift = max(0.0, swing) * LIFT
            out[fi] = bw[fi] + fwd * (swing * STRIDE) + up * foot_lift

            # Knee: halfway between hip and foot + forward lead proportional
            # to how far the foot is off the ground (makes the knee "bend")
            knee_lead = fwd * (abs(swing) * KNEE_LED)
            out[ki] = (out[hi] + out[fi]) * 0.5 + knee_lead

        # ── Arms (opposite phase to the same-side leg) ────────────────────────
        # left arm swings with right leg (rs), right arm swings with left leg (ls)
        for side, arm_swing in (('l', rs), ('r', ls)):
            shi = jx[f'shoulder_{side}']
            eli = jx[f'elbow_{side}']
            hdi = jx[f'hand_{side}']

            # Shoulder bobs with the torso
            out[shi] = bw[shi] + bob_v

            # Hand swings forward/back; arm bends slightly on forward swing
            out[hdi] = (bw[hdi]
                        + fwd * (arm_swing * ARM_SW)
                        + up  * (-abs(arm_swing) * ARM_SW * 0.25))

            # Elbow: midpoint shoulder ↔ hand (gives a natural bend)
            out[eli] = (out[shi] + out[hdi]) * 0.5

        return out

    # ── Initial render ───────────────────────────────────────────────────────
    show_bones    = True
    current_bones = _apply_transform(local_bones, skel_pos, skel_yaw)
    display_bones = current_bones.copy()

    renderer.update_skinning_uniforms(palette, backend="cpu")
    renderer.update_particles(
        bind_world_positions=positions,
        cpu_positions=positions,
        deformation_offsets=deform,
        colors=colors,
        sizes=sizes,
        brightness=opacities,
        bone_indices=bone_indices,
        bone_weights=bone_weights,
    )
    renderer.update_bone_lines(bone_positions=display_bones, parents=PARENTS, enabled=show_bones)

    clock   = pygame.time.Clock()
    running = True
    while running:
        dt = min(clock.tick(144) / 1000.0, 1.0 / 20.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_4:
                show_bones = not show_bones
            elif event.type == pygame.WINDOWRESIZED:
                renderer.resize(event.x, event.y)
            camera.handle_event(event)

        keys = pygame.key.get_pressed()
        camera.update(dt, keys)

        walking = bool(keys[pygame.K_w] or keys[pygame.K_s]
                       or keys[pygame.K_UP] or keys[pygame.K_DOWN])
        turning = bool(keys[pygame.K_a] or keys[pygame.K_d]
                       or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT])
        moving  = walking or turning

        # ── Smooth speed-factor ramp ─────────────────────────────────────────
        if moving:
            walk_speed_factor = min(1.0, walk_speed_factor + 5.0 * dt)
        else:
            walk_speed_factor = max(0.0, walk_speed_factor - 6.0 * dt)

        walk_phase = (walk_phase + WALK_PHASE_RATE * walk_speed_factor * dt) % (2 * np.pi)

        # ── Translate / rotate skeleton ──────────────────────────────────────
        pose_changed = False
        fwd_x = float(np.sin(skel_yaw))
        fwd_z = float(np.cos(skel_yaw))

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            skel_pos[0] += fwd_x * WALK_SPEED * dt
            skel_pos[2] += fwd_z * WALK_SPEED * dt
            pose_changed = True
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            skel_pos[0] -= fwd_x * WALK_SPEED * dt
            skel_pos[2] -= fwd_z * WALK_SPEED * dt
            pose_changed = True
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            skel_yaw -= TURN_SPEED * dt
            pose_changed = True
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            skel_yaw += TURN_SPEED * dt
            pose_changed = True

        # ── Recompute animated pose whenever skeleton moves OR animation runs ─
        if pose_changed or walk_speed_factor > 0.001:
            current_bones = _apply_transform(local_bones, skel_pos, skel_yaw)
            display_bones = _animate_walk(current_bones, skel_yaw,
                                          walk_phase, walk_speed_factor)
            renderer.update_bone_lines(bone_positions=display_bones,
                                       parents=PARENTS, enabled=show_bones)

        renderer.render(camera, show_bones=show_bones)
        pygame.display.set_caption(
            f"3DGS PLY Viewer | {n:,} Gaussians"
            f" | [4] Bones {'ON' if show_bones else 'OFF'}"
            " | W/S=walk  A/D=turn | Scroll=zoom | Drag=orbit | ESC=quit"
        )

    renderer.shutdown()


# ---------------------------------------------------------------------------

def _screen_to_world_on_plane(
    mouse_pos: tuple[int, int],
    camera: Camera,
    viewport_w: int,
    viewport_h: int,
    plane_y: float = 1.15,
) -> np.ndarray | None:
    if viewport_w <= 1 or viewport_h <= 1:
        return None

    mx, my = mouse_pos
    x = (2.0 * mx / viewport_w) - 1.0
    y = 1.0 - (2.0 * my / viewport_h)

    eye = camera.position
    view = look_at(eye, camera.target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    proj = perspective(np.deg2rad(58.0), viewport_w / viewport_h, 0.05, 50.0)

    inv = np.linalg.inv(proj @ view)
    p0 = inv @ np.array([x, y, -1.0, 1.0], dtype=np.float32)
    p1 = inv @ np.array([x, y, 1.0, 1.0], dtype=np.float32)
    p0 = p0[:3] / max(1e-8, p0[3])
    p1 = p1[:3] / max(1e-8, p1[3])

    ray_dir = p1 - p0
    if abs(ray_dir[1]) < 1e-6:
        return None
    t = (plane_y - p0[1]) / ray_dir[1]
    if t <= 0.0:
        return None
    return (p0 + t * ray_dir).astype(np.float32)


def _cross_marker(center: np.ndarray, size: float, color: tuple[float, float, float]) -> list[float]:
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r, g, b = color
    return [
        cx - size,
        cy,
        cz,
        r,
        g,
        b,
        cx + size,
        cy,
        cz,
        r,
        g,
        b,
        cx,
        cy - size,
        cz,
        r,
        g,
        b,
        cx,
        cy + size,
        cz,
        r,
        g,
        b,
        cx,
        cy,
        cz - size,
        r,
        g,
        b,
        cx,
        cy,
        cz + size,
        r,
        g,
        b,
    ]


def _resolve_render_colors(particle_data, skin_mode: str) -> np.ndarray:
    if skin_mode == "dominant":
        return dominant_bone_debug_colors(
            bone_indices=particle_data.bone_indices,
            bone_weights=particle_data.bone_weights,
            joint_count=len(JOINT_NAMES),
        )
    if skin_mode == "weights":
        return dominant_weight_debug_colors(particle_data.bone_weights)
    return particle_data.colors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime gaussian splat human prototype")
    parser.add_argument(
        "--motion-prompt",
        type=str,
        default="idle",
        help="Initial motion prompt: idle, walk, wave, dance, point (optionally with intensity, e.g. dance:1.2)",
    )
    parser.add_argument(
        "--splat-scale",
        type=float,
        default=6.0,
        metavar="FACTOR",
        help="Size multiplier for PLY Gaussian splats (default 6). Increase if splats are barely visible.",
    )
    parser.add_argument(
        "--ply",
        type=str,
        default=None,
        metavar="PATH",
        help="Load and display a 3DGS PLY file as a static scene (skips procedural human)",
    )
    parser.add_argument(
        "--face-image",
        type=str,
        default=None,
        help="Path to a face image to convert into splats and place on the dummy head",
    )
    parser.add_argument(
        "--face-particles",
        type=int,
        default=3200,
        help="Number of face splats to generate from the face image",
    )
    parser.add_argument(
        "--debug-skinning",
        choices=["off", "dominant", "weights"],
        default="off",
        help="Visualize skinning with dominant bone colors or dominant-weight heatmap",
    )
    parser.add_argument(
        "--skinning-backend",
        choices=["gpu", "cpu", "compare"],
        default="gpu",
        help="Skinning backend: gpu (default), cpu, or compare (GPU render + CPU/GPU diff metric)",
    )
    parser.add_argument(
        "--show-bones",
        action="store_true",
        help="Show animated skeleton lines for debug visualization",
    )
    parser.add_argument(
        "--show-ik-debug",
        action="store_true",
        help="Show IK chains, targets, and bend direction debug lines",
    )
    parser.add_argument(
        "--deform-strength",
        type=float,
        default=1.0,
        help="Neural additive deformation strength (0 disables)",
    )
    parser.add_argument(
        "--deform-rate",
        type=float,
        default=30.0,
        help="Neural deformation inference rate in Hz",
    )
    parser.add_argument(
        "--joint-damping",
        type=float,
        default=6.0,
        help="Joint angular damping for velocity interpolation",
    )
    parser.add_argument(
        "--joint-stiffness",
        type=float,
        default=17.5,
        help="Joint stiffness driving velocity toward target rotations",
    )
    parser.add_argument(
        "--joint-max-velocity",
        type=float,
        default=10.5,
        help="Maximum angular velocity per joint (rad/s)",
    )
    parser.add_argument(
        "--joint-inertia",
        type=float,
        default=1.0,
        help="Joint inertia for spring-damper dynamics",
    )
    parser.add_argument(
        "--joint-critical-damping",
        action="store_true",
        help="Enable critically damped spring behavior",
    )
    parser.add_argument(
        "--joint-damping-ratio",
        type=float,
        default=1.0,
        help="Damping ratio used when --joint-critical-damping is enabled",
    )
    parser.add_argument(
        "--root-inertia",
        type=float,
        default=1.45,
        help="Root movement inertia (higher feels heavier)",
    )
    parser.add_argument(
        "--root-friction",
        type=float,
        default=2.0,
        help="Root movement friction damping",
    )
    parser.add_argument(
        "--balance-strength",
        type=float,
        default=1.0,
        help="Balance compensation strength (0 disables)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.ply:
        print(f"Loading PLY: {args.ply}")
        positions, colors, sizes, opacities = load_ply_splats(args.ply)
        n = positions.shape[0]
        print(f"  {n:,} Gaussians loaded")
        sizes *= args.splat_scale          # world-space boost so point sprites are visible
        opacities = np.clip(opacities * 1.4, 0.0, 1.0)  # compensate for low-alpha captures
        renderer = GaussianSplatRenderer(width=1280, height=720, max_particles=n + 1, max_bones=128)
        centroid = positions.mean(axis=0)
        print(f"  Centroid: {centroid}")
        camera = Camera(target=centroid.astype(np.float32), distance=4.0)
        bone_positions = _fit_skeleton_to_ply(positions)
        _run_ply_loop(renderer, camera, positions, colors, sizes, opacities, bone_positions)
        return

    body_particle_count = 700_000
    renderer = GaussianSplatRenderer(width=1280, height=720, max_particles=body_particle_count + max(500, args.face_particles), max_bones=128)
    camera = Camera(target=np.array([0.0, 1.0, 0.0], dtype=np.float32))

    particle_data = generate_human_particles(total_particles=body_particle_count, seed=5)
    if args.face_image:
        particle_data = attach_face_splats(
            base=particle_data,
            image_path=args.face_image,
            face_particle_count=max(500, args.face_particles),
            seed=17,
        )

    animator = SkeletalAnimator(walk_frequency_hz=1.55)
    motion_ai = MotionAIController(fps=60.0, seed=19, sequence_len=84)
    motion_ai.enqueue_prompt(args.motion_prompt)
    joint_dynamics = JointDynamics(
        joint_count=len(JOINT_NAMES),
        damping=args.joint_damping,
        stiffness=args.joint_stiffness,
        inertia=args.joint_inertia,
        max_velocity=args.joint_max_velocity,
        max_acceleration=120.0,
        velocity_smoothing=0.35,
        critical_damping=args.joint_critical_damping,
        damping_ratio=args.joint_damping_ratio,
    )
    secondary_motion = SecondaryMotionController(response=7.2)
    root_motion = RootMotionController(
        walk_speed=1.15,
        run_speed=2.20,
        inertia=args.root_inertia,
        friction=args.root_friction,
        accel_gain=8.0,
        turn_inertia=2.2,
        turn_damping=2.9,
        dir_smoothing=8.8,
    )
    motion_physics = MotionPhysicsController(
        joint_dynamics=joint_dynamics,
        secondary_motion=secondary_motion,
        root_motion=root_motion,
        foot_planting=FootPlantingController(),
        balance_system=BalanceSystemController(strength=args.balance_strength),
    )
    deformer = NeuralDeformationController(
        local_positions=particle_data.local_positions,
        bind_world_positions=particle_data.bind_world_positions,
        bone_indices=particle_data.bone_indices,
        bone_weights=particle_data.bone_weights,
        rest_joint_positions=particle_data.rest_joint_positions,
        joint_count=len(JOINT_NAMES),
        pose_state_dim=len(PROMPT_TOKENS) + 2,
        strength=args.deform_strength,
        inference_hz=args.deform_rate,
        seed=41,
    )

    positions_world = np.empty_like(particle_data.local_positions)
    gpu_est_positions = np.empty_like(particle_data.local_positions)
    deformation_offsets = np.zeros_like(particle_data.local_positions)
    skin_mode = args.debug_skinning
    render_colors = _resolve_render_colors(particle_data, skin_mode)
    show_controls_ui = True
    show_bones = args.show_bones
    show_ik_debug = args.show_ik_debug
    show_motion_debug = False
    skin_backend = args.skinning_backend
    compare_rms = 0.0
    deform_rms = 0.0
    deform_max = 0.0
    root_speed = 0.0
    root_accel = 0.0
    move_state = "IDLE"
    balance_error = 0.0
    foot_state_l = "PLANTED"
    foot_state_r = "PLANTED"
    hand_targets: dict[str, np.ndarray | None] = {"l": None, "r": None}
    pending_hand_click: tuple[int, int] | None = None
    character_offset = np.zeros(3, dtype=np.float32)
    ground_y = 0.0
    command_mode = False
    command_buffer = ""

    clock = pygame.time.Clock()
    running = True
    paused = False

    fps_accum = 0.0
    fps_frames = 0
    last_title_update = time.perf_counter()

    while running:
        dt = min(clock.tick(144) / 1000.0, 1.0 / 20.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and command_mode:
                if event.key == pygame.K_RETURN:
                    motion_ai.enqueue_prompt(command_buffer)
                    command_mode = False
                    command_buffer = ""
                elif event.key == pygame.K_ESCAPE:
                    command_mode = False
                    command_buffer = ""
                elif event.key == pygame.K_BACKSPACE:
                    command_buffer = command_buffer[:-1]
                else:
                    if event.unicode and event.unicode.isprintable():
                        command_buffer += event.unicode
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SLASH:
                command_mode = True
                command_buffer = ""
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_9:
                motion_ai.enqueue_prompt("idle")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_0:
                motion_ai.enqueue_prompt("walk")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFTBRACKET:
                motion_ai.enqueue_prompt("wave")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHTBRACKET:
                motion_ai.enqueue_prompt("dance")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSLASH:
                motion_ai.enqueue_prompt("point")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = not paused
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_1:
                skin_mode = "off"
                render_colors = _resolve_render_colors(particle_data, skin_mode)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_2:
                skin_mode = "dominant"
                render_colors = _resolve_render_colors(particle_data, skin_mode)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_3:
                skin_mode = "weights"
                render_colors = _resolve_render_colors(particle_data, skin_mode)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                show_controls_ui = not show_controls_ui
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_4:
                show_bones = not show_bones
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_8:
                show_ik_debug = not show_ik_debug
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                show_motion_debug = not show_motion_debug
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                hand_targets["l"] = None
                hand_targets["r"] = None
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_5:
                skin_backend = "gpu"
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_6:
                skin_backend = "cpu"
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_7:
                skin_backend = "compare"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                pending_hand_click = event.pos
            elif event.type == pygame.WINDOWRESIZED:
                renderer.resize(event.x, event.y)

            camera.handle_event(event)

        keys = pygame.key.get_pressed()
        camera.update(dt, keys)

        ai_sample = motion_ai.update(dt=dt, paused=paused)
        desired_root_velocity = ai_sample.locomotion_delta / max(1e-4, dt)
        physics_out = motion_physics.update(
            animation_target_pose=ai_sample.pose,
            desired_root_velocity=desired_root_velocity,
            character_offset=character_offset,
            dt=dt,
            paused=paused,
            ground_y=ground_y,
        )
        character_offset += physics_out.root_delta

        status = motion_physics.debug_status()
        root_speed = float(status["root_speed"])
        root_accel = float(status["root_accel"])
        move_state = str(status["move_state"])
        foot_state_l = str(status["foot_l"])
        foot_state_r = str(status["foot_r"])
        balance_error = float(status["balance_error"])

        pose_state = np.zeros((len(PROMPT_TOKENS) + 2,), dtype=np.float32)
        prompt_idx = PROMPT_TOKENS.get(ai_sample.prompt, PROMPT_TOKENS["idle"])
        pose_state[prompt_idx] = 1.0
        pose_state[len(PROMPT_TOKENS)] = ai_sample.sequence_blend
        pose_state[len(PROMPT_TOKENS) + 1] = ai_sample.intensity

        world_rot, world_pos = animator.get_joint_world_transforms(
            dt=dt,
            paused=paused,
            target_pose=physics_out.pose,
            blend=1.0,
        )

        world_pos = (world_pos + character_offset[None, :]).astype(np.float32)

        deformation_offsets = deformer.predict_offsets(
            world_rot=world_rot,
            world_pos=world_pos,
            pose_state=pose_state,
            dt=dt,
            paused=paused,
        )
        deform_rms = deformer.stats.rms
        deform_max = deformer.stats.max_abs

        if pending_hand_click is not None:
            target = _screen_to_world_on_plane(
                mouse_pos=pending_hand_click,
                camera=camera,
                viewport_w=renderer.width,
                viewport_h=renderer.height,
                plane_y=1.15,
            )
            pending_hand_click = None
            if target is not None:
                hand_l = world_pos[JOINT_INDEX["hand_l"]]
                hand_r = world_pos[JOINT_INDEX["hand_r"]]
                if np.linalg.norm(target - hand_l) < np.linalg.norm(target - hand_r):
                    hand_targets["l"] = target
                else:
                    hand_targets["r"] = target

        ik_debug_vertices: list[float] = []

        # Arm IK: mouse-click targets for hand reaching.
        for side, root_name, mid_name, end_name, sign in (
            ("l", "shoulder_l", "elbow_l", "hand_l", -1.0),
            ("r", "shoulder_r", "elbow_r", "hand_r", 1.0),
        ):
            target = hand_targets[side]
            if target is None:
                continue

            root_idx = JOINT_INDEX[root_name]
            mid_idx = JOINT_INDEX[mid_name]
            end_idx = JOINT_INDEX[end_name]
            root = world_pos[root_idx]
            pole = root + np.array([0.35 * sign, 0.14, -0.22], dtype=np.float32)

            chain = solve_limb_ik(
                world_pos=world_pos,
                world_rot=world_rot,
                root_idx=root_idx,
                mid_idx=mid_idx,
                end_idx=end_idx,
                target=target,
                pole_vector=pole,
                blend=0.95,
                iterations=14,
                tolerance=1e-3,
            )

            if show_ik_debug:
                c = (0.95, 0.55, 0.15)
                ik_debug_vertices.extend([*chain[0], *c, *chain[1], *c, *chain[1], *c, *chain[2], *c])
                ik_debug_vertices.extend([*chain[0], 0.35, 0.95, 0.45, *pole, 0.35, 0.95, 0.45])
                ik_debug_vertices.extend(_cross_marker(target, 0.05, (0.98, 0.88, 0.22)))

        # Leg IK: automatic foot grounding on ground plane.
        for hip_name, knee_name, foot_name, sign in (
            ("hip_l", "knee_l", "foot_l", -1.0),
            ("hip_r", "knee_r", "foot_r", 1.0),
        ):
            root_idx = JOINT_INDEX[hip_name]
            mid_idx = JOINT_INDEX[knee_name]
            end_idx = JOINT_INDEX[foot_name]

            curr_foot = world_pos[end_idx].copy()
            side = "l" if foot_name.endswith("_l") else "r"
            target = motion_physics.foot_target(side=side, current_foot=curr_foot, ground_y=ground_y)
            pole = world_pos[root_idx] + np.array([0.20 * sign, 0.22, -0.25], dtype=np.float32)

            chain = solve_limb_ik(
                world_pos=world_pos,
                world_rot=world_rot,
                root_idx=root_idx,
                mid_idx=mid_idx,
                end_idx=end_idx,
                target=target,
                pole_vector=pole,
                blend=0.88,
                iterations=12,
                tolerance=1e-3,
            )

            if show_ik_debug:
                c = (0.22, 0.80, 0.98)
                ik_debug_vertices.extend([*chain[0], *c, *chain[1], *c, *chain[1], *c, *chain[2], *c])
                ik_debug_vertices.extend([*chain[0], 0.35, 0.95, 0.45, *pole, 0.35, 0.95, 0.45])
                target_color = (0.26, 0.95, 0.42) if (
                    (side == "l" and foot_state_l == "PLANTED")
                    or (side == "r" and foot_state_r == "PLANTED")
                ) else (0.30, 0.85, 0.95)
                ik_debug_vertices.extend(_cross_marker(target, 0.04, target_color))

        if show_motion_debug:
            ik_debug_vertices.extend(motion_physics.build_debug_overlay_vertices())
        palette = build_matrix_palette(
            world_rot=world_rot,
            world_pos=world_pos,
            rest_joint_positions=particle_data.rest_joint_positions,
            max_bones=renderer.max_bones,
        )

        backend_for_shader = "gpu"
        if skin_backend in ("cpu", "compare"):
            apply_skinning(
                bind_world_positions=particle_data.bind_world_positions,
                world_rot=world_rot,
                world_pos=world_pos,
                rest_joint_positions=particle_data.rest_joint_positions,
                bone_indices=particle_data.bone_indices,
                bone_weights=particle_data.bone_weights,
                out=positions_world,
            )
            positions_world += deformation_offsets
            backend_for_shader = "cpu" if skin_backend == "cpu" else "gpu"
        else:
            # GPU skinning path still receives additive neural deformation in shader.
            positions_world[:] = particle_data.bind_world_positions

        if skin_backend == "compare":
            # CPU reference vs matrix-palette estimate (GPU-equivalent math).
            apply_skinning_matrix_palette(
                bind_world_positions=particle_data.bind_world_positions,
                bone_indices=particle_data.bone_indices,
                bone_weights=particle_data.bone_weights,
                matrix_palette=palette,
                out=gpu_est_positions,
            )
            gpu_est_positions += deformation_offsets
            compare_rms = float(np.sqrt(np.mean((positions_world - gpu_est_positions) ** 2)))

        renderer.update_skinning_uniforms(palette, backend=backend_for_shader)
        renderer.update_bone_lines(bone_positions=world_pos, parents=PARENTS, enabled=show_bones)
        renderer.update_ik_debug_lines(
            vertices=np.asarray(ik_debug_vertices, dtype=np.float32) if ik_debug_vertices else None,
            enabled=(show_ik_debug or show_motion_debug),
        )

        renderer.update_particles(
            bind_world_positions=particle_data.bind_world_positions,
            cpu_positions=positions_world,
            deformation_offsets=deformation_offsets,
            colors=render_colors,
            sizes=particle_data.sizes,
            brightness=particle_data.brightness,
            bone_indices=particle_data.bone_indices,
            bone_weights=particle_data.bone_weights,
            face_mask=particle_data.face_mask,
        )
        renderer.render(camera, show_bones=show_bones, show_ik_debug=show_ik_debug)

        fps_accum += dt
        fps_frames += 1
        now = time.perf_counter()
        if now - last_title_update >= 0.25:
            fps = fps_frames / max(1e-6, fps_accum)
            state = "PAUSED" if paused else "RUNNING"
            skin_label = f"SKIN:{skin_mode.upper()}"
            backend_label = f"BACKEND:{skin_backend.upper()}"
            title = (
                f"Gaussian Splat Human Prototype | {state} | AI:{ai_sample.prompt.upper()} | {skin_label} | {backend_label} "
                f"| Blend:{ai_sample.sequence_blend:0.2f} | t:{motion_ai.command_age:0.2f} "
                f"| {fps:5.1f} FPS | Particles: {particle_data.local_positions.shape[0]}"
            )
            title += " | JOINT:SPRING-DAMPER"
            title += f" | ROOT:{move_state} v:{root_speed:0.2f} a:{root_accel:0.2f}"
            title += f" | BAL err:{balance_error:0.3f}"
            title += f" | FEET:L-{foot_state_l} R-{foot_state_r}"
            title += f" | DeformRMS:{deform_rms:0.5f} | DeformMax:{deform_max:0.5f}"
            if skin_backend == "compare":
                title += f" | CPUvsGPU-RMS:{compare_rms:0.6f}"
            if show_controls_ui:
                title += (
                    " | Buttons: [1 OFF] [2 BONE] [3 WEIGHT] [4 BONES] [5 GPU] [6 CPU] [7 CMP]"
                    " | [8 IKDBG] [T PHYDBG] [C CLR-HAND] | Prompt: / + Enter"
                    " | Hotkeys: [9 IDLE] [0 WALK] [LBRACKET WAVE] [RBRACKET DANCE] [BACKSLASH POINT]"
                    " | Prompt format: walk | dance:1.2"
                    " | Mouse RClick: Hand Target | Camera: Arrows"
                    " | P Pause | H Hide"
                )
            if command_mode:
                title += f" | CMD>{command_buffer}_"
            pygame.display.set_caption(title)
            last_title_update = now
            fps_accum = 0.0
            fps_frames = 0

    renderer.shutdown()


if __name__ == "__main__":
    main()
