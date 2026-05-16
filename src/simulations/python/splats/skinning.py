from __future__ import annotations

import numpy as np

from animation import JOINT_NAMES, PARENTS, REST_OFFSETS


def normalize_weights(weights: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize Nx4 skinning weights and recover invalid rows safely."""
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] != 4:
        raise ValueError("weights must have shape (N, 4)")

    sums = w.sum(axis=1, keepdims=True)
    valid = sums[:, 0] > eps
    out = np.zeros_like(w)
    out[valid] = w[valid] / sums[valid]

    # Fallback for degenerate rows.
    out[~valid, 0] = 1.0
    return out.astype(np.float32)


def get_rest_joint_world_positions(
    parents: np.ndarray | None = None,
    rest_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Compute rest-pose joint world positions from hierarchy offsets."""
    if parents is None:
        parents = PARENTS
    if rest_offsets is None:
        rest_offsets = REST_OFFSETS

    out = np.zeros_like(rest_offsets, dtype=np.float32)
    for j in range(len(parents)):
        p = parents[j]
        if p < 0:
            out[j] = rest_offsets[j]
        else:
            out[j] = out[p] + rest_offsets[j]
    return out


def migrate_single_joint_to_weights(
    joint_indices: np.ndarray,
    local_positions: np.ndarray,
    parents: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Migrate legacy rigid 1-joint particles into 4-bone skinning data.

    Heuristic blends parent/self/child using particle local Y to smooth elbow/knee/
    shoulder/hip bends without regenerating particles.
    """
    if parents is None:
        parents = PARENTS

    j_idx = np.asarray(joint_indices, dtype=np.int32)
    local = np.asarray(local_positions, dtype=np.float32)

    if j_idx.ndim != 1:
        raise ValueError("joint_indices must have shape (N,)")
    if local.ndim != 2 or local.shape[1] != 3 or local.shape[0] != j_idx.shape[0]:
        raise ValueError("local_positions must have shape (N, 3)")

    n = j_idx.shape[0]
    bone_indices = np.zeros((n, 4), dtype=np.int32)
    bone_weights = np.zeros((n, 4), dtype=np.float32)

    # Build first-child lookup for fast vectorized child assignment.
    child_first = np.full(len(parents), -1, dtype=np.int32)
    for child, parent in enumerate(parents):
        if parent >= 0 and child_first[parent] == -1:
            child_first[parent] = child

    parent_idx = parents[j_idx]
    child_idx = child_first[j_idx]

    # Local vertical direction estimate: above joint => parent influence, below => child.
    y = local[:, 1]
    up = np.clip(0.5 + 0.5 * np.tanh(4.5 * y), 0.0, 1.0).astype(np.float32)
    down = 1.0 - up

    # Stronger smoothing around primary bending joints.
    elbow_knee_shoulder_hip = {
        "shoulder_l",
        "shoulder_r",
        "elbow_l",
        "elbow_r",
        "hip_l",
        "hip_r",
        "knee_l",
        "knee_r",
    }
    smooth_strength = np.array(
        [0.34 if JOINT_NAMES[j] in elbow_knee_shoulder_hip else 0.20 for j in j_idx],
        dtype=np.float32,
    )

    w_parent = np.where(parent_idx >= 0, smooth_strength * up, 0.0).astype(np.float32)
    w_child = np.where(child_idx >= 0, smooth_strength * down, 0.0).astype(np.float32)
    w_self = np.clip(1.0 - w_parent - w_child, 0.0, 1.0).astype(np.float32)

    bone_indices[:, 0] = j_idx
    bone_indices[:, 1] = np.where(parent_idx >= 0, parent_idx, j_idx)
    bone_indices[:, 2] = np.where(child_idx >= 0, child_idx, j_idx)
    bone_indices[:, 3] = j_idx

    bone_weights[:, 0] = w_self
    bone_weights[:, 1] = w_parent
    bone_weights[:, 2] = w_child
    bone_weights[:, 3] = 0.0

    return bone_indices, normalize_weights(bone_weights)


def blend_joint_transforms(
    bind_world_positions: np.ndarray,
    world_rot: np.ndarray,
    world_pos: np.ndarray,
    rest_joint_positions: np.ndarray,
    bone_indices: np.ndarray,
    bone_weights: np.ndarray,
) -> np.ndarray:
    """Blend per-bone deformed positions for each particle using 4 influences."""
    idx = np.asarray(bone_indices, dtype=np.int32)
    w = normalize_weights(bone_weights)
    bind_world = np.asarray(bind_world_positions, dtype=np.float32)

    rot = world_rot[idx]  # (N, 4, 3, 3)
    pos = world_pos[idx]  # (N, 4, 3)
    rest = rest_joint_positions[idx]  # (N, 4, 3)

    # Offset from each influence joint in rest pose.
    rest_offset = bind_world[:, None, :] - rest
    influenced = np.einsum("nijk,nik->nij", rot, rest_offset, dtype=np.float32) + pos
    return np.sum(influenced * w[:, :, None], axis=1, dtype=np.float32).astype(np.float32)


def apply_skinning(
    bind_world_positions: np.ndarray,
    world_rot: np.ndarray,
    world_pos: np.ndarray,
    rest_joint_positions: np.ndarray,
    bone_indices: np.ndarray,
    bone_weights: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Apply blended linear blend skinning to persistent particles."""
    bind_world = np.asarray(bind_world_positions, dtype=np.float32)
    if out is None:
        out = np.empty_like(bind_world)

    out[:] = blend_joint_transforms(
        bind_world_positions=bind_world,
        world_rot=world_rot,
        world_pos=world_pos,
        rest_joint_positions=rest_joint_positions,
        bone_indices=bone_indices,
        bone_weights=bone_weights,
    )
    return out


def build_matrix_palette(
    world_rot: np.ndarray,
    world_pos: np.ndarray,
    rest_joint_positions: np.ndarray,
    max_bones: int = 128,
) -> np.ndarray:
    """Build Nx4x4 skinning matrix palette (current_world * inverse_bind)."""
    joint_count = world_rot.shape[0]
    if joint_count > max_bones:
        raise ValueError(f"joint_count={joint_count} exceeds max_bones={max_bones}")

    palette = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], max_bones, axis=0)

    rot = world_rot.astype(np.float32, copy=False)
    rest = rest_joint_positions.astype(np.float32, copy=False)
    pos = world_pos.astype(np.float32, copy=False)

    t = pos - np.einsum("nij,nj->ni", rot, rest, dtype=np.float32)

    palette[:joint_count, :3, :3] = rot
    palette[:joint_count, :3, 3] = t
    return palette


def apply_skinning_matrix_palette(
    bind_world_positions: np.ndarray,
    bone_indices: np.ndarray,
    bone_weights: np.ndarray,
    matrix_palette: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Apply matrix-palette skinning on CPU (matches GPU shader path)."""
    bind = np.asarray(bind_world_positions, dtype=np.float32)
    idx = np.asarray(bone_indices, dtype=np.int32)
    w = normalize_weights(bone_weights)

    if out is None:
        out = np.empty_like(bind)

    mats = matrix_palette[idx]  # (N, 4, 4, 4)
    vec4 = np.concatenate([bind, np.ones((bind.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = np.einsum("nijk,nk->nij", mats, vec4, dtype=np.float32)[..., :3]
    out[:] = np.sum(transformed * w[:, :, None], axis=1, dtype=np.float32)
    return out


def dominant_bone_debug_colors(
    bone_indices: np.ndarray,
    bone_weights: np.ndarray,
    joint_count: int,
) -> np.ndarray:
    """Color particles by dominant influence bone index."""
    idx = np.asarray(bone_indices, dtype=np.int32)
    w = normalize_weights(bone_weights)
    dominant_slot = np.argmax(w, axis=1)
    dominant_joint = idx[np.arange(idx.shape[0]), dominant_slot]

    hues = (dominant_joint.astype(np.float32) % joint_count) / max(1, joint_count)
    rgb = np.stack(
        [
            0.55 + 0.45 * np.sin(6.28318 * (hues + 0.00)),
            0.55 + 0.45 * np.sin(6.28318 * (hues + 0.33)),
            0.55 + 0.45 * np.sin(6.28318 * (hues + 0.66)),
        ],
        axis=1,
    )
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def dominant_weight_debug_colors(bone_weights: np.ndarray) -> np.ndarray:
    """Visualize confidence of dominant skinning weight (green=rigid, red=blended)."""
    w = normalize_weights(bone_weights)
    dom = np.max(w, axis=1)
    return np.stack([1.0 - dom, dom, 0.25 + 0.55 * dom], axis=1).astype(np.float32)
