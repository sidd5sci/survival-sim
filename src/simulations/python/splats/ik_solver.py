from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _normalize(a)
    b = _normalize(b)

    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.9999:
        return np.eye(3, dtype=np.float32)

    if dot < -0.9999:
        axis = _normalize(np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float32)))
        if np.linalg.norm(axis) < 1e-5:
            axis = _normalize(np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
        x, y, z = axis
        return np.array(
            [
                [-1.0 + 2.0 * x * x, 2.0 * x * y, 2.0 * x * z],
                [2.0 * x * y, -1.0 + 2.0 * y * y, 2.0 * y * z],
                [2.0 * x * z, 2.0 * y * z, -1.0 + 2.0 * z * z],
            ],
            dtype=np.float32,
        )

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float32)
    r = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - dot) / (s * s + 1e-8))
    return r.astype(np.float32)


def _apply_pole_vector(chain: np.ndarray, pole: np.ndarray) -> np.ndarray:
    root, mid, end = chain
    axis = _normalize(end - root)
    if np.linalg.norm(axis) < 1e-6:
        return chain

    mid_proj = root + axis * np.dot(mid - root, axis)
    pole_proj = root + axis * np.dot(pole - root, axis)

    mid_dir = _normalize(mid - mid_proj)
    pole_dir = _normalize(pole - pole_proj)
    if np.linalg.norm(mid_dir) < 1e-6 or np.linalg.norm(pole_dir) < 1e-6:
        return chain

    r = np.linalg.norm(mid - mid_proj)
    new_mid = mid_proj + pole_dir * r
    out = chain.copy()
    out[1] = new_mid
    return out


def fabrik_solve_chain(
    root: np.ndarray,
    mid: np.ndarray,
    end: np.ndarray,
    target: np.ndarray,
    iterations: int = 12,
    tolerance: float = 1e-3,
    pole_vector: np.ndarray | None = None,
) -> np.ndarray:
    """FABRIK solve for a 3-joint chain (root-mid-end)."""
    chain = np.stack([root, mid, end], axis=0).astype(np.float32)
    target = target.astype(np.float32)

    lengths = np.array(
        [
            np.linalg.norm(chain[1] - chain[0]),
            np.linalg.norm(chain[2] - chain[1]),
        ],
        dtype=np.float32,
    )
    total_len = float(lengths.sum())

    root_pos = chain[0].copy()
    dist_to_target = np.linalg.norm(target - root_pos)

    if dist_to_target >= total_len - 1e-6:
        direction = _normalize(target - root_pos)
        chain[1] = root_pos + direction * lengths[0]
        chain[2] = chain[1] + direction * lengths[1]
    else:
        for _ in range(iterations):
            chain[2] = target
            chain[1] = chain[2] + _normalize(chain[1] - chain[2]) * lengths[1]
            chain[0] = chain[1] + _normalize(chain[0] - chain[1]) * lengths[0]

            chain[0] = root_pos
            chain[1] = chain[0] + _normalize(chain[1] - chain[0]) * lengths[0]
            chain[2] = chain[1] + _normalize(chain[2] - chain[1]) * lengths[1]

            if np.linalg.norm(chain[2] - target) < tolerance:
                break

    if pole_vector is not None:
        chain = _apply_pole_vector(chain, pole_vector.astype(np.float32))

    return chain


def solve_limb_ik(
    world_pos: np.ndarray,
    world_rot: np.ndarray,
    root_idx: int,
    mid_idx: int,
    end_idx: int,
    target: np.ndarray,
    pole_vector: np.ndarray | None = None,
    blend: float = 1.0,
    iterations: int = 12,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """Solve limb IK in-place on world transforms and return solved chain positions."""
    old_chain = np.stack(
        [
            world_pos[root_idx],
            world_pos[mid_idx],
            world_pos[end_idx],
        ],
        axis=0,
    ).astype(np.float32)

    solved = fabrik_solve_chain(
        root=old_chain[0],
        mid=old_chain[1],
        end=old_chain[2],
        target=target,
        iterations=iterations,
        tolerance=tolerance,
        pole_vector=pole_vector,
    )

    b = float(np.clip(blend, 0.0, 1.0))
    chain = (1.0 - b) * old_chain + b * solved

    old_r = _normalize(old_chain[1] - old_chain[0])
    new_r = _normalize(chain[1] - chain[0])
    old_m = _normalize(old_chain[2] - old_chain[1])
    new_m = _normalize(chain[2] - chain[1])

    r_root = _rotation_between(old_r, new_r)
    r_mid = _rotation_between(old_m, new_m)

    world_rot[root_idx] = (r_root @ world_rot[root_idx]).astype(np.float32)
    world_rot[mid_idx] = (r_mid @ world_rot[mid_idx]).astype(np.float32)

    world_pos[root_idx] = chain[0]
    world_pos[mid_idx] = chain[1]
    world_pos[end_idx] = chain[2]
    return chain
