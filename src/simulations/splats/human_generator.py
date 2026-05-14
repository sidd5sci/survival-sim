from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from animation import JOINT_INDEX, JOINT_NAMES
from skinning import get_rest_joint_world_positions, migrate_single_joint_to_weights, normalize_weights


@dataclass
class HumanParticleData:
    """Container for procedural human splat particles in bind pose."""

    local_positions: np.ndarray
    colors: np.ndarray
    sizes: np.ndarray
    brightness: np.ndarray
    bone_indices: np.ndarray | None = None
    bone_weights: np.ndarray | None = None
    joint_indices: np.ndarray | None = None
    bind_world_positions: np.ndarray | None = None
    rest_joint_positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.local_positions = self.local_positions.astype(np.float32, copy=False)
        self.colors = self.colors.astype(np.float32, copy=False)
        self.sizes = self.sizes.astype(np.float32, copy=False)
        self.brightness = self.brightness.astype(np.float32, copy=False)

        if self.bone_indices is None or self.bone_weights is None:
            if self.joint_indices is None:
                raise ValueError("HumanParticleData requires either joint_indices or (bone_indices, bone_weights)")
            b_idx, b_w = migrate_single_joint_to_weights(
                joint_indices=self.joint_indices,
                local_positions=self.local_positions,
            )
            self.bone_indices = b_idx
            self.bone_weights = b_w
        else:
            self.bone_indices = self.bone_indices.astype(np.int32, copy=False)
            self.bone_weights = normalize_weights(self.bone_weights)

        if self.joint_indices is None:
            self.joint_indices = self.bone_indices[:, 0].astype(np.int32, copy=False)
        else:
            self.joint_indices = self.joint_indices.astype(np.int32, copy=False)

        if self.rest_joint_positions is None:
            self.rest_joint_positions = get_rest_joint_world_positions()
        else:
            self.rest_joint_positions = self.rest_joint_positions.astype(np.float32, copy=False)

        if self.bind_world_positions is None:
            self.bind_world_positions = (
                self.local_positions + self.rest_joint_positions[self.joint_indices]
            ).astype(np.float32, copy=False)
        else:
            self.bind_world_positions = self.bind_world_positions.astype(np.float32, copy=False)


def _pack_particle_data(
    positions: list[np.ndarray],
    colors: list[np.ndarray],
    sizes: list[np.ndarray],
    brightness: list[np.ndarray],
    joints: list[np.ndarray],
) -> HumanParticleData:
    local_positions = np.concatenate(positions, axis=0).astype(np.float32)
    color_arr = np.concatenate(colors, axis=0).astype(np.float32)
    size_arr = np.concatenate(sizes, axis=0).astype(np.float32)
    brightness_arr = np.concatenate(brightness, axis=0).astype(np.float32)
    joint_arr = np.concatenate(joints, axis=0).astype(np.int32)

    # Vertical brightness gradient adds a subtle fake lighting cue.
    y = local_positions[:, 1]
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-6)
    brightness_arr *= 0.82 + 0.30 * y_norm

    return HumanParticleData(
        local_positions=local_positions,
        colors=color_arr,
        sizes=size_arr,
        brightness=brightness_arr,
        joint_indices=joint_arr,
    )


def generate_face_splats_from_image(
    image_path: str | Path,
    face_particle_count: int = 3200,
    seed: int = 13,
) -> HumanParticleData:
    """
    Convert a face photo into gaussian-style splat particles for the head joint.

    This samples image pixels and maps them onto a curved face patch in the head's
    local space so the splats follow head animation.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Face image not found: {path}")

    rng = np.random.default_rng(seed)
    img = Image.open(path).convert("RGB")

    # Center crop to square to keep face framing stable.
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((256, 256), Image.Resampling.LANCZOS)

    rgb = np.asarray(img, dtype=np.float32) / 255.0

    # Slightly oversample and keep points inside an oval face mask.
    candidate = int(face_particle_count * 2.2)
    u = rng.random(candidate, dtype=np.float32)
    v = rng.random(candidate, dtype=np.float32)

    ex = (u - 0.5) / 0.46
    ey = (v - 0.50) / 0.56
    mask = (ex * ex + ey * ey) <= 1.0
    u = u[mask]
    v = v[mask]

    if u.size < face_particle_count:
        deficit = face_particle_count - u.size
        u = np.concatenate([u, rng.random(deficit, dtype=np.float32)])
        v = np.concatenate([v, rng.random(deficit, dtype=np.float32)])

    idx = rng.choice(u.size, size=face_particle_count, replace=False)
    u = u[idx]
    v = v[idx]

    px = np.clip((u * 255.0).astype(np.int32), 0, 255)
    py = np.clip((v * 255.0).astype(np.int32), 0, 255)
    colors = rgb[py, px].astype(np.float32)

    # Map image UVs to head-local curved patch: X lateral, Y vertical, Z forward.
    x = (u - 0.5) * 0.20
    y = (0.5 - v) * 0.24
    curvature = (x / 0.11) ** 2 + (y / 0.13) ** 2
    z = 0.09 - 0.018 * curvature

    positions = np.stack([x, y, z], axis=1).astype(np.float32)

    # Point size/brightness tuned for soft portrait-like face splats.
    sizes = rng.uniform(2.6, 5.2, size=face_particle_count).astype(np.float32)
    luma = colors @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    brightness = (0.68 + 0.70 * luma).astype(np.float32)

    joints = np.full(face_particle_count, JOINT_INDEX["head"], dtype=np.int32)

    return HumanParticleData(
        local_positions=positions,
        colors=colors,
        sizes=sizes,
        brightness=brightness,
        joint_indices=joints,
    )


def attach_face_splats(
    base: HumanParticleData,
    image_path: str | Path,
    face_particle_count: int = 3200,
    seed: int = 13,
) -> HumanParticleData:
    """Blend generated face-image splats on top of the existing human particles."""
    face = generate_face_splats_from_image(
        image_path=image_path,
        face_particle_count=face_particle_count,
        seed=seed,
    )
    return HumanParticleData(
        local_positions=np.concatenate([base.local_positions, face.local_positions], axis=0),
        colors=np.concatenate([base.colors, face.colors], axis=0),
        sizes=np.concatenate([base.sizes, face.sizes], axis=0),
        brightness=np.concatenate([base.brightness, face.brightness], axis=0),
        bone_indices=np.concatenate([base.bone_indices, face.bone_indices], axis=0),
        bone_weights=np.concatenate([base.bone_weights, face.bone_weights], axis=0),
        joint_indices=np.concatenate([base.joint_indices, face.joint_indices], axis=0),
    )


def _sample_ellipsoid(rng: np.random.Generator, count: int, radii: tuple[float, float, float]) -> np.ndarray:
    pts = rng.normal(size=(count, 3)).astype(np.float32)
    lengths = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
    dirs = pts / lengths
    # Cubic root keeps point density more uniform across volume.
    radius_scale = np.cbrt(rng.random(count, dtype=np.float32))[:, None]
    return dirs * radius_scale * np.asarray(radii, dtype=np.float32)


def _sample_limb(
    rng: np.random.Generator,
    count: int,
    length: float,
    radius: float,
    axis: int = 1,
    direction: float = -1.0,
) -> np.ndarray:
    # Volumetric cylinder-like distribution with slight taper for a low-poly silhouette.
    t = rng.random(count, dtype=np.float32)
    ang = rng.random(count, dtype=np.float32) * np.float32(2.0 * np.pi)
    taper = 1.0 - 0.25 * t
    r = np.sqrt(rng.random(count, dtype=np.float32)) * radius * taper

    coords = np.zeros((count, 3), dtype=np.float32)
    coords[:, axis] = direction * t * length

    radial_a = (axis + 1) % 3
    radial_b = (axis + 2) % 3
    coords[:, radial_a] = np.cos(ang) * r
    coords[:, radial_b] = np.sin(ang) * r
    return coords


def _offset_local(local_pts: np.ndarray, offset_xyz: tuple[float, float, float]) -> np.ndarray:
    return (local_pts + np.asarray(offset_xyz, dtype=np.float32)[None, :]).astype(np.float32)


def _append_part(
    positions: list[np.ndarray],
    colors: list[np.ndarray],
    sizes: list[np.ndarray],
    brightness: list[np.ndarray],
    joints: list[np.ndarray],
    local_pts: np.ndarray,
    color_rgb: tuple[float, float, float],
    size_range: tuple[float, float],
    brightness_range: tuple[float, float],
    joint_name: str,
    rng: np.random.Generator,
) -> None:
    count = local_pts.shape[0]
    positions.append(local_pts)

    color_noise = (rng.random((count, 3), dtype=np.float32) - 0.5) * 0.08
    part_colors = np.clip(np.asarray(color_rgb, dtype=np.float32) + color_noise, 0.0, 1.0)
    colors.append(part_colors)

    s_min, s_max = size_range
    sizes.append(rng.uniform(s_min, s_max, size=count).astype(np.float32))

    b_min, b_max = brightness_range
    brightness.append(rng.uniform(b_min, b_max, size=count).astype(np.float32))

    joint_id = JOINT_INDEX[joint_name]
    joints.append(np.full(count, joint_id, dtype=np.int32))


def generate_human_particles(total_particles: int = 14000, seed: int = 7) -> HumanParticleData:
    """
    Procedurally generate a stylized low-poly human as gaussian splat particles.

    Particles are authored in each joint's local space (bind pose). The animation
    system applies per-joint transforms every frame to produce motion.
    """
    if total_particles < len(JOINT_NAMES) * 10:
        raise ValueError("total_particles too low for articulated body generation")

    rng = np.random.default_rng(seed)

    positions: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    brightness: list[np.ndarray] = []
    joints: list[np.ndarray] = []

    # Soft allocation ratios by body part.
    n_head = int(total_particles * 0.09)
    n_torso = int(total_particles * 0.31)
    n_arms = int(total_particles * 0.25)
    n_legs = int(total_particles * 0.29)
    n_extras = total_particles - (n_head + n_torso + n_arms + n_legs)

    # Head
    _append_part(
        positions,
        colors,
        sizes,
        brightness,
        joints,
        _sample_ellipsoid(rng, n_head, (0.11, 0.14, 0.11)),
        color_rgb=(0.86, 0.72, 0.62),
        size_range=(6.5, 9.5),
        brightness_range=(0.86, 1.06),
        joint_name="head",
        rng=rng,
    )

    # Torso chunks distributed across pelvis/spine/chest/neck for better silhouette.
    pelvis_n = int(n_torso * 0.31)
    spine_n = int(n_torso * 0.24)
    chest_n = int(n_torso * 0.35)
    neck_n = n_torso - pelvis_n - spine_n - chest_n

    _append_part(
        positions,
        colors,
        sizes,
        brightness,
        joints,
        _sample_ellipsoid(rng, pelvis_n, (0.21, 0.12, 0.13)),
        color_rgb=(0.26, 0.58, 0.84),
        size_range=(5.0, 8.2),
        brightness_range=(0.76, 1.00),
        joint_name="pelvis",
        rng=rng,
    )

    _append_part(
        positions,
        colors,
        sizes,
        brightness,
        joints,
        _sample_ellipsoid(rng, spine_n, (0.12, 0.17, 0.08)),
        color_rgb=(0.22, 0.54, 0.80),
        size_range=(4.8, 8.0),
        brightness_range=(0.74, 0.98),
        joint_name="spine",
        rng=rng,
    )

    _append_part(
        positions,
        colors,
        sizes,
        brightness,
        joints,
        _sample_ellipsoid(rng, chest_n, (0.24, 0.18, 0.13)),
        color_rgb=(0.20, 0.50, 0.75),
        size_range=(5.0, 8.8),
        brightness_range=(0.78, 1.03),
        joint_name="chest",
        rng=rng,
    )

    _append_part(
        positions,
        colors,
        sizes,
        brightness,
        joints,
        _sample_ellipsoid(rng, neck_n, (0.07, 0.10, 0.07)),
        color_rgb=(0.78, 0.66, 0.58),
        size_range=(3.8, 6.0),
        brightness_range=(0.82, 1.06),
        joint_name="neck",
        rng=rng,
    )

    # Arms
    arm_each = n_arms // 4
    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        _append_part(
            positions,
            colors,
            sizes,
            brightness,
            joints,
            _offset_local(_sample_limb(rng, arm_each, length=0.34, radius=0.055), (0.02 * side_sign, -0.01, 0.01)),
            color_rgb=(0.80, 0.66, 0.58),
            size_range=(3.6, 6.5),
            brightness_range=(0.72, 0.98),
            joint_name=f"shoulder_{side}",
            rng=rng,
        )
        _append_part(
            positions,
            colors,
            sizes,
            brightness,
            joints,
            _offset_local(_sample_limb(rng, arm_each, length=0.31, radius=0.046), (0.02 * side_sign, 0.0, 0.01)),
            color_rgb=(0.84, 0.69, 0.60),
            size_range=(3.2, 6.0),
            brightness_range=(0.72, 1.02),
            joint_name=f"elbow_{side}",
            rng=rng,
        )

    # Legs
    leg_each = n_legs // 4
    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        _append_part(
            positions,
            colors,
            sizes,
            brightness,
            joints,
            _offset_local(_sample_limb(rng, leg_each, length=0.45, radius=0.072), (0.018 * side_sign, 0.0, 0.0)),
            color_rgb=(0.30, 0.32, 0.36),
            size_range=(4.0, 7.0),
            brightness_range=(0.65, 0.90),
            joint_name=f"hip_{side}",
            rng=rng,
        )
        _append_part(
            positions,
            colors,
            sizes,
            brightness,
            joints,
            _offset_local(_sample_limb(rng, leg_each, length=0.44, radius=0.064), (0.016 * side_sign, 0.0, 0.0)),
            color_rgb=(0.26, 0.28, 0.32),
            size_range=(3.8, 6.8),
            brightness_range=(0.62, 0.88),
            joint_name=f"knee_{side}",
            rng=rng,
        )

        foot_n = max(100, n_extras // 4)
        foot_pts = _sample_ellipsoid(rng, foot_n, (0.082, 0.042, 0.21))
        foot_pts[:, 2] += 0.10
        foot_pts[:, 0] += 0.014 * side_sign
        _append_part(
            positions,
            colors,
            sizes,
            brightness,
            joints,
            foot_pts,
            color_rgb=(0.24, 0.25, 0.30),
            size_range=(3.0, 5.5),
            brightness_range=(0.60, 0.84),
            joint_name=f"foot_{side}",
            rng=rng,
        )

    return _pack_particle_data(positions, colors, sizes, brightness, joints)
