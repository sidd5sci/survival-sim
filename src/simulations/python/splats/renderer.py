from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np
import pygame

# Optional gsplat integration — falls back to point-sprite renderer if not installed
# or if the CUDA backend is unavailable (e.g. CPU-only machines / macOS without CUDA).
try:
    import torch as _torch
    from gsplat import rasterization as _gsplat_rasterize
    # gsplat sets its internal _C extension to None when CUDA is absent and prints a
    # warning. Verify the backend is actually functional before enabling the path.
    from gsplat.cuda import _wrapper as _gsplat_wrapper
    _GSPLAT_AVAILABLE = getattr(_gsplat_wrapper, "_C", None) is not None
except (ImportError, AttributeError):
    _GSPLAT_AVAILABLE = False

# World-space scale factor: maps human_generator size units (~3–10) to meters (~0.007–0.022 m).
_GSPLAT_SCALE_FACTOR: float = 0.0022


@dataclass
class Camera:
    target: np.ndarray
    distance: float = 3.6
    yaw: float = 0.0
    pitch: float = 0.18
    auto_orbit_speed: float = 0.0
    move_speed: float = 1.7

    def __post_init__(self) -> None:
        self.auto_yaw = 0.0
        self.dragging = False
        self.last_mouse = (0, 0)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.dragging = True
            self.last_mouse = event.pos
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            dx = event.pos[0] - self.last_mouse[0]
            dy = event.pos[1] - self.last_mouse[1]
            self.last_mouse = event.pos
            self.yaw += dx * 0.0055
            self.pitch = np.clip(self.pitch + dy * 0.0035, -1.2, 1.2)
        elif event.type == pygame.MOUSEWHEEL:
            self.distance = np.clip(self.distance - event.y * 0.18, 1.8, 9.0)

    def update(self, dt: float, keys: pygame.key.ScancodeWrapper) -> None:
        self.auto_yaw += self.auto_orbit_speed * dt
        yaw = self.yaw + self.auto_yaw

        forward = np.array([np.sin(yaw), 0.0, np.cos(yaw)], dtype=np.float32)
        right = np.array([forward[2], 0.0, -forward[0]], dtype=np.float32)

        move = np.zeros(3, dtype=np.float32)
        if keys[pygame.K_UP]:
            move += forward
        if keys[pygame.K_DOWN]:
            move -= forward
        if keys[pygame.K_RIGHT]:
            move += right
        if keys[pygame.K_LEFT]:
            move -= right

        mag = np.linalg.norm(move)
        if mag > 1e-6:
            move = (move / mag) * self.move_speed * dt
            self.target += move

    @property
    def position(self) -> np.ndarray:
        yaw = self.yaw + self.auto_yaw
        x = self.distance * np.cos(self.pitch) * np.sin(yaw)
        y = self.distance * np.sin(self.pitch)
        z = self.distance * np.cos(self.pitch) * np.cos(yaw)
        return self.target + np.array([x, y, z], dtype=np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(fov_y / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    fwd = _normalize(target - eye)
    side = _normalize(np.cross(fwd, up))
    up2 = np.cross(side, fwd)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = side
    m[1, :3] = up2
    m[2, :3] = -fwd
    m[0, 3] = -np.dot(side, eye)
    m[1, 3] = -np.dot(up2, eye)
    m[2, 3] = np.dot(fwd, eye)
    return m


class GaussianSplatRenderer:
    """ModernGL point-splat renderer with gaussian alpha falloff in fragment shader."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        max_particles: int = 100000,
        max_bones: int = 128,
    ):
        pygame.init()
        # Explicitly request a core GL context so ModernGL can attach reliably.
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

        pygame.display.set_caption("Gaussian Splat Human Prototype")
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        pygame.display.set_mode((width, height), flags)

        self.width = width
        self.height = height
        self.max_particles = max_particles
        self.max_bones = max_bones

        self.ctx = moderngl.create_context(require=330)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.fov_y = np.deg2rad(58.0)

        shader_dir = Path(__file__).resolve().parent / "shaders"
        vertex_src = (shader_dir / "vertex.glsl").read_text(encoding="utf-8")
        fragment_src = (shader_dir / "fragment.glsl").read_text(encoding="utf-8")
        face_vertex_src = (shader_dir / "face_vertex.glsl").read_text(encoding="utf-8")
        face_fragment_src = (shader_dir / "face_fragment.glsl").read_text(encoding="utf-8")

        self.program = self.ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)
        self.face_program = self.ctx.program(vertex_shader=face_vertex_src, fragment_shader=face_fragment_src)

        # Lightweight line program for orientation helpers (wire mesh + axes).
        self.line_program = self.ctx.program(
            vertex_shader="""
                #version 330 core
                in vec3 in_pos;
                in vec3 in_color;

                uniform mat4 u_mvp;
                out vec3 v_color;

                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330 core
                in vec3 v_color;
                out vec4 fragColor;

                void main() {
                    fragColor = vec4(v_color, 0.95);
                }
            """,
        )

        particle_dtype = np.dtype(
            [
                ("in_bind_pos", "f4", 3),
                ("in_cpu_pos", "f4", 3),
                ("in_deform", "f4", 3),
                ("in_color", "f4", 3),
                ("in_size", "f4"),
                ("in_brightness", "f4"),
                ("in_bone_idx", "f4", 4),
                ("in_bone_w", "f4", 4),
            ]
        )
        self._particle_dtype = particle_dtype

        self.vbo = self.ctx.buffer(reserve=max_particles * particle_dtype.itemsize, dynamic=True)
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (
                    self.vbo,
                    "3f 3f 3f 3f 1f 1f 4f 4f",
                    "in_bind_pos",
                    "in_cpu_pos",
                    "in_deform",
                    "in_color",
                    "in_size",
                    "in_brightness",
                    "in_bone_idx",
                    "in_bone_w",
                )
            ],
        )
        self.face_capacity = max(16384, max_particles // 4)
        self.face_vbo = self.ctx.buffer(reserve=self.face_capacity * particle_dtype.itemsize, dynamic=True)
        self.face_vao = self.ctx.vertex_array(
            self.face_program,
            [
                (
                    self.face_vbo,
                    "3f 3f 3f 3f 1f 1f 4f 4f",
                    "in_bind_pos",
                    "in_cpu_pos",
                    "in_deform",
                    "in_color",
                    "in_size",
                    "in_brightness",
                    "in_bone_idx",
                    "in_bone_w",
                )
            ],
        )

        self.grid_vbo, self.grid_vao, self.grid_vertex_count = self._build_wire_mesh_plane(size=4.0, step=0.25)
        self.bone_vbo = self.ctx.buffer(reserve=4096, dynamic=True)
        self.bone_vao = self.ctx.vertex_array(
            self.line_program,
            [
                (
                    self.bone_vbo,
                    "3f 3f",
                    "in_pos",
                    "in_color",
                )
            ],
        )
        self.bone_vertex_count = 0
        self.ik_vbo = self.ctx.buffer(reserve=8192, dynamic=True)
        self.ik_vao = self.ctx.vertex_array(
            self.line_program,
            [
                (
                    self.ik_vbo,
                    "3f 3f",
                    "in_pos",
                    "in_color",
                )
            ],
        )
        self.ik_vertex_count = 0

        self.particle_count = 0
        self.face_particle_count = 0
        self.program["u_skinning_backend"].value = 1
        self.face_program["u_skinning_backend"].value = 1

        # gsplat rendering path (preferred when library is installed)
        self._use_gsplat = _GSPLAT_AVAILABLE
        if self._use_gsplat:
            self._init_gsplat()

    # ------------------------------------------------------------------
    # gsplat setup
    # ------------------------------------------------------------------

    def _init_gsplat(self) -> None:
        """Set up gsplat fullscreen-blit resources and torch tensor storage."""
        self._gsplat_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

        # Particle tensors — populated by update_particles / update_skinning_uniforms
        self._gsplat_bind_pos: _torch.Tensor | None = None
        self._gsplat_quats: _torch.Tensor | None = None
        self._gsplat_scales: _torch.Tensor | None = None
        self._gsplat_opacities: _torch.Tensor | None = None
        self._gsplat_colors: _torch.Tensor | None = None
        self._gsplat_bone_idx: _torch.Tensor | None = None
        self._gsplat_bone_w: _torch.Tensor | None = None
        self._gsplat_palette: _torch.Tensor | None = None
        self._gsplat_n: int = 0

        # Full-screen quad that blits the gsplat RGBA texture
        blit_vert = """
#version 330 core
in vec2 in_pos;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    // Flip V: GL textures start at bottom-left; image tensors start at top-left.
    v_uv = vec2(in_pos.x * 0.5 + 0.5, 1.0 - (in_pos.y * 0.5 + 0.5));
}
"""
        blit_frag = """
#version 330 core
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(u_tex, v_uv);
}
"""
        self._blit_prog = self.ctx.program(vertex_shader=blit_vert, fragment_shader=blit_frag)
        # Two triangles (6 vertices) that cover NDC [-1, 1]
        quad = np.array(
            [-1.0, -1.0,  1.0, -1.0, -1.0,  1.0,
              1.0, -1.0,  1.0,  1.0, -1.0,  1.0],
            dtype=np.float32,
        )
        self._blit_vbo = self.ctx.buffer(quad.tobytes())
        self._blit_vao = self.ctx.vertex_array(
            self._blit_prog, [(self._blit_vbo, "2f", "in_pos")]
        )
        self._blit_prog["u_tex"] = 0
        # RGBA texture; recreated lazily on resize
        self._gsplat_tex = self.ctx.texture((self.width, self.height), 4)
        self._gsplat_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # ------------------------------------------------------------------
    # gsplat skinning (LBS in PyTorch on device)
    # ------------------------------------------------------------------

    def _apply_gsplat_skinning(self) -> _torch.Tensor:
        """Linear blend skinning of bind positions using the current matrix palette."""
        bp = self._gsplat_bind_pos   # (N, 3)
        bi = self._gsplat_bone_idx   # (N, 4) int64
        bw = self._gsplat_bone_w     # (N, 4) float32
        pal = self._gsplat_palette   # (B, 4, 4)

        N = bp.shape[0]
        B = pal.shape[0]
        ones = _torch.ones(N, 1, device=bp.device, dtype=_torch.float32)
        p_h = _torch.cat([bp, ones], dim=1)  # (N, 4)

        out = _torch.zeros(N, 3, device=bp.device, dtype=_torch.float32)
        for k in range(4):
            idx = bi[:, k].long().clamp(0, B - 1)  # (N,)
            w = bw[:, k]                             # (N,)
            mats = pal[idx]                          # (N, 4, 4)
            transformed = _torch.bmm(mats, p_h.unsqueeze(-1)).squeeze(-1)  # (N, 4)
            out += w.unsqueeze(1) * transformed[:, :3]
        return out  # (N, 3)

    # ------------------------------------------------------------------
    # gsplat rasterization pass
    # ------------------------------------------------------------------

    def _render_gsplat(self, camera: Camera) -> None:
        """Run gsplat rasterization and blit result as a fullscreen quad."""
        if self._gsplat_n == 0 or self._gsplat_bind_pos is None or self._gsplat_palette is None:
            return

        dev = self._gsplat_device
        W, H = self.width, self.height

        with _torch.no_grad():
            means = self._apply_gsplat_skinning()  # (N, 3) skinned world positions

            # Convert OpenGL view matrix to OpenCV convention (flip Y and Z rows)
            eye = camera.position
            view_gl = look_at(eye, camera.target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
            view_cv = view_gl.copy()
            view_cv[1, :] = -view_gl[1, :]
            view_cv[2, :] = -view_gl[2, :]
            viewmats = _torch.from_numpy(view_cv[None]).to(dev)  # (1, 4, 4)

            # Pinhole intrinsics matching our symmetric FOV
            fy = H / (2.0 * np.tan(self.fov_y * 0.5))
            K = np.array([[fy, 0.0, W / 2.0],
                           [0.0, fy, H / 2.0],
                           [0.0, 0.0, 1.0]], dtype=np.float32)
            Ks = _torch.from_numpy(K[None]).to(dev)  # (1, 3, 3)

            # Background colour matching the GL clear colour
            bg = _torch.tensor([[0.025, 0.030, 0.045]], device=dev, dtype=_torch.float32)

            renders, alphas, _info = _gsplat_rasterize(
                means=means,
                quats=self._gsplat_quats,
                scales=self._gsplat_scales,
                opacities=self._gsplat_opacities,
                colors=self._gsplat_colors,
                viewmats=viewmats,
                Ks=Ks,
                width=W,
                height=H,
                near_plane=0.05,
                far_plane=50.0,
                render_mode="RGB",
                backgrounds=bg,
            )
            # renders: (1, H, W, 3); alphas: (1, H, W, 1)
            rgb = renders[0]
            alpha = alphas[0]
            rgba = _torch.cat([rgb, alpha], dim=-1).clamp(0.0, 1.0)
            rgba_np = (rgba * 255).to(_torch.uint8).cpu().numpy()  # (H, W, 4) uint8

        # Recreate texture on viewport resize
        if self._gsplat_tex.width != W or self._gsplat_tex.height != H:
            self._gsplat_tex.release()
            self._gsplat_tex = self.ctx.texture((W, H), 4)
            self._gsplat_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self._gsplat_tex.write(rgba_np.tobytes())
        self._gsplat_tex.use(0)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._blit_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _build_wire_mesh_plane(self, size: float, step: float) -> tuple[moderngl.Buffer, moderngl.VertexArray, int]:
        verts: list[float] = []

        # Ground wire mesh on XZ plane to show where "down" is.
        v = -size
        while v <= size + 1e-6:
            major = abs(v) < 1e-6
            line_color = (0.30, 0.32, 0.38) if not major else (0.50, 0.52, 0.58)

            # Parallel to X axis.
            verts.extend([-size, 0.0, v, *line_color])
            verts.extend([size, 0.0, v, *line_color])

            # Parallel to Z axis.
            verts.extend([v, 0.0, -size, *line_color])
            verts.extend([v, 0.0, size, *line_color])
            v += step

        # Axis guides for orientation: X=red, Y(up)=green, Z=blue.
        axis = 1.3
        verts.extend([0.0, 0.0, 0.0, 0.95, 0.30, 0.30])
        verts.extend([axis, 0.0, 0.0, 0.95, 0.30, 0.30])

        verts.extend([0.0, 0.0, 0.0, 0.35, 0.95, 0.35])
        verts.extend([0.0, axis, 0.0, 0.35, 0.95, 0.35])

        verts.extend([0.0, 0.0, 0.0, 0.35, 0.55, 0.95])
        verts.extend([0.0, 0.0, axis, 0.35, 0.55, 0.95])

        data = np.asarray(verts, dtype=np.float32)
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(
            self.line_program,
            [
                (
                    vbo,
                    "3f 3f",
                    "in_pos",
                    "in_color",
                )
            ],
        )
        return vbo, vao, data.size // 6

    def resize(self, width: int, height: int) -> None:
        self.width = max(1, width)
        self.height = max(1, height)
        self.ctx.viewport = (0, 0, self.width, self.height)

    def update_particles(
        self,
        bind_world_positions: np.ndarray,
        cpu_positions: np.ndarray,
        deformation_offsets: np.ndarray,
        colors: np.ndarray,
        sizes: np.ndarray,
        brightness: np.ndarray,
        bone_indices: np.ndarray,
        bone_weights: np.ndarray,
        face_mask: np.ndarray | None = None,
    ) -> None:
        count = bind_world_positions.shape[0]
        if count > self.max_particles:
            raise ValueError(f"Particle count {count} exceeds max_particles={self.max_particles}")

        if face_mask is None:
            face_mask_arr = np.zeros((count,), dtype=bool)
        else:
            face_mask_arr = np.asarray(face_mask, dtype=bool)
            if face_mask_arr.shape[0] != count:
                raise ValueError("face_mask must have shape (N,) matching particle count")

        body_mask = ~face_mask_arr
        body_count = int(np.sum(body_mask))
        face_count = int(np.sum(face_mask_arr))

        if face_count > self.face_capacity:
            raise ValueError(f"Face particle count {face_count} exceeds face_capacity={self.face_capacity}")

        if body_count > 0:
            packed = np.empty(body_count, dtype=self._particle_dtype)
            packed["in_bind_pos"] = bind_world_positions[body_mask].astype(np.float32, copy=False)
            packed["in_cpu_pos"] = cpu_positions[body_mask].astype(np.float32, copy=False)
            packed["in_deform"] = deformation_offsets[body_mask].astype(np.float32, copy=False)
            packed["in_color"] = colors[body_mask].astype(np.float32, copy=False)
            packed["in_size"] = sizes[body_mask].astype(np.float32, copy=False)
            packed["in_brightness"] = brightness[body_mask].astype(np.float32, copy=False)
            packed["in_bone_idx"] = bone_indices[body_mask].astype(np.float32, copy=False)
            packed["in_bone_w"] = bone_weights[body_mask].astype(np.float32, copy=False)
            self.vbo.write(packed.tobytes(), offset=0)

        if face_count > 0:
            packed_face = np.empty(face_count, dtype=self._particle_dtype)
            packed_face["in_bind_pos"] = bind_world_positions[face_mask_arr].astype(np.float32, copy=False)
            packed_face["in_cpu_pos"] = cpu_positions[face_mask_arr].astype(np.float32, copy=False)
            packed_face["in_deform"] = deformation_offsets[face_mask_arr].astype(np.float32, copy=False)
            packed_face["in_color"] = colors[face_mask_arr].astype(np.float32, copy=False)
            packed_face["in_size"] = sizes[face_mask_arr].astype(np.float32, copy=False)
            packed_face["in_brightness"] = brightness[face_mask_arr].astype(np.float32, copy=False)
            packed_face["in_bone_idx"] = bone_indices[face_mask_arr].astype(np.float32, copy=False)
            packed_face["in_bone_w"] = bone_weights[face_mask_arr].astype(np.float32, copy=False)
            self.face_vbo.write(packed_face.tobytes(), offset=0)

        self.particle_count = body_count
        self.face_particle_count = face_count

        # Store all-particle tensors for gsplat (body + face combined, before split)
        if self._use_gsplat:
            dev = self._gsplat_device
            N = count
            all_bind = bind_world_positions.astype(np.float32, copy=False)
            all_colors = colors.astype(np.float32, copy=False)
            all_sizes = sizes.astype(np.float32, copy=False)
            all_brightness = brightness.astype(np.float32, copy=False)
            all_bi = bone_indices.reshape(N, -1)[:, :4].astype(np.int32, copy=False)
            all_bw = bone_weights.reshape(N, -1)[:, :4].astype(np.float32, copy=False)

            self._gsplat_bind_pos = _torch.from_numpy(all_bind).to(dev)
            self._gsplat_colors = _torch.from_numpy(all_colors).to(dev)
            # Identity quaternions [w, x, y, z] = [1, 0, 0, 0] — isotropic Gaussians
            q = _torch.zeros(N, 4, device=dev, dtype=_torch.float32)
            q[:, 0] = 1.0
            self._gsplat_quats = q
            # Isotropic scale derived from per-particle size; contiguous for gsplat
            s = _torch.from_numpy(all_sizes).to(dev)
            self._gsplat_scales = (s.unsqueeze(1).expand(-1, 3) * _GSPLAT_SCALE_FACTOR).contiguous()
            # Opacity from brightness clamped to [0, 1]
            self._gsplat_opacities = _torch.from_numpy(all_brightness).to(dev).clamp(0.05, 1.0)
            self._gsplat_bone_idx = _torch.from_numpy(all_bi).to(dev)
            self._gsplat_bone_w = _torch.from_numpy(all_bw).to(dev)
            self._gsplat_n = N

    def update_skinning_uniforms(self, matrix_palette: np.ndarray, backend: str = "gpu") -> None:
        palette = matrix_palette.astype(np.float32, copy=False)
        if palette.shape != (self.max_bones, 4, 4):
            raise ValueError(f"matrix_palette must have shape ({self.max_bones}, 4, 4)")
        self.program["u_bone_mats"].write(palette.transpose(0, 2, 1).tobytes())
        self.program["u_skinning_backend"].value = 1 if backend == "gpu" else 0
        self.face_program["u_bone_mats"].write(palette.transpose(0, 2, 1).tobytes())
        self.face_program["u_skinning_backend"].value = 1 if backend == "gpu" else 0
        if self._use_gsplat:
            self._gsplat_palette = _torch.from_numpy(palette).to(self._gsplat_device)

    def update_bone_lines(self, bone_positions: np.ndarray, parents: np.ndarray, enabled: bool) -> None:
        if not enabled:
            self.bone_vertex_count = 0
            return

        verts: list[float] = []
        for j, p in enumerate(parents):
            if p < 0:
                continue
            a = bone_positions[p]
            b = bone_positions[j]
            c = (0.95, 0.82, 0.30)
            verts.extend([float(a[0]), float(a[1]), float(a[2]), *c])
            verts.extend([float(b[0]), float(b[1]), float(b[2]), *c])

        if not verts:
            self.bone_vertex_count = 0
            return

        data = np.asarray(verts, dtype=np.float32)
        needed = data.nbytes
        if needed > self.bone_vbo.size:
            self.bone_vbo.orphan(size=needed)
        self.bone_vbo.write(data.tobytes(), offset=0)
        self.bone_vertex_count = data.size // 6

    def update_ik_debug_lines(self, vertices: np.ndarray | None, enabled: bool) -> None:
        if (not enabled) or vertices is None or vertices.size == 0:
            self.ik_vertex_count = 0
            return

        data = np.asarray(vertices, dtype=np.float32)
        needed = data.nbytes
        if needed > self.ik_vbo.size:
            self.ik_vbo.orphan(size=needed)
        self.ik_vbo.write(data.tobytes(), offset=0)
        self.ik_vertex_count = data.size // 6

    def render(self, camera: Camera, show_bones: bool = False, show_ik_debug: bool = False) -> None:
        self.ctx.clear(0.025, 0.030, 0.045, 1.0)  # Dark cinematic background.

        eye = camera.position
        view = look_at(eye, camera.target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        proj = perspective(self.fov_y, self.width / self.height, 0.05, 50.0)
        mvp = proj @ view

        if self._use_gsplat:
            # gsplat produces a properly depth-sorted, alpha-composited image.
            self._render_gsplat(camera)
        else:
            # Fallback: ModernGL point-sprite Gaussians.
            point_scale = self.height / (2.0 * np.tan(self.fov_y * 0.5))
            self.program["u_mvp"].write(mvp.T.astype("f4").tobytes())
            self.program["u_view"].write(view.T.astype("f4").tobytes())
            self.program["u_point_scale"].value = float(point_scale)
            self.face_program["u_mvp"].write(mvp.T.astype("f4").tobytes())
            self.face_program["u_view"].write(view.T.astype("f4").tobytes())
            self.face_program["u_point_scale"].value = float(point_scale)
            if self.particle_count > 0:
                self.vao.render(mode=moderngl.POINTS, vertices=self.particle_count)
            if self.face_particle_count > 0:
                self.face_vao.render(mode=moderngl.POINTS, vertices=self.face_particle_count)

        # Draw orientation guides last as overlays (depth test off so they
        # are always visible regardless of the splat depth buffer).
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.line_program["u_mvp"].write(mvp.T.astype("f4").tobytes())
        self.grid_vao.render(mode=moderngl.LINES, vertices=self.grid_vertex_count)
        if show_bones and self.bone_vertex_count > 0:
            self.bone_vao.render(mode=moderngl.LINES, vertices=self.bone_vertex_count)
        if show_ik_debug and self.ik_vertex_count > 0:
            self.ik_vao.render(mode=moderngl.LINES, vertices=self.ik_vertex_count)
        self.ctx.enable(moderngl.DEPTH_TEST)

        pygame.display.flip()

    def shutdown(self) -> None:
        if self._use_gsplat:
            self._blit_vao.release()
            self._blit_vbo.release()
            self._blit_prog.release()
            self._gsplat_tex.release()
        self.ik_vao.release()
        self.ik_vbo.release()
        self.bone_vao.release()
        self.bone_vbo.release()
        self.grid_vao.release()
        self.grid_vbo.release()
        self.line_program.release()
        self.face_vao.release()
        self.face_vbo.release()
        self.face_program.release()
        self.vao.release()
        self.vbo.release()
        self.program.release()
        pygame.quit()
