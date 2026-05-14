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
)


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
        default=6.8,
        help="Joint angular damping for velocity interpolation",
    )
    parser.add_argument(
        "--joint-stiffness",
        type=float,
        default=16.0,
        help="Joint stiffness driving velocity toward target rotations",
    )
    parser.add_argument(
        "--joint-max-velocity",
        type=float,
        default=8.0,
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
        default=2.2,
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

    renderer = GaussianSplatRenderer(width=1280, height=720, max_particles=100000, max_bones=128)
    camera = Camera(target=np.array([0.0, 1.0, 0.0], dtype=np.float32))

    particle_data = generate_human_particles(total_particles=14000, seed=5)
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
    secondary_motion = SecondaryMotionController(response=8.5)
    root_motion = RootMotionController(
        walk_speed=0.95,
        run_speed=1.85,
        inertia=args.root_inertia,
        friction=args.root_friction,
        accel_gain=6.8,
        turn_inertia=2.8,
        turn_damping=3.4,
        dir_smoothing=7.5,
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
