#version 330 core

in vec3 in_bind_pos;
in vec3 in_cpu_pos;
in vec3 in_deform;
in vec3 in_color;
in float in_size;
in float in_brightness;
in vec4 in_bone_idx;
in vec4 in_bone_w;

uniform mat4 u_mvp;
uniform mat4 u_view;
uniform float u_point_scale;
uniform int u_skinning_backend; // 0=CPU, 1=GPU
uniform mat4 u_bone_mats[128];

out vec3 v_color;
out float v_brightness;
out float v_depth;

vec3 skin_gpu(vec3 bind_pos, vec4 idx, vec4 w) {
    ivec4 bi = ivec4(idx + 0.5);
    vec4 p = vec4(bind_pos, 1.0);
    vec4 s =
        (u_bone_mats[bi.x] * p) * w.x +
        (u_bone_mats[bi.y] * p) * w.y +
        (u_bone_mats[bi.z] * p) * w.z +
        (u_bone_mats[bi.w] * p) * w.w;
    return s.xyz;
}

void main() {
    vec3 world_pos = in_cpu_pos;
    if (u_skinning_backend == 1) {
        world_pos = skin_gpu(in_bind_pos, in_bone_idx, in_bone_w);
    }
    world_pos += in_deform;

    vec4 view_pos = u_view * vec4(world_pos, 1.0);
    float dist = max(0.001, -view_pos.z);

    gl_Position = u_mvp * vec4(world_pos, 1.0);
    gl_PointSize = clamp((in_size * 0.86 * u_point_scale) / dist, 1.2, 30.0);

    v_color = in_color;
    v_brightness = in_brightness;
    v_depth = dist;
}
