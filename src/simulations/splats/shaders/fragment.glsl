#version 330 core

in vec3 v_color;
in float v_brightness;
in float v_depth;

out vec4 fragColor;

void main() {
    // Each point sprite becomes a gaussian-like transparent splat.
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);

    float gaussian = exp(-4.5 * r2);
    float soft_edge = smoothstep(1.0, 0.32, r2);
    float alpha = gaussian * soft_edge * 0.92;

    if (alpha < 0.02) {
        discard;
    }

    // Fake local normal from point sprite UV for simple lighting illusion.
    float z = sqrt(max(0.0, 1.0 - r2));
    vec3 approx_normal = normalize(vec3(p, z));
    vec3 light_dir = normalize(vec3(0.55, 1.1, 0.35));

    float ndl = max(0.18, dot(approx_normal, light_dir));
    float depth_fade = clamp(1.25 - 0.013 * v_depth, 0.40, 1.0);
    float brightness = (0.45 + 0.55 * v_brightness) * ndl * depth_fade;

    vec3 color = v_color * brightness;
    fragColor = vec4(color, alpha);
}
