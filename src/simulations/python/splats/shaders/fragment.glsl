#version 330 core

in vec3 v_color;
in float v_brightness;
in float v_depth;

out vec4 fragColor;

void main() {
    // Each point sprite becomes a gaussian-like transparent splat.
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);

    float gaussian = exp(-6.6 * r2);
    float soft_edge = smoothstep(1.0, 0.24, r2);
    float edge_def = 1.0 + 0.20 * smoothstep(0.44, 0.88, r2);
    float alpha = gaussian * soft_edge * edge_def;
    // Keep distant splats visible; only gently attenuate very far points.
    alpha *= mix(1.0, 0.72, smoothstep(8.0, 30.0, v_depth));
    alpha = clamp(alpha, 0.0, 0.92);

    if (alpha < 0.02) {
        discard;
    }

    // Fake local normal from point sprite UV for simple lighting illusion.
    float z = sqrt(max(0.0, 1.0 - r2));
    vec3 approx_normal = normalize(vec3(p, z));
    vec3 light_dir = normalize(vec3(0.55, 1.1, 0.35));

    float ndl = max(0.20, dot(approx_normal, light_dir));
    float rim = pow(clamp(1.0 - approx_normal.z, 0.0, 1.0), 1.6);

    // Slightly darken far splats and gently boost near splats for depth contrast.
    float depth_contrast = mix(1.12, 0.62, smoothstep(0.8, 18.0, v_depth));
    float depth_fade = mix(1.0, 0.70, smoothstep(1.0, 24.0, v_depth));

    // Increase per-particle brightness separation so motion remains readable.
    float brightness_var = pow(clamp(v_brightness, 0.0, 1.25), 1.14);
    float brightness = (0.34 + 0.86 * brightness_var) * ndl * depth_fade;

    vec3 color = v_color * brightness;
    color *= depth_contrast;
    color += vec3(0.13, 0.15, 0.18) * rim;

    // Mild contrast enhancement around mid-tones.
    color = clamp((color - 0.5) * 1.14 + 0.5, 0.0, 1.0);
    fragColor = vec4(color, alpha);
}
