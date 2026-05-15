#version 330 core

in vec3 v_color;
in float v_brightness;
in float v_depth;

out vec4 fragColor;

void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);

    // Sharper face splats preserve eye, mouth, and nasal structure.
    float gaussian = exp(-8.4 * r2);
    float edge = smoothstep(1.0, 0.18, r2);
    float alpha = gaussian * edge;
    alpha *= mix(1.0, 0.75, smoothstep(6.0, 24.0, v_depth));
    alpha = clamp(alpha, 0.0, 0.97);

    if (alpha < 0.03) {
        discard;
    }

    float z = sqrt(max(0.0, 1.0 - r2));
    vec3 n = normalize(vec3(p.x * 0.95, p.y * 0.95, z));
    vec3 key_light = normalize(vec3(0.30, 0.90, 0.55));
    vec3 fill_light = normalize(vec3(-0.45, 0.35, 0.70));

    float key = max(0.12, dot(n, key_light));
    float fill = max(0.0, dot(n, fill_light));
    float rim = pow(clamp(1.0 - n.z, 0.0, 1.0), 2.0);

    float depth_contrast = mix(1.10, 0.72, smoothstep(0.7, 14.0, v_depth));
    float bright = (0.30 + 0.90 * pow(clamp(v_brightness, 0.0, 1.35), 1.08));

    vec3 color = v_color * bright * (0.72 * key + 0.28 * fill + 0.22);
    color *= depth_contrast;
    color += vec3(0.16, 0.17, 0.19) * rim;

    // Face-focused contrast enhancement for feature readability.
    color = clamp((color - 0.5) * 1.20 + 0.5, 0.0, 1.0);

    fragColor = vec4(color, alpha);
}
