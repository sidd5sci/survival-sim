import type { GaussianSplatData, Segmentation } from "./plyInspector";

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];

export interface CanonicalAvatar {
  positions: Float32Array;
  quaternions: Float32Array;
  scales: Float32Array;
  center: Vec3;
  scaleFactor: number;
  scaleFactors: Vec3;
  canonicalFromWorldQuat: Quat;
  worldFromCanonicalQuat: Quat;
  basis: {
    right: Vec3;
    up: Vec3;
    forward: Vec3;
  };
  height: number;
}

export interface SkeletonBone {
  id: number;
  name: string;
  parentId: number;
  localBindPosition: Vec3;
  localBindRotation: Quat;
  worldBindPosition: Vec3;
  worldBindRotation: Quat;
  localPosePosition: Vec3;
  localPoseRotation: Quat;
  worldPosePosition: Vec3;
  worldPoseRotation: Quat;
}

export interface SkeletonRig {
  bones: SkeletonBone[];
}

export interface SkinningData {
  influencesPerSplat: number;
  boneIds: Uint16Array;
  weights: Float32Array;
}

export interface WeightDebugData {
  dominantBone: Uint16Array;
  dominantWeight: Float32Array;
  blendScore: Float32Array;
  boneColors: Float32Array;
}

export interface TestControls {
  armRotateDeg: number;
  spineBendDeg: number;
  headRotateDeg: number;
  torsoTwistDeg: number;
}

export interface DeformedSplats {
  positions: Float32Array;
  quaternions: Float32Array;
  scales: Float32Array;
}

export interface AnimationReadyAvatar {
  formatVersion: string;
  sourceFile: string;
  canonical: {
    center: Vec3;
    scaleFactor: number;
    basis: {
      right: Vec3;
      up: Vec3;
      forward: Vec3;
    };
    normalizedHeight: number;
  };
  skeleton: {
    bones: Array<{
      id: number;
      name: string;
      parentId: number;
      localBindPosition: Vec3;
      localBindRotation: Quat;
      worldBindPosition: Vec3;
      worldBindRotation: Quat;
      localPosePosition: Vec3;
      localPoseRotation: Quat;
      worldPosePosition: Vec3;
      worldPoseRotation: Quat;
    }>;
  };
  skinning: {
    influencesPerSplat: number;
    boneIds: number[];
    weights: number[];
  };
}

const EPS = 1e-8;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / Math.max(EPS, edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize(v: Vec3): Vec3 {
  const n = Math.max(EPS, length(v));
  return [v[0] / n, v[1] / n, v[2] / n];
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function mul(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function lerp(a: Vec3, b: Vec3, t: number): Vec3 {
  return [a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t, a[2] * (1 - t) + b[2] * t];
}

function identityQuat(): Quat {
  return [0, 0, 0, 1];
}

function quatNormalize(q: Quat): Quat {
  const n = Math.max(EPS, Math.hypot(q[0], q[1], q[2], q[3]));
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function quatMultiply(a: Quat, b: Quat): Quat {
  return quatNormalize([
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ]);
}

function quatConjugate(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

function quatInverse(q: Quat): Quat {
  return quatConjugate(quatNormalize(q));
}

function quatDot(a: Quat, b: Quat): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

function quatNlerp(a: Quat, b: Quat, t: number): Quat {
  const q2: Quat = quatDot(a, b) < 0 ? [-b[0], -b[1], -b[2], -b[3]] : b;
  return quatNormalize([
    a[0] * (1 - t) + q2[0] * t,
    a[1] * (1 - t) + q2[1] * t,
    a[2] * (1 - t) + q2[2] * t,
    a[3] * (1 - t) + q2[3] * t,
  ]);
}

function quatWeightedAverage(values: Array<{ q: Quat; w: number }>): Quat {
  if (values.length === 0) return identityQuat();
  let out: Quat = values[0].q;
  let total = Math.max(EPS, values[0].w);
  for (let i = 1; i < values.length; i += 1) {
    const v = values[i];
    total += Math.max(0, v.w);
    out = quatNlerp(out, v.q, v.w / total);
  }
  return quatNormalize(out);
}

function rotateByQuat(v: Vec3, q: Quat): Vec3 {
  const p: Quat = [v[0], v[1], v[2], 0];
  const r = quatMultiply(quatMultiply(q, p), quatInverse(q));
  return [r[0], r[1], r[2]];
}

function quatFromAxisAngle(axis: Vec3, radians: number): Quat {
  const n = normalize(axis);
  const s = Math.sin(radians * 0.5);
  return quatNormalize([n[0] * s, n[1] * s, n[2] * s, Math.cos(radians * 0.5)]);
}

function quatFromBasis(right: Vec3, up: Vec3, forward: Vec3): Quat {
  const m00 = right[0];
  const m01 = up[0];
  const m02 = forward[0];
  const m10 = right[1];
  const m11 = up[1];
  const m12 = forward[1];
  const m20 = right[2];
  const m21 = up[2];
  const m22 = forward[2];

  const trace = m00 + m11 + m22;
  let qx = 0;
  let qy = 0;
  let qz = 0;
  let qw = 1;

  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    qw = 0.25 * s;
    qx = (m21 - m12) / s;
    qy = (m02 - m20) / s;
    qz = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    qw = (m21 - m12) / s;
    qx = 0.25 * s;
    qy = (m01 + m10) / s;
    qz = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    qw = (m02 - m20) / s;
    qx = (m01 + m10) / s;
    qy = 0.25 * s;
    qz = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    qw = (m10 - m01) / s;
    qx = (m02 + m20) / s;
    qy = (m12 + m21) / s;
    qz = 0.25 * s;
  }

  return quatNormalize([qx, qy, qz, qw]);
}

function toRadians(deg: number): number {
  return (deg * Math.PI) / 180;
}

function centroidForLabel(positions: Float32Array, labels: Uint8Array, labelValue: number): Vec3 {
  let cx = 0;
  let cy = 0;
  let cz = 0;
  let n = 0;
  for (let i = 0; i < labels.length; i += 1) {
    if (labels[i] !== labelValue) continue;
    cx += positions[i * 3];
    cy += positions[i * 3 + 1];
    cz += positions[i * 3 + 2];
    n += 1;
  }
  if (n <= 0) return [0, 0, 1];
  return [cx / n, cy / n, cz / n];
}

function covarianceMatrix(positions: Float32Array, center: Vec3): number[][] {
  const c = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const n = Math.max(1, positions.length / 3);

  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i] - center[0];
    const y = positions[i + 1] - center[1];
    const z = positions[i + 2] - center[2];

    c[0][0] += x * x;
    c[0][1] += x * y;
    c[0][2] += x * z;
    c[1][0] += y * x;
    c[1][1] += y * y;
    c[1][2] += y * z;
    c[2][0] += z * x;
    c[2][1] += z * y;
    c[2][2] += z * z;
  }

  for (let r = 0; r < 3; r += 1) {
    for (let col = 0; col < 3; col += 1) {
      c[r][col] /= n;
    }
  }

  return c;
}

function matVec3(m: number[][], v: Vec3): Vec3 {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

function powerIteration(m: number[][], seed: Vec3, iterations = 18): Vec3 {
  let v = normalize(seed);
  for (let i = 0; i < iterations; i += 1) {
    v = normalize(matVec3(m, v));
  }
  return v;
}

function projectPlane(v: Vec3, normal: Vec3): Vec3 {
  const d = dot(v, normal);
  return sub(v, mul(normal, d));
}

function computeBounding(positions: Float32Array): { min: Vec3; max: Vec3 } {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    if (x < min[0]) min[0] = x;
    if (y < min[1]) min[1] = y;
    if (z < min[2]) min[2] = z;
    if (x > max[0]) max[0] = x;
    if (y > max[1]) max[1] = y;
    if (z > max[2]) max[2] = z;
  }
  return { min, max };
}

export function prepareCanonicalAvatar(data: GaussianSplatData, segmentation: Segmentation): CanonicalAvatar {
  const box = computeBounding(data.positions);
  const center: Vec3 = [
    (box.min[0] + box.max[0]) * 0.5,
    (box.min[1] + box.max[1]) * 0.5,
    (box.min[2] + box.max[2]) * 0.5,
  ];

  const cov = covarianceMatrix(data.positions, center);
  const e1 = powerIteration(cov, [1, 0.1, 0.1]);
  const e2Raw = powerIteration(cov, [0.1, 1, 0.3]);
  const e2 = normalize(sub(e2Raw, mul(e1, dot(e2Raw, e1))));
  const e3 = normalize(cross(e1, e2));

  const faceCentroid = centroidForLabel(data.positions, segmentation.labels, 1);
  const bodyCentroid = centroidForLabel(data.positions, segmentation.labels, 3);
  const headDir = normalize(sub(faceCentroid, bodyCentroid));

  const candidates: Vec3[] = [e1, e2, e3];
  let up = e1;
  let best = -Infinity;
  for (const c of candidates) {
    const score = Math.abs(dot(c, headDir));
    if (score > best) {
      best = score;
      up = dot(c, headDir) >= 0 ? c : mul(c, -1);
    }
  }

  let forward = projectPlane(sub(faceCentroid, center), up);
  if (length(forward) < 1e-5) {
    forward = projectPlane(e2, up);
  }
  forward = normalize(forward);
  let right = normalize(cross(up, forward));
  forward = normalize(cross(right, up));

  const canonicalQuat = quatInverse(quatFromBasis(right, up, forward));

  const centered = new Float32Array(data.positions.length);
  for (let i = 0; i < data.count; i += 1) {
    const p = sub(
      [data.positions[i * 3], data.positions[i * 3 + 1], data.positions[i * 3 + 2]],
      center,
    );
    const cpos = rotateByQuat(p, canonicalQuat);
    centered[i * 3] = cpos[0];
    centered[i * 3 + 1] = cpos[1];
    centered[i * 3 + 2] = cpos[2];
  }

  const cBox = computeBounding(centered);
  const rawHeight = Math.max(1e-6, cBox.max[1] - cBox.min[1]);
  const targetHeight = 1.75;
  const scaleFactor = targetHeight / rawHeight;
  const scaleFactors: Vec3 = [scaleFactor, scaleFactor, scaleFactor];

  const positions = new Float32Array(centered.length);
  for (let i = 0; i < centered.length; i += 1) {
    positions[i] = centered[i] * scaleFactor;
  }

  const scales = new Float32Array(data.scalesWorld.length);
  for (let i = 0; i < scales.length; i += 1) {
    scales[i] = data.scalesWorld[i] * scaleFactor;
  }

  const quaternions = new Float32Array(data.quaternions.length);
  for (let i = 0; i < data.count; i += 1) {
    const q: Quat = [
      data.quaternions[i * 4],
      data.quaternions[i * 4 + 1],
      data.quaternions[i * 4 + 2],
      data.quaternions[i * 4 + 3],
    ];
    const out = quatMultiply(canonicalQuat, q);
    quaternions[i * 4] = out[0];
    quaternions[i * 4 + 1] = out[1];
    quaternions[i * 4 + 2] = out[2];
    quaternions[i * 4 + 3] = out[3];
  }

  return {
    positions,
    quaternions,
    scales,
    center,
    scaleFactor,
    scaleFactors,
    canonicalFromWorldQuat: canonicalQuat,
    worldFromCanonicalQuat: quatInverse(canonicalQuat),
    basis: { right, up, forward },
    height: targetHeight,
  };
}

interface BoneSeed {
  name: string;
  parent: string | null;
  worldBindPosition: Vec3;
}

const REQUIRED_BONES = [
  "pelvis",
  "spine",
  "chest",
  "neck",
  "head",
  "leftShoulder",
  "leftElbow",
  "leftWrist",
  "rightShoulder",
  "rightElbow",
  "rightWrist",
  "leftHip",
  "leftKnee",
  "leftAnkle",
  "rightHip",
  "rightKnee",
  "rightAnkle",
] as const;

export function fitHumanoidSkeleton(canonical: CanonicalAvatar): SkeletonRig {
  const box = computeBounding(canonical.positions);
  const min = box.min;
  const max = box.max;
  const cx = (min[0] + max[0]) * 0.5;
  const sx = max[0] - min[0];
  const sy = max[1] - min[1];
  const zFront = max[2] - (max[2] - min[2]) * 0.2;
  const zMid = (min[2] + max[2]) * 0.5;

  const y = (t: number) => min[1] + sy * t;
  const x = (t: number) => cx + sx * t;

  const seeds: BoneSeed[] = [
    { name: "pelvis", parent: null, worldBindPosition: [cx, y(0.48), zMid] },
    { name: "spine", parent: "pelvis", worldBindPosition: [cx, y(0.61), zMid] },
    { name: "chest", parent: "spine", worldBindPosition: [cx, y(0.72), zMid] },
    { name: "neck", parent: "chest", worldBindPosition: [cx, y(0.82), zMid] },
    { name: "head", parent: "neck", worldBindPosition: [cx, y(0.90), zFront] },

    { name: "leftShoulder", parent: "chest", worldBindPosition: [x(-0.14), y(0.75), zMid] },
    { name: "leftElbow", parent: "leftShoulder", worldBindPosition: [x(-0.23), y(0.68), zMid] },
    { name: "leftWrist", parent: "leftElbow", worldBindPosition: [x(-0.30), y(0.62), zMid] },

    { name: "rightShoulder", parent: "chest", worldBindPosition: [x(0.14), y(0.75), zMid] },
    { name: "rightElbow", parent: "rightShoulder", worldBindPosition: [x(0.23), y(0.68), zMid] },
    { name: "rightWrist", parent: "rightElbow", worldBindPosition: [x(0.30), y(0.62), zMid] },

    { name: "leftHip", parent: "pelvis", worldBindPosition: [x(-0.08), y(0.40), zMid] },
    { name: "leftKnee", parent: "leftHip", worldBindPosition: [x(-0.08), y(0.24), zMid] },
    { name: "leftAnkle", parent: "leftKnee", worldBindPosition: [x(-0.08), y(0.09), zFront] },

    { name: "rightHip", parent: "pelvis", worldBindPosition: [x(0.08), y(0.40), zMid] },
    { name: "rightKnee", parent: "rightHip", worldBindPosition: [x(0.08), y(0.24), zMid] },
    { name: "rightAnkle", parent: "rightKnee", worldBindPosition: [x(0.08), y(0.09), zFront] },
  ];

  const idByName = new Map<string, number>();
  for (let i = 0; i < seeds.length; i += 1) {
    idByName.set(seeds[i].name, i);
  }

  const parentIds = seeds.map((seed) => (seed.parent ? (idByName.get(seed.parent) ?? -1) : -1));
  const childIds: number[][] = Array.from({ length: seeds.length }, () => []);
  for (let i = 0; i < seeds.length; i += 1) {
    const p = parentIds[i];
    if (p >= 0) childIds[p].push(i);
  }

  const worldBindRotations: Quat[] = new Array(seeds.length);
  const worldForward = (id: number): Vec3 => {
    const children = childIds[id];
    const selfPos = seeds[id].worldBindPosition;
    if (children.length > 0) {
      const acc: Vec3 = [0, 0, 0];
      let n = 0;
      for (const childId of children) {
        const cPos = seeds[childId].worldBindPosition;
        const dir = sub(cPos, selfPos);
        const d = length(dir);
        if (d <= EPS) continue;
        const u = mul(dir, 1 / d);
        acc[0] += u[0];
        acc[1] += u[1];
        acc[2] += u[2];
        n += 1;
      }
      if (n > 0) {
        const avg: Vec3 = [acc[0] / n, acc[1] / n, acc[2] / n];
        if (length(avg) > EPS) return normalize(avg);
      }
    }
    const p = parentIds[id];
    if (p >= 0) {
      const pPos = seeds[p].worldBindPosition;
      const f = sub(selfPos, pPos);
      if (length(f) > EPS) return normalize(f);
    }
    return [0, 1, 0];
  };

  for (let id = 0; id < seeds.length; id += 1) {
    const forward = worldForward(id);
    const parentId = parentIds[id];
    const parentRot = parentId >= 0 ? worldBindRotations[parentId] : identityQuat();
    const parentUp = rotateByQuat([0, 1, 0], parentRot);
    const parentRight = rotateByQuat([1, 0, 0], parentRot);

    let upRef = sub(parentUp, mul(forward, dot(parentUp, forward)));
    if (length(upRef) <= EPS) {
      upRef = sub(parentRight, mul(forward, dot(parentRight, forward)));
    }
    if (length(upRef) <= EPS) {
      const fallback: Vec3 = Math.abs(forward[1]) < 0.98 ? [0, 1, 0] : [0, 0, 1];
      upRef = sub(fallback, mul(forward, dot(fallback, forward)));
    }

    const up = normalize(upRef);
    let right = cross(up, forward);
    if (length(right) <= EPS) {
      right = cross([1, 0, 0], forward);
      if (length(right) <= EPS) right = cross([0, 0, 1], forward);
    }
    right = normalize(right);
    const orthoUp = normalize(cross(forward, right));
    worldBindRotations[id] = quatFromBasis(right, orthoUp, forward);
  }

  const bones: SkeletonBone[] = seeds.map((seed, id) => {
    const parentId = parentIds[id];
    const parentWorldPos: Vec3 = parentId >= 0 ? seeds[parentId].worldBindPosition : [0, 0, 0];
    const parentWorldRot: Quat = parentId >= 0 ? worldBindRotations[parentId] : identityQuat();
    const invParentWorldRot = quatInverse(parentWorldRot);
    const worldOffset: Vec3 = parentId >= 0 ? sub(seed.worldBindPosition, parentWorldPos) : seed.worldBindPosition;
    const localBindPosition: Vec3 = parentId >= 0 ? rotateByQuat(worldOffset, invParentWorldRot) : worldOffset;

    const worldBindRotation = worldBindRotations[id];
    const localBindRotation =
      parentId >= 0
        ? quatMultiply(invParentWorldRot, worldBindRotation)
        : worldBindRotation;

    return {
      id,
      name: seed.name,
      parentId,
      localBindPosition,
      localBindRotation,
      worldBindPosition: seed.worldBindPosition,
      worldBindRotation,
      localPosePosition: localBindPosition,
      localPoseRotation: localBindRotation,
      worldPosePosition: seed.worldBindPosition,
      worldPoseRotation: worldBindRotation,
    };
  });

  for (const name of REQUIRED_BONES) {
    if (!idByName.has(name)) {
      throw new Error(`Required bone missing after skeleton fit: ${name}`);
    }
  }

  return { bones };
}

function findBoneId(rig: SkeletonRig, name: string): number {
  const idx = rig.bones.findIndex((b) => b.name === name);
  if (idx < 0) throw new Error(`Bone not found: ${name}`);
  return idx;
}

function safeFindBoneIdByMap(nameToId: Map<string, number>, name: string): number {
  const id = nameToId.get(name);
  return typeof id === "number" ? id : -1;
}

function isShoulderElbowWrist(name: string): boolean {
  return name.includes("Shoulder") || name.includes("Elbow") || name.includes("Wrist");
}

function isHipKneeAnkle(name: string): boolean {
  return name.includes("Hip") || name.includes("Knee") || name.includes("Ankle");
}

function regionBias(label: number, name: string): number {
  if (label === 1) {
    if (name === "head") return 1.55;
    if (name === "neck") return 1.35;
    if (name === "chest") return 0.95;
    return 0.25;
  }

  if (label === 2) {
    if (name === "head") return 1.25;
    if (name === "neck") return 1.2;
    if (name === "chest") return 1.25;
    if (name === "spine") return 1.2;
    if (name.includes("Shoulder")) return 0.95;
    return 0.45;
  }

  if (name === "pelvis") return 1.2;
  if (name === "spine") return 0.95;
  if (name === "chest") return 0.8;
  if (isHipKneeAnkle(name)) return 1.22;
  if (isShoulderElbowWrist(name)) return 1.08;
  return 0.6;
}

function sideBias(x: number, name: string): number {
  if (name.startsWith("left") && x < 0) return 1.2;
  if (name.startsWith("right") && x > 0) return 1.2;
  if (name.startsWith("left") || name.startsWith("right")) return 0.58;
  return 1;
}

type PreparedTransitionPair = {
  aId: number;
  bId: number;
  a: Vec3;
  b: Vec3;
  radius: number;
  gain: number;
};

function prepareTransitionPairs(rig: SkeletonRig): PreparedTransitionPair[] {
  const nameToId = new Map<string, number>();
  for (const bone of rig.bones) nameToId.set(bone.name, bone.id);

  const specs: Array<[string, string, number, number]> = [
    ["leftShoulder", "leftElbow", 0.11, 0.46],
    ["rightShoulder", "rightElbow", 0.11, 0.46],
    ["leftElbow", "leftWrist", 0.09, 0.43],
    ["rightElbow", "rightWrist", 0.09, 0.43],
    ["leftHip", "leftKnee", 0.12, 0.5],
    ["rightHip", "rightKnee", 0.12, 0.5],
    ["leftKnee", "leftAnkle", 0.1, 0.45],
    ["rightKnee", "rightAnkle", 0.1, 0.45],
    ["pelvis", "spine", 0.12, 0.38],
    ["chest", "neck", 0.1, 0.36],
  ];

  const prepared: PreparedTransitionPair[] = [];
  for (const [aName, bName, radius, gain] of specs) {
    const aId = safeFindBoneIdByMap(nameToId, aName);
    const bId = safeFindBoneIdByMap(nameToId, bName);
    if (aId < 0 || bId < 0) continue;
    prepared.push({
      aId,
      bId,
      a: rig.bones[aId].worldBindPosition,
      b: rig.bones[bId].worldBindPosition,
      radius,
      gain,
    });
  }

  return prepared;
}

function transitionBoost(p: Vec3, boneCount: number, pairs: PreparedTransitionPair[]): number[] {
  const out = Array(boneCount).fill(1);

  for (const pair of pairs) {
    const { aId, bId, a, b, radius, gain } = pair;
    const ab = sub(b, a);
    const ap = sub(p, a);
    const t = clamp(dot(ap, ab) / Math.max(EPS, dot(ab, ab)), 0, 1);
    const closest = add(a, mul(ab, t));
    const d = length(sub(p, closest));
    const boost = 1 + gain * (1 - smoothstep(radius * 0.35, radius, d));
    out[aId] *= boost;
    out[bId] *= boost;
  }

  return out;
}

function smoothSkinningPass(
  positions: Float32Array,
  labels: Uint8Array,
  boneIds: Uint16Array,
  weights: Float32Array,
  influencesPerSplat: number,
  sameRegionBlend: number,
  crossRegionBlend: number,
) {
  const n = labels.length;
  const grid = new Map<string, number[]>();

  const box = computeBounding(positions);
  const cell = Math.max(0.018, (box.max[1] - box.min[1]) / 32);

  const keyFor = (x: number, y: number, z: number) => {
    const ix = Math.floor((x - box.min[0]) / cell);
    const iy = Math.floor((y - box.min[1]) / cell);
    const iz = Math.floor((z - box.min[2]) / cell);
    return `${ix}|${iy}|${iz}`;
  };

  for (let i = 0; i < n; i += 1) {
    const key = keyFor(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key)?.push(i);
  }

  const outWeights = new Float32Array(weights);

  for (let i = 0; i < n; i += 1) {
    const px = positions[i * 3];
    const py = positions[i * 3 + 1];
    const pz = positions[i * 3 + 2];

    const ix = Math.floor((px - box.min[0]) / cell);
    const iy = Math.floor((py - box.min[1]) / cell);
    const iz = Math.floor((pz - box.min[2]) / cell);

    const localNeighbors: number[] = [];
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dz = -1; dz <= 1; dz += 1) {
          const key = `${ix + dx}|${iy + dy}|${iz + dz}`;
          const list = grid.get(key);
          if (list) localNeighbors.push(...list);
        }
      }
    }

    if (localNeighbors.length <= 1) continue;

    const accum = new Map<number, number>();
    for (let j = 0; j < influencesPerSplat; j += 1) {
      const bid = boneIds[i * influencesPerSplat + j];
      const w = outWeights[i * influencesPerSplat + j];
      accum.set(bid, w * 0.86);
    }

    let sameCount = 0;
    let crossCount = 0;
    for (const ni of localNeighbors) {
      if (ni === i) continue;
      if (labels[ni] === labels[i]) sameCount += 1;
      else crossCount += 1;
    }

    for (const ni of localNeighbors) {
      if (ni === i) continue;
      const sameRegion = labels[ni] === labels[i];
      const denom = sameRegion ? Math.max(1, sameCount) : Math.max(1, crossCount);
      const blend = sameRegion ? sameRegionBlend : crossRegionBlend;
      for (let j = 0; j < influencesPerSplat; j += 1) {
        const bid = boneIds[ni * influencesPerSplat + j];
        const w = outWeights[ni * influencesPerSplat + j] * blend / denom;
        accum.set(bid, (accum.get(bid) ?? 0) + w);
      }
    }

    const sorted = Array.from(accum.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, influencesPerSplat);

    let total = 0;
    for (const [, w] of sorted) total += w;
    total = Math.max(EPS, total);

    for (let j = 0; j < influencesPerSplat; j += 1) {
      const next = sorted[j] ?? sorted[0];
      boneIds[i * influencesPerSplat + j] = next[0];
      outWeights[i * influencesPerSplat + j] = next[1] / total;
    }
  }

  weights.set(outWeights);
}

export function assignSkinningWeights(
  canonical: CanonicalAvatar,
  rig: SkeletonRig,
  segmentation: Segmentation,
): SkinningData {
  const influencesPerSplat = 4;
  const n = segmentation.labels.length;
  const boneIds = new Uint16Array(n * influencesPerSplat);
  const weights = new Float32Array(n * influencesPerSplat);

  const box = computeBounding(canonical.positions);
  const sigma = Math.max(0.07, (box.max[1] - box.min[1]) * 0.12);
  const preparedPairs = prepareTransitionPairs(rig);

  for (let i = 0; i < n; i += 1) {
    const p: Vec3 = [
      canonical.positions[i * 3],
      canonical.positions[i * 3 + 1],
      canonical.positions[i * 3 + 2],
    ];
    const label = segmentation.labels[i];
    const x = p[0];

    const boosts = transitionBoost(p, rig.bones.length, preparedPairs);
    const scored = rig.bones.map((bone) => {
      const bp = bone.worldBindPosition;
      const d = length(sub(p, bp));
      const base = Math.exp(-(d * d) / (2 * sigma * sigma));
      const rb = regionBias(label, bone.name);
      const sb = sideBias(x, bone.name);
      const transition = boosts[bone.id];
      const score = base * rb * sb * transition;
      return { bid: bone.id, score };
    });

    const top = scored.sort((a, b) => b.score - a.score).slice(0, influencesPerSplat);

    let total = 0;
    for (const t of top) total += t.score;
    total = Math.max(EPS, total);

    for (let j = 0; j < influencesPerSplat; j += 1) {
      const t = top[j] ?? top[0];
      boneIds[i * influencesPerSplat + j] = t.bid;
      weights[i * influencesPerSplat + j] = t.score / total;
    }
  }

  smoothSkinningPass(canonical.positions, segmentation.labels, boneIds, weights, influencesPerSplat, 0.16, 0.05);
  smoothSkinningPass(canonical.positions, segmentation.labels, boneIds, weights, influencesPerSplat, 0.12, 0.04);

  return {
    influencesPerSplat,
    boneIds,
    weights,
  };
}

function buildBoneColorLUT(count: number): Float32Array {
  const lut = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    const h = (i * 0.61803398875) % 1;
    const s = 0.62;
    const v = 0.95;
    const k = (n: number) => (n + h * 6) % 6;
    const f = (n: number) => v - v * s * Math.max(0, Math.min(Math.min(k(n), 4 - k(n)), 1));
    lut[i * 3] = f(5);
    lut[i * 3 + 1] = f(3);
    lut[i * 3 + 2] = f(1);
  }
  return lut;
}

export function buildWeightDebugData(skinning: SkinningData, boneCount: number): WeightDebugData {
  const n = skinning.weights.length / skinning.influencesPerSplat;
  const dominantBone = new Uint16Array(n);
  const dominantWeight = new Float32Array(n);
  const blendScore = new Float32Array(n);
  const boneColors = buildBoneColorLUT(boneCount);

  for (let i = 0; i < n; i += 1) {
    let maxW = -1;
    let maxId = 0;
    let entropy = 0;

    for (let j = 0; j < skinning.influencesPerSplat; j += 1) {
      const w = skinning.weights[i * skinning.influencesPerSplat + j];
      const bid = skinning.boneIds[i * skinning.influencesPerSplat + j];
      if (w > maxW) {
        maxW = w;
        maxId = bid;
      }
      entropy += w > EPS ? -w * Math.log(w) : 0;
    }

    dominantBone[i] = maxId;
    dominantWeight[i] = maxW;
    blendScore[i] = clamp(entropy / Math.log(skinning.influencesPerSplat), 0, 1);
  }

  return { dominantBone, dominantWeight, blendScore, boneColors };
}

export function applyTestPose(rig: SkeletonRig, controls: TestControls): SkeletonRig {
  const next: SkeletonRig = {
    bones: rig.bones.map((b) => ({ ...b })),
  };

  const arm = toRadians(clamp(controls.armRotateDeg, -80, 80));
  const spine = toRadians(clamp(controls.spineBendDeg, -42, 42));
  const head = toRadians(clamp(controls.headRotateDeg, -85, 85));
  const twist = toRadians(clamp(controls.torsoTwistDeg, -65, 65));

  const leftShoulder = findBoneId(next, "leftShoulder");
  const rightShoulder = findBoneId(next, "rightShoulder");
  const spineId = findBoneId(next, "spine");
  const chestId = findBoneId(next, "chest");
  const neckId = findBoneId(next, "neck");
  const headId = findBoneId(next, "head");

  next.bones[leftShoulder].localPoseRotation = quatMultiply(
    quatFromAxisAngle([0, 0, 1], -arm),
    next.bones[leftShoulder].localBindRotation,
  );
  next.bones[rightShoulder].localPoseRotation = quatMultiply(
    quatFromAxisAngle([0, 0, 1], arm),
    next.bones[rightShoulder].localBindRotation,
  );

  next.bones[spineId].localPoseRotation = quatMultiply(
    quatMultiply(
      quatFromAxisAngle([1, 0, 0], spine * 0.45),
      quatFromAxisAngle([0, 1, 0], twist * 0.32),
    ),
    next.bones[spineId].localBindRotation,
  );

  next.bones[chestId].localPoseRotation = quatMultiply(
    quatMultiply(
      quatFromAxisAngle([1, 0, 0], spine * 0.40),
      quatFromAxisAngle([0, 1, 0], twist * 0.55),
    ),
    next.bones[chestId].localBindRotation,
  );

  next.bones[neckId].localPoseRotation = quatMultiply(
    quatMultiply(
      quatFromAxisAngle([1, 0, 0], spine * 0.15),
      quatFromAxisAngle([0, 1, 0], twist * 0.25),
    ),
    next.bones[neckId].localBindRotation,
  );

  next.bones[headId].localPoseRotation = quatMultiply(
    quatFromAxisAngle([0, 1, 0], head),
    next.bones[headId].localBindRotation,
  );

  for (let i = 0; i < next.bones.length; i += 1) {
    const b = next.bones[i];
    if (b.parentId < 0) {
      b.worldPoseRotation = b.localPoseRotation;
      b.worldPosePosition = b.localPosePosition;
      continue;
    }

    const parent = next.bones[b.parentId];
    b.worldPoseRotation = quatMultiply(parent.worldPoseRotation, b.localPoseRotation);
    b.worldPosePosition = add(parent.worldPosePosition, rotateByQuat(b.localPosePosition, parent.worldPoseRotation));
  }

  return next;
}

function correctiveJointBlend(point: Vec3, rig: SkeletonRig): number {
  const joints = [
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftKnee",
    "rightKnee",
    "neck",
    "pelvis",
  ].map((name) => rig.bones[findBoneId(rig, name)].worldBindPosition);

  let minD = Infinity;
  for (const j of joints) {
    minD = Math.min(minD, length(sub(point, j)));
  }

  // Higher correction near joints, fades in body core/extremities.
  return 1 - smoothstep(0.02, 0.17, minD);
}

function conservativeScaleFactor(
  p: Vec3,
  rigBind: SkeletonRig,
  rigPose: SkeletonRig,
  skinning: SkinningData,
  index: number,
): number {
  let delta = 0;
  for (let j = 0; j < skinning.influencesPerSplat; j += 1) {
    const bid = skinning.boneIds[index * skinning.influencesPerSplat + j];
    const w = skinning.weights[index * skinning.influencesPerSplat + j];
    const b0 = rigBind.bones[bid].worldBindPosition;
    const b1 = rigPose.bones[bid].worldPosePosition;
    const d0 = length(sub(p, b0));
    const d1 = length(sub(p, b1));
    const ratio = d0 > EPS ? d1 / d0 : 1;
    delta += (ratio - 1) * w;
  }

  // Limit stretch/compression to preserve identity and continuity.
  return clamp(1 + delta * 0.18, 0.88, 1.12);
}

export function deformSplats(
  canonical: CanonicalAvatar,
  rigBind: SkeletonRig,
  rigPose: SkeletonRig,
  skinning: SkinningData,
): DeformedSplats {
  const n = canonical.positions.length / 3;
  const outPos = new Float32Array(canonical.positions.length);
  const outQuat = new Float32Array(canonical.quaternions.length);
  const outScale = new Float32Array(canonical.scales.length);

  for (let i = 0; i < n; i += 1) {
    const p: Vec3 = [
      canonical.positions[i * 3],
      canonical.positions[i * 3 + 1],
      canonical.positions[i * 3 + 2],
    ];

    const weightedPos: Array<{ p: Vec3; w: number }> = [];
    const weightedRot: Array<{ q: Quat; w: number }> = [];

    for (let j = 0; j < skinning.influencesPerSplat; j += 1) {
      const bid = skinning.boneIds[i * skinning.influencesPerSplat + j];
      const w = skinning.weights[i * skinning.influencesPerSplat + j];
      const bindBone = rigBind.bones[bid];
      const poseBone = rigPose.bones[bid];

      const pLocal = rotateByQuat(sub(p, bindBone.worldBindPosition), quatInverse(bindBone.worldBindRotation));
      const pPose = add(rotateByQuat(pLocal, poseBone.worldPoseRotation), poseBone.worldPosePosition);
      weightedPos.push({ p: pPose, w });

      const delta = quatMultiply(poseBone.worldPoseRotation, quatInverse(bindBone.worldBindRotation));
      weightedRot.push({ q: delta, w });
    }

    let pos = [0, 0, 0] as Vec3;
    for (const v of weightedPos) pos = add(pos, mul(v.p, v.w));

    // Pose-space corrective deformation around joints to reduce collapse/twist/stretch artifacts.
    const corr = correctiveJointBlend(p, rigBind);
    if (corr > EPS) {
      const core = weightedPos[0].p;
      pos = lerp(pos, core, corr * 0.17);
    }

    outPos[i * 3] = pos[0];
    outPos[i * 3 + 1] = pos[1];
    outPos[i * 3 + 2] = pos[2];

    const qDelta = quatWeightedAverage(weightedRot);
    const qBind: Quat = [
      canonical.quaternions[i * 4],
      canonical.quaternions[i * 4 + 1],
      canonical.quaternions[i * 4 + 2],
      canonical.quaternions[i * 4 + 3],
    ];

    // Covariance deformation: orientation transforms follow weighted bone rotation.
    let qOut = quatMultiply(qDelta, qBind);

    // Joint stabilization in orientation space.
    if (corr > EPS) {
      qOut = quatNlerp(qOut, qBind, corr * 0.13);
    }

    outQuat[i * 4] = qOut[0];
    outQuat[i * 4 + 1] = qOut[1];
    outQuat[i * 4 + 2] = qOut[2];
    outQuat[i * 4 + 3] = qOut[3];

    // Covariance scale update with conservative local deformation to keep continuity.
    const sFactor = conservativeScaleFactor(p, rigBind, rigPose, skinning, i);
    const sx = canonical.scales[i * 3];
    const sy = canonical.scales[i * 3 + 1];
    const sz = canonical.scales[i * 3 + 2];

    const faceKeep = smoothstep(0.64, 0.92, p[1]);
    const faceFactor = 1 - faceKeep * 0.85;
    const finalFactor = 1 + (sFactor - 1) * faceFactor;

    outScale[i * 3] = sx * finalFactor;
    outScale[i * 3 + 1] = sy * finalFactor;
    outScale[i * 3 + 2] = sz * finalFactor;
  }

  return { positions: outPos, quaternions: outQuat, scales: outScale };
}

export function buildAnimationReadyAvatar(
  sourceFile: string,
  canonical: CanonicalAvatar,
  rigPose: SkeletonRig,
  skinning: SkinningData,
): AnimationReadyAvatar {
  return {
    formatVersion: "gaussian-avatar-rig.v2",
    sourceFile,
    canonical: {
      center: canonical.center,
      scaleFactor: canonical.scaleFactor,
      basis: canonical.basis,
      normalizedHeight: canonical.height,
    },
    skeleton: {
      bones: rigPose.bones.map((b) => ({
        id: b.id,
        name: b.name,
        parentId: b.parentId,
        localBindPosition: b.localBindPosition,
        localBindRotation: b.localBindRotation,
        worldBindPosition: b.worldBindPosition,
        worldBindRotation: b.worldBindRotation,
        localPosePosition: b.localPosePosition,
        localPoseRotation: b.localPoseRotation,
        worldPosePosition: b.worldPosePosition,
        worldPoseRotation: b.worldPoseRotation,
      })),
    },
    skinning: {
      influencesPerSplat: skinning.influencesPerSplat,
      boneIds: Array.from(skinning.boneIds),
      weights: Array.from(skinning.weights),
    },
  };
}
