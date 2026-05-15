type PlyScalarType =
  | "char"
  | "uchar"
  | "short"
  | "ushort"
  | "int"
  | "uint"
  | "float"
  | "double";

type TypedArrayLike = Float32Array | Uint32Array;

interface PlyProperty {
  name: string;
  type: PlyScalarType;
}

interface ParsedHeader {
  vertexCount: number;
  properties: PlyProperty[];
  headerByteLength: number;
}

export interface GaussianSplatData {
  count: number;
  positions: Float32Array;
  scalesRaw: Float32Array;
  scalesWorld: Float32Array;
  quaternions: Float32Array;
  opacitiesRaw: Float32Array;
  opacities: Float32Array;
  shDc: Float32Array;
  shRest: Float32Array;
  covarianceDiag: Float32Array;
  covarianceTrace: Float32Array;
  covarianceAnisotropy: Float32Array;
  propertyNames: string[];
}

export interface Histogram {
  min: number;
  max: number;
  bins: number[];
}

export interface Heatmap2D {
  width: number;
  height: number;
  values: number[];
}

export interface Segmentation {
  labels: Uint8Array;
  counts: {
    face: number;
    head: number;
    body: number;
  };
}

export interface ClusterResult {
  labels: Uint32Array;
  centroids: number[][];
  counts: number[];
}

export interface BoneDefinition {
  id: number;
  name: string;
  parentId: number;
  bindPosition: [number, number, number];
}

export interface BindingResult {
  boneIds: Uint16Array;
  weights: Float32Array;
  bones: BoneDefinition[];
  bindPose: {
    rootPosition: [number, number, number];
    bboxMin: [number, number, number];
    bboxMax: [number, number, number];
  };
}

export interface AvatarMetadata {
  formatVersion: string;
  sourceFile: string;
  vertexCount: number;
  attributes: {
    positions: boolean;
    quaternion: boolean;
    scale: boolean;
    opacity: boolean;
    sphericalHarmonics: boolean;
    covarianceDerived: boolean;
  };
  statistics: {
    splatCount: number;
    scaleHistogram: Histogram;
    opacityHistogram: Histogram;
    covarianceTraceHistogram: Histogram;
  };
  segmentation: {
    faceCount: number;
    headCount: number;
    bodyCount: number;
  };
  clustering: {
    k: number;
    centroids: number[][];
    counts: number[];
  };
  skeleton: {
    bones: BoneDefinition[];
    bindPose: {
      rootPosition: [number, number, number];
      bboxMin: [number, number, number];
      bboxMax: [number, number, number];
    };
  };
  skinning: {
    influencesPerSplat: number;
    boneIds: number[];
    weights: number[];
  };
}

const TYPE_SIZE: Record<PlyScalarType, number> = {
  char: 1,
  uchar: 1,
  short: 2,
  ushort: 2,
  int: 4,
  uint: 4,
  float: 4,
  double: 8,
};

function clamp01(v: number): number {
  if (v <= 0) return 0;
  if (v >= 1) return 1;
  return v;
}

function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

function normalizeQuat(qx: number, qy: number, qz: number, qw: number): [number, number, number, number] {
  const n = Math.hypot(qx, qy, qz, qw);
  if (n <= 1e-8) return [0, 0, 0, 1];
  return [qx / n, qy / n, qz / n, qw / n];
}

function quatToMat3(qx: number, qy: number, qz: number, qw: number): number[] {
  const xx = qx * qx;
  const yy = qy * qy;
  const zz = qz * qz;
  const xy = qx * qy;
  const xz = qx * qz;
  const yz = qy * qz;
  const wx = qw * qx;
  const wy = qw * qy;
  const wz = qw * qz;

  return [
    1 - 2 * (yy + zz),
    2 * (xy - wz),
    2 * (xz + wy),
    2 * (xy + wz),
    1 - 2 * (xx + zz),
    2 * (yz - wx),
    2 * (xz - wy),
    2 * (yz + wx),
    1 - 2 * (xx + yy),
  ];
}

function parsePlyHeader(buffer: ArrayBuffer): ParsedHeader {
  const bytes = new Uint8Array(buffer);
  const decoder = new TextDecoder("utf-8");
  const maxScan = Math.min(bytes.length, 256 * 1024);
  const headerChunk = decoder.decode(bytes.slice(0, maxScan));
  const endToken = "end_header\n";
  let headerEnd = headerChunk.indexOf(endToken);
  let tokenLen = endToken.length;

  if (headerEnd < 0) {
    const alt = "end_header\r\n";
    headerEnd = headerChunk.indexOf(alt);
    tokenLen = alt.length;
  }

  if (headerEnd < 0) {
    throw new Error("Invalid PLY: missing end_header marker.");
  }

  const headerText = headerChunk.slice(0, headerEnd);
  const lines = headerText.split(/\r?\n/);

  if (!lines[0]?.startsWith("ply")) {
    throw new Error("Invalid PLY: missing magic header.");
  }

  const formatLine = lines.find((line) => line.startsWith("format "));
  if (!formatLine || !formatLine.includes("binary_little_endian")) {
    throw new Error("Only binary_little_endian PLY files are supported.");
  }

  let vertexCount = 0;
  let inVertexElement = false;
  const properties: PlyProperty[] = [];

  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts.length === 0) continue;

    if (parts[0] === "element") {
      inVertexElement = parts[1] === "vertex";
      if (inVertexElement) {
        vertexCount = Number(parts[2] || 0);
      }
      continue;
    }

    if (inVertexElement && parts[0] === "property") {
      if (parts[1] === "list") {
        throw new Error("List properties are not supported for vertex attributes.");
      }
      const type = parts[1] as PlyScalarType;
      if (!(type in TYPE_SIZE)) {
        throw new Error(`Unsupported property type: ${parts[1]}`);
      }
      properties.push({ name: parts[2], type });
    }
  }

  if (vertexCount <= 0 || properties.length === 0) {
    throw new Error("Invalid PLY: vertex element not found or empty.");
  }

  const headerByteLength = headerEnd + tokenLen;
  return { vertexCount, properties, headerByteLength };
}

function readValue(view: DataView, offset: number, type: PlyScalarType): number {
  switch (type) {
    case "char":
      return view.getInt8(offset);
    case "uchar":
      return view.getUint8(offset);
    case "short":
      return view.getInt16(offset, true);
    case "ushort":
      return view.getUint16(offset, true);
    case "int":
      return view.getInt32(offset, true);
    case "uint":
      return view.getUint32(offset, true);
    case "float":
      return view.getFloat32(offset, true);
    case "double":
      return view.getFloat64(offset, true);
    default:
      return 0;
  }
}

function computeStride(properties: PlyProperty[]): number {
  return properties.reduce((acc, p) => acc + TYPE_SIZE[p.type], 0);
}

function buildPropertyOffsets(properties: PlyProperty[]): Map<string, { offset: number; type: PlyScalarType }> {
  const map = new Map<string, { offset: number; type: PlyScalarType }>();
  let offset = 0;
  for (const p of properties) {
    map.set(p.name, { offset, type: p.type });
    offset += TYPE_SIZE[p.type];
  }
  return map;
}

function readProperty(
  view: DataView,
  base: number,
  propertyMap: Map<string, { offset: number; type: PlyScalarType }>,
  name: string,
  fallback = 0,
): number {
  const prop = propertyMap.get(name);
  if (!prop) return fallback;
  return readValue(view, base + prop.offset, prop.type);
}

function boundingBox(positions: Float32Array): { min: [number, number, number]; max: [number, number, number] } {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];

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

function quantile(values: Float32Array, q: number): number {
  const arr = Array.from(values);
  arr.sort((a, b) => a - b);
  if (arr.length === 0) return 0;
  const idx = Math.max(0, Math.min(arr.length - 1, Math.floor(q * (arr.length - 1))));
  return arr[idx];
}

export function parseGaussianPly(buffer: ArrayBuffer): GaussianSplatData {
  const header = parsePlyHeader(buffer);
  const view = new DataView(buffer, header.headerByteLength);
  const propertyMap = buildPropertyOffsets(header.properties);
  const stride = computeStride(header.properties);

  const count = header.vertexCount;
  const positions = new Float32Array(count * 3);
  const scalesRaw = new Float32Array(count * 3);
  const scalesWorld = new Float32Array(count * 3);
  const quaternions = new Float32Array(count * 4);
  const opacitiesRaw = new Float32Array(count);
  const opacities = new Float32Array(count);
  const shDc = new Float32Array(count * 3);

  const shRestNames = header.properties
    .map((p) => p.name)
    .filter((n) => n.startsWith("f_rest_"))
    .sort((a, b) => Number(a.split("_").pop()) - Number(b.split("_").pop()));

  const shRest = new Float32Array(count * shRestNames.length);
  const covarianceDiag = new Float32Array(count * 3);
  const covarianceTrace = new Float32Array(count);
  const covarianceAnisotropy = new Float32Array(count);

  for (let i = 0; i < count; i += 1) {
    const base = i * stride;

    const px = readProperty(view, base, propertyMap, "x", 0);
    const py = readProperty(view, base, propertyMap, "y", 0);
    const pz = readProperty(view, base, propertyMap, "z", 0);

    positions[i * 3] = px;
    positions[i * 3 + 1] = py;
    positions[i * 3 + 2] = pz;

    const s0Raw = readProperty(view, base, propertyMap, "scale_0", 0);
    const s1Raw = readProperty(view, base, propertyMap, "scale_1", 0);
    const s2Raw = readProperty(view, base, propertyMap, "scale_2", 0);

    scalesRaw[i * 3] = s0Raw;
    scalesRaw[i * 3 + 1] = s1Raw;
    scalesRaw[i * 3 + 2] = s2Raw;

    const s0 = Math.exp(s0Raw);
    const s1 = Math.exp(s1Raw);
    const s2 = Math.exp(s2Raw);
    scalesWorld[i * 3] = s0;
    scalesWorld[i * 3 + 1] = s1;
    scalesWorld[i * 3 + 2] = s2;

    const q0 = readProperty(view, base, propertyMap, "rot_0", 0);
    const q1 = readProperty(view, base, propertyMap, "rot_1", 0);
    const q2 = readProperty(view, base, propertyMap, "rot_2", 0);
    const q3 = readProperty(view, base, propertyMap, "rot_3", 1);
    const [qx, qy, qz, qw] = normalizeQuat(q0, q1, q2, q3);
    quaternions[i * 4] = qx;
    quaternions[i * 4 + 1] = qy;
    quaternions[i * 4 + 2] = qz;
    quaternions[i * 4 + 3] = qw;

    const opacityRaw = readProperty(view, base, propertyMap, "opacity", 0);
    opacitiesRaw[i] = opacityRaw;
    opacities[i] = clamp01(sigmoid(opacityRaw));

    shDc[i * 3] = readProperty(view, base, propertyMap, "f_dc_0", 0);
    shDc[i * 3 + 1] = readProperty(view, base, propertyMap, "f_dc_1", 0);
    shDc[i * 3 + 2] = readProperty(view, base, propertyMap, "f_dc_2", 0);

    for (let j = 0; j < shRestNames.length; j += 1) {
      shRest[i * shRestNames.length + j] = readProperty(view, base, propertyMap, shRestNames[j], 0);
    }

    const r = quatToMat3(qx, qy, qz, qw);
    const s2x = s0 * s0;
    const s2y = s1 * s1;
    const s2z = s2 * s2;

    const c00 = r[0] * r[0] * s2x + r[1] * r[1] * s2y + r[2] * r[2] * s2z;
    const c11 = r[3] * r[3] * s2x + r[4] * r[4] * s2y + r[5] * r[5] * s2z;
    const c22 = r[6] * r[6] * s2x + r[7] * r[7] * s2y + r[8] * r[8] * s2z;

    covarianceDiag[i * 3] = c00;
    covarianceDiag[i * 3 + 1] = c11;
    covarianceDiag[i * 3 + 2] = c22;

    const trace = c00 + c11 + c22;
    covarianceTrace[i] = trace;

    const maxS = Math.max(s0, s1, s2);
    const minS = Math.max(1e-8, Math.min(s0, s1, s2));
    covarianceAnisotropy[i] = maxS / minS;
  }

  return {
    count,
    positions,
    scalesRaw,
    scalesWorld,
    quaternions,
    opacitiesRaw,
    opacities,
    shDc,
    shRest,
    covarianceDiag,
    covarianceTrace,
    covarianceAnisotropy,
    propertyNames: header.properties.map((p) => p.name),
  };
}

export function histogram(values: TypedArrayLike, bins = 40): Histogram {
  let min = Infinity;
  let max = -Infinity;

  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1, bins: Array(bins).fill(0) };
  }

  const out = Array(bins).fill(0);
  const span = Math.max(1e-8, max - min);

  for (let i = 0; i < values.length; i += 1) {
    const t = (values[i] - min) / span;
    const idx = Math.max(0, Math.min(bins - 1, Math.floor(t * bins)));
    out[idx] += 1;
  }

  return { min, max, bins: out };
}

export function segmentRegions(data: GaussianSplatData): Segmentation {
  const { positions } = data;
  const count = data.count;
  const labels = new Uint8Array(count);

  const ys = new Float32Array(count);
  const zs = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    ys[i] = positions[i * 3 + 1];
    zs[i] = positions[i * 3 + 2];
  }

  const headY = quantile(ys, 0.82);
  const faceY = quantile(ys, 0.86);
  const faceZ = quantile(zs, 0.62);

  let face = 0;
  let head = 0;
  let body = 0;

  for (let i = 0; i < count; i += 1) {
    const y = ys[i];
    const z = zs[i];

    if (y >= faceY && z >= faceZ) {
      labels[i] = 1;
      face += 1;
      continue;
    }

    if (y >= headY) {
      labels[i] = 2;
      head += 1;
      continue;
    }

    labels[i] = 3;
    body += 1;
  }

  return {
    labels,
    counts: { face, head, body },
  };
}

function makeHeatmap(
  positions: Float32Array,
  indices: number[],
  width: number,
  height: number,
  axisA: 0 | 1 | 2,
  axisB: 0 | 1 | 2,
): Heatmap2D {
  const values = Array(width * height).fill(0);

  if (indices.length === 0) {
    return { width, height, values };
  }

  let minA = Infinity;
  let maxA = -Infinity;
  let minB = Infinity;
  let maxB = -Infinity;

  for (const i of indices) {
    const a = positions[i * 3 + axisA];
    const b = positions[i * 3 + axisB];
    if (a < minA) minA = a;
    if (a > maxA) maxA = a;
    if (b < minB) minB = b;
    if (b > maxB) maxB = b;
  }

  const spanA = Math.max(1e-8, maxA - minA);
  const spanB = Math.max(1e-8, maxB - minB);

  for (const i of indices) {
    const a = positions[i * 3 + axisA];
    const b = positions[i * 3 + axisB];
    const x = Math.max(0, Math.min(width - 1, Math.floor(((a - minA) / spanA) * width)));
    const y = Math.max(0, Math.min(height - 1, Math.floor(((b - minB) / spanB) * height)));
    values[y * width + x] += 1;
  }

  return { width, height, values };
}

export function faceAndBodyHeatmaps(data: GaussianSplatData, segmentation: Segmentation): {
  face: Heatmap2D;
  body: Heatmap2D;
} {
  const faceIndices: number[] = [];
  const bodyIndices: number[] = [];

  for (let i = 0; i < segmentation.labels.length; i += 1) {
    const label = segmentation.labels[i];
    if (label === 1) faceIndices.push(i);
    if (label === 3) bodyIndices.push(i);
  }

  return {
    face: makeHeatmap(data.positions, faceIndices, 64, 64, 0, 1),
    body: makeHeatmap(data.positions, bodyIndices, 64, 64, 0, 1),
  };
}

function dist3(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  const dx = ax - bx;
  const dy = ay - by;
  const dz = az - bz;
  return Math.hypot(dx, dy, dz);
}

export function kMeansClusters(data: GaussianSplatData, k = 8, iterations = 10): ClusterResult {
  const count = data.count;
  const labels = new Uint32Array(count);
  const centroids: number[][] = [];

  const step = Math.max(1, Math.floor(count / k));
  for (let i = 0; i < k; i += 1) {
    const idx = Math.min(count - 1, i * step);
    centroids.push([
      data.positions[idx * 3],
      data.positions[idx * 3 + 1],
      data.positions[idx * 3 + 2],
    ]);
  }

  for (let iter = 0; iter < iterations; iter += 1) {
    const sums = Array.from({ length: k }, () => [0, 0, 0]);
    const counts = Array(k).fill(0);

    for (let i = 0; i < count; i += 1) {
      const px = data.positions[i * 3];
      const py = data.positions[i * 3 + 1];
      const pz = data.positions[i * 3 + 2];

      let best = 0;
      let bestDist = Infinity;
      for (let c = 0; c < k; c += 1) {
        const d = dist3(px, py, pz, centroids[c][0], centroids[c][1], centroids[c][2]);
        if (d < bestDist) {
          bestDist = d;
          best = c;
        }
      }

      labels[i] = best;
      sums[best][0] += px;
      sums[best][1] += py;
      sums[best][2] += pz;
      counts[best] += 1;
    }

    for (let c = 0; c < k; c += 1) {
      if (counts[c] <= 0) continue;
      centroids[c][0] = sums[c][0] / counts[c];
      centroids[c][1] = sums[c][1] / counts[c];
      centroids[c][2] = sums[c][2] / counts[c];
    }
  }

  const outCounts = Array(k).fill(0);
  for (let i = 0; i < count; i += 1) {
    outCounts[labels[i]] += 1;
  }

  return { labels, centroids, counts: outCounts };
}

function canonicalBones(min: [number, number, number], max: [number, number, number]): BoneDefinition[] {
  const cx = (min[0] + max[0]) * 0.5;
  const sy = max[1] - min[1];
  const sx = max[0] - min[0];
  const zFront = max[2] - (max[2] - min[2]) * 0.25;
  const zMid = (min[2] + max[2]) * 0.5;

  const y = (t: number) => min[1] + sy * t;
  const x = (t: number) => cx + sx * t;

  return [
    { id: 0, name: "root", parentId: -1, bindPosition: [cx, y(0.52), zMid] },
    { id: 1, name: "pelvis", parentId: 0, bindPosition: [cx, y(0.48), zMid] },
    { id: 2, name: "spineLower", parentId: 1, bindPosition: [cx, y(0.58), zMid] },
    { id: 3, name: "spineUpper", parentId: 2, bindPosition: [cx, y(0.70), zMid] },
    { id: 4, name: "neck", parentId: 3, bindPosition: [cx, y(0.82), zMid] },
    { id: 5, name: "head", parentId: 4, bindPosition: [cx, y(0.90), zFront] },

    { id: 6, name: "lShoulder", parentId: 3, bindPosition: [x(-0.12), y(0.76), zMid] },
    { id: 7, name: "lUpperArm", parentId: 6, bindPosition: [x(-0.20), y(0.72), zMid] },
    { id: 8, name: "lForeArm", parentId: 7, bindPosition: [x(-0.28), y(0.66), zMid] },
    { id: 9, name: "lHand", parentId: 8, bindPosition: [x(-0.34), y(0.62), zMid] },

    { id: 10, name: "rShoulder", parentId: 3, bindPosition: [x(0.12), y(0.76), zMid] },
    { id: 11, name: "rUpperArm", parentId: 10, bindPosition: [x(0.20), y(0.72), zMid] },
    { id: 12, name: "rForeArm", parentId: 11, bindPosition: [x(0.28), y(0.66), zMid] },
    { id: 13, name: "rHand", parentId: 12, bindPosition: [x(0.34), y(0.62), zMid] },

    { id: 14, name: "lUpperLeg", parentId: 1, bindPosition: [x(-0.08), y(0.40), zMid] },
    { id: 15, name: "lLowerLeg", parentId: 14, bindPosition: [x(-0.08), y(0.22), zMid] },
    { id: 16, name: "lFoot", parentId: 15, bindPosition: [x(-0.08), y(0.06), zFront] },

    { id: 17, name: "rUpperLeg", parentId: 1, bindPosition: [x(0.08), y(0.40), zMid] },
    { id: 18, name: "rLowerLeg", parentId: 17, bindPosition: [x(0.08), y(0.22), zMid] },
    { id: 19, name: "rFoot", parentId: 18, bindPosition: [x(0.08), y(0.06), zFront] },
  ];
}

export function buildBinding(data: GaussianSplatData, influencesPerSplat = 4): BindingResult {
  const box = boundingBox(data.positions);
  const bones = canonicalBones(box.min, box.max);

  const count = data.count;
  const boneIds = new Uint16Array(count * influencesPerSplat);
  const weights = new Float32Array(count * influencesPerSplat);

  const sigma = Math.max(1e-6, (box.max[1] - box.min[1]) * 0.12);

  for (let i = 0; i < count; i += 1) {
    const px = data.positions[i * 3];
    const py = data.positions[i * 3 + 1];
    const pz = data.positions[i * 3 + 2];

    const scored = bones
      .map((b) => {
        const d = dist3(px, py, pz, b.bindPosition[0], b.bindPosition[1], b.bindPosition[2]);
        const w = Math.exp(-(d * d) / (2 * sigma * sigma));
        return { id: b.id, w };
      })
      .sort((a, b) => b.w - a.w)
      .slice(0, influencesPerSplat);

    let norm = 0;
    for (const s of scored) norm += s.w;
    norm = Math.max(1e-8, norm);

    for (let j = 0; j < influencesPerSplat; j += 1) {
      const entry = scored[j] ?? scored[0];
      boneIds[i * influencesPerSplat + j] = entry.id;
      weights[i * influencesPerSplat + j] = entry.w / norm;
    }
  }

  return {
    boneIds,
    weights,
    bones,
    bindPose: {
      rootPosition: bones[0].bindPosition,
      bboxMin: box.min,
      bboxMax: box.max,
    },
  };
}

export function buildAvatarMetadata(
  sourceFile: string,
  data: GaussianSplatData,
  segmentation: Segmentation,
  clusters: ClusterResult,
  binding: BindingResult,
): AvatarMetadata {
  const scaleMagnitude = new Float32Array(data.count);
  for (let i = 0; i < data.count; i += 1) {
    const sx = data.scalesWorld[i * 3];
    const sy = data.scalesWorld[i * 3 + 1];
    const sz = data.scalesWorld[i * 3 + 2];
    scaleMagnitude[i] = (sx + sy + sz) / 3;
  }

  return {
    formatVersion: "splat-avatar.v1",
    sourceFile,
    vertexCount: data.count,
    attributes: {
      positions: true,
      quaternion: data.propertyNames.includes("rot_0") && data.propertyNames.includes("rot_3"),
      scale: data.propertyNames.includes("scale_0") && data.propertyNames.includes("scale_2"),
      opacity: data.propertyNames.includes("opacity"),
      sphericalHarmonics: data.propertyNames.some((p) => p.startsWith("f_dc_") || p.startsWith("f_rest_")),
      covarianceDerived: true,
    },
    statistics: {
      splatCount: data.count,
      scaleHistogram: histogram(scaleMagnitude, 40),
      opacityHistogram: histogram(data.opacities, 40),
      covarianceTraceHistogram: histogram(data.covarianceTrace, 40),
    },
    segmentation: {
      faceCount: segmentation.counts.face,
      headCount: segmentation.counts.head,
      bodyCount: segmentation.counts.body,
    },
    clustering: {
      k: clusters.centroids.length,
      centroids: clusters.centroids,
      counts: clusters.counts,
    },
    skeleton: {
      bones: binding.bones,
      bindPose: binding.bindPose,
    },
    skinning: {
      influencesPerSplat: 4,
      boneIds: Array.from(binding.boneIds),
      weights: Array.from(binding.weights),
    },
  };
}

export function orientationVectors(data: GaussianSplatData, sampleCount = 2000): Array<{
  x: number;
  y: number;
  z: number;
  dx: number;
  dy: number;
  dz: number;
}> {
  const out: Array<{ x: number; y: number; z: number; dx: number; dy: number; dz: number }> = [];
  const step = Math.max(1, Math.floor(data.count / sampleCount));

  for (let i = 0; i < data.count; i += step) {
    const x = data.positions[i * 3];
    const y = data.positions[i * 3 + 1];
    const z = data.positions[i * 3 + 2];

    const qx = data.quaternions[i * 4];
    const qy = data.quaternions[i * 4 + 1];
    const qz = data.quaternions[i * 4 + 2];
    const qw = data.quaternions[i * 4 + 3];
    const r = quatToMat3(qx, qy, qz, qw);

    out.push({
      x,
      y,
      z,
      dx: r[2],
      dy: r[5],
      dz: r[8],
    });
  }

  return out;
}
