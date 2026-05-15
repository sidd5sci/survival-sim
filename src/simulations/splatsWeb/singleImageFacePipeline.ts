/*
  Single-image neural 3D face reconstruction pipeline for gaussian avatar generation.
  Scope: face only, no animation, no body.

  This file is intentionally self-contained and uses provider interfaces so you can plug in:
  - MediaPipe FaceMesh
  - DECA / EMOCA
  while preserving a stable output format for gaussian avatar generation.
*/

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];

export interface FaceCropConfig {
  outputSize: number;
  cropScale: number;
  minFaceConfidence: number;
}

export interface HeadPose {
  yawDeg: number;
  pitchDeg: number;
  rollDeg: number;
}

export interface LandmarkPoint {
  id: number;
  uv: Vec2;
  confidence: number;
}

export interface FaceDetectionResult {
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
  };
  alignedImage: ImageData;
  alignedTransform: number[];
}

export interface LandmarkEstimationResult {
  landmarks: LandmarkPoint[];
  headPose: HeadPose;
  depthValues: Float32Array;
}

export interface FaceMesh3D {
  vertices: Float32Array;
  indices: Uint32Array;
  normals: Float32Array;
  uvs: Float32Array;
}

export interface GaussianSplatFace {
  position: Vec3;
  rotation: Quat;
  scale: Vec3;
  opacity: number;
  color: Vec3;
}

export interface FaceGaussianAvatar {
  sourceResolution: [number, number];
  cropResolution: [number, number];
  headPose: HeadPose;
  mesh: FaceMesh3D;
  splats: GaussianSplatFace[];
  visualization: {
    supersplatPayload: {
      points: number[][];
      quats: number[][];
      scales: number[][];
      colors: number[][];
      opacity: number[];
    };
    webglBuffers: {
      positions: Float32Array;
      quaternions: Float32Array;
      scales: Float32Array;
      colors: Float32Array;
      opacities: Float32Array;
    };
  };
}

export interface FaceLandmarkProvider {
  name: "mediapipe" | "deca" | "emoca" | "custom";
  estimate(croppedFace: ImageData): Promise<LandmarkEstimationResult>;
}

export interface FaceDetectorProvider {
  detect(input: ImageData): Promise<FaceDetectionResult["bbox"]>;
}

export interface PipelineOptions {
  crop?: Partial<FaceCropConfig>;
  detector?: FaceDetectorProvider;
  landmarkProvider: FaceLandmarkProvider;
  targetSplatCount?: number;
  baseOpacity?: number;
}

const DEFAULT_CROP: FaceCropConfig = {
  outputSize: 512,
  cropScale: 1.35,
  minFaceConfidence: 0.35,
};

const EPS = 1e-8;

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

function normalize(v: Vec3): Vec3 {
  const n = Math.max(EPS, Math.hypot(v[0], v[1], v[2]));
  return [v[0] / n, v[1] / n, v[2] / n];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function mul(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function quatFromBasis(tangent: Vec3, bitangent: Vec3, normal: Vec3): Quat {
  const m00 = tangent[0];
  const m01 = bitangent[0];
  const m02 = normal[0];
  const m10 = tangent[1];
  const m11 = bitangent[1];
  const m12 = normal[1];
  const m20 = tangent[2];
  const m21 = bitangent[2];
  const m22 = normal[2];

  const trace = m00 + m11 + m22;
  let x = 0;
  let y = 0;
  let z = 0;
  let w = 1;

  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    w = 0.25 * s;
    x = (m21 - m12) / s;
    y = (m02 - m20) / s;
    z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }

  const n = Math.max(EPS, Math.hypot(x, y, z, w));
  return [x / n, y / n, z / n, w / n];
}

function sampleBilinear(input: ImageData, x: number, y: number): Vec3 {
  const { width, height, data } = input;
  const x0 = clamp(Math.floor(x), 0, width - 1);
  const y0 = clamp(Math.floor(y), 0, height - 1);
  const x1 = clamp(x0 + 1, 0, width - 1);
  const y1 = clamp(y0 + 1, 0, height - 1);

  const tx = x - x0;
  const ty = y - y0;

  const p00 = (y0 * width + x0) * 4;
  const p10 = (y0 * width + x1) * 4;
  const p01 = (y1 * width + x0) * 4;
  const p11 = (y1 * width + x1) * 4;

  const c = [0, 0, 0];
  for (let i = 0; i < 3; i += 1) {
    const v00 = data[p00 + i];
    const v10 = data[p10 + i];
    const v01 = data[p01 + i];
    const v11 = data[p11 + i];
    const top = v00 * (1 - tx) + v10 * tx;
    const bot = v01 * (1 - tx) + v11 * tx;
    c[i] = (top * (1 - ty) + bot * ty) / 255;
  }

  return [c[0], c[1], c[2]];
}

export class BrowserFaceDetectorProvider implements FaceDetectorProvider {
  async detect(input: ImageData): Promise<FaceDetectionResult["bbox"]> {
    const AnyWindow = globalThis as unknown as {
      FaceDetector?: new (cfg: { fastMode: boolean; maxDetectedFaces: number }) => {
        detect(img: ImageData): Promise<Array<{ boundingBox: { x: number; y: number; width: number; height: number } }>>;
      };
    };

    if (AnyWindow.FaceDetector) {
      const fd = new AnyWindow.FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
      const faces = await fd.detect(input);
      if (faces.length > 0) {
        const b = faces[0].boundingBox;
        return { x: b.x, y: b.y, width: b.width, height: b.height, confidence: 0.9 };
      }
    }

    // Conservative fallback: centered face prior.
    return {
      x: input.width * 0.24,
      y: input.height * 0.16,
      width: input.width * 0.52,
      height: input.height * 0.62,
      confidence: 0.4,
    };
  }
}

export class PlaceholderFaceLandmarkProvider implements FaceLandmarkProvider {
  name: "custom" = "custom";

  async estimate(croppedFace: ImageData): Promise<LandmarkEstimationResult> {
    const w = croppedFace.width;
    const h = croppedFace.height;

    // Minimal neutral template landmarks for pipeline continuity.
    const points: LandmarkPoint[] = [
      { id: 0, uv: [0.5, 0.28], confidence: 0.9 }, // forehead
      { id: 1, uv: [0.34, 0.42], confidence: 0.9 }, // left eye
      { id: 2, uv: [0.66, 0.42], confidence: 0.9 }, // right eye
      { id: 3, uv: [0.5, 0.53], confidence: 0.9 }, // nose bridge
      { id: 4, uv: [0.5, 0.64], confidence: 0.9 }, // nose tip
      { id: 5, uv: [0.38, 0.76], confidence: 0.85 }, // mouth left
      { id: 6, uv: [0.62, 0.76], confidence: 0.85 }, // mouth right
      { id: 7, uv: [0.5, 0.9], confidence: 0.8 }, // chin
    ];

    const depthValues = new Float32Array(points.length);
    for (let i = 0; i < points.length; i += 1) {
      const [u, v] = points[i].uv;
      const dy = Math.abs(v - 0.55);
      const dx = Math.abs(u - 0.5);
      // Keep neutral identity-friendly shallow depth profile.
      depthValues[i] = -0.08 + 0.11 * (1 - dy) - 0.05 * dx;
    }

    const eyeVecX = points[2].uv[0] - points[1].uv[0];
    const eyeVecY = points[2].uv[1] - points[1].uv[1];

    const pose: HeadPose = {
      yawDeg: 0,
      pitchDeg: ((points[4].uv[1] - points[0].uv[1]) - 0.25) * 80,
      rollDeg: (Math.atan2(eyeVecY * h, eyeVecX * w) * 180) / Math.PI,
    };

    return { landmarks: points, headPose: pose, depthValues };
  }
}

export class SingleImageFaceGaussianPipeline {
  private readonly options: Required<Pick<PipelineOptions, "targetSplatCount" | "baseOpacity">> & {
    crop: FaceCropConfig;
    detector: FaceDetectorProvider;
    landmarkProvider: FaceLandmarkProvider;
  };

  constructor(options: PipelineOptions) {
    this.options = {
      crop: { ...DEFAULT_CROP, ...(options.crop ?? {}) },
      detector: options.detector ?? new BrowserFaceDetectorProvider(),
      landmarkProvider: options.landmarkProvider,
      targetSplatCount: options.targetSplatCount ?? 4000,
      baseOpacity: options.baseOpacity ?? 0.85,
    };
  }

  async run(image: ImageData): Promise<FaceGaussianAvatar> {
    // STEP 1: Face detection, alignment, and consistent crop.
    const det = await this.detectAlignAndCrop(image);

    // STEP 2: Landmarks, head pose, and depth estimation (provider-driven).
    const lmk = await this.options.landmarkProvider.estimate(det.alignedImage);

    // STEP 3: 3D face reconstruction (neutral mesh + normals + UV + depth structure).
    const mesh = this.reconstructMesh(det.alignedImage, lmk);

    // STEP 4: Gaussian conversion (anisotropic, covariance-aware, surface aligned).
    const splats = this.convertMeshToGaussians(det.alignedImage, mesh, this.options.targetSplatCount, this.options.baseOpacity);

    // STEP 5: Visualization payload (SuperSplat + WebGL buffers).
    return {
      sourceResolution: [image.width, image.height],
      cropResolution: [det.alignedImage.width, det.alignedImage.height],
      headPose: lmk.headPose,
      mesh,
      splats,
      visualization: this.buildVisualizationPayload(splats),
    };
  }

  private async detectAlignAndCrop(input: ImageData): Promise<FaceDetectionResult> {
    const bbox = await this.options.detector.detect(input);
    if (bbox.confidence < this.options.crop.minFaceConfidence) {
      throw new Error("Face confidence too low for stable reconstruction.");
    }

    const side = Math.max(bbox.width, bbox.height) * this.options.crop.cropScale;
    const cx = bbox.x + bbox.width * 0.5;
    const cy = bbox.y + bbox.height * 0.5;

    const outSize = this.options.crop.outputSize;
    const aligned = new ImageData(outSize, outSize);

    for (let oy = 0; oy < outSize; oy += 1) {
      for (let ox = 0; ox < outSize; ox += 1) {
        const sx = cx - side * 0.5 + (ox / (outSize - 1)) * side;
        const sy = cy - side * 0.5 + (oy / (outSize - 1)) * side;
        const [r, g, b] = sampleBilinear(input, sx, sy);
        const dst = (oy * outSize + ox) * 4;
        aligned.data[dst] = Math.round(r * 255);
        aligned.data[dst + 1] = Math.round(g * 255);
        aligned.data[dst + 2] = Math.round(b * 255);
        aligned.data[dst + 3] = 255;
      }
    }

    // 3x3 affine-like descriptor mapping crop uv -> source xy.
    const alignedTransform = [
      side / outSize,
      0,
      cx - side * 0.5,
      0,
      side / outSize,
      cy - side * 0.5,
      0,
      0,
      1,
    ];

    return { bbox, alignedImage: aligned, alignedTransform };
  }

  private reconstructMesh(croppedFace: ImageData, lmk: LandmarkEstimationResult): FaceMesh3D {
    const n = lmk.landmarks.length;
    const vertices = new Float32Array(n * 3);
    const uvs = new Float32Array(n * 2);

    for (let i = 0; i < n; i += 1) {
      const [u, v] = lmk.landmarks[i].uv;
      const z = lmk.depthValues[i] ?? 0;
      vertices[i * 3] = (u - 0.5) * 2;
      vertices[i * 3 + 1] = (0.5 - v) * 2;
      vertices[i * 3 + 2] = z;
      uvs[i * 2] = clamp(u, 0, 1);
      uvs[i * 2 + 1] = clamp(v, 0, 1);
    }

    // Lightweight face triangulation fan around nose bridge landmark index 3.
    const center = Math.min(3, n - 1);
    const ring = [...Array(n).keys()].filter((id) => id !== center);
    ring.sort((a, b) => {
      const ax = uvs[a * 2] - uvs[center * 2];
      const ay = uvs[a * 2 + 1] - uvs[center * 2 + 1];
      const bx = uvs[b * 2] - uvs[center * 2];
      const by = uvs[b * 2 + 1] - uvs[center * 2 + 1];
      return Math.atan2(ay, ax) - Math.atan2(by, bx);
    });

    const triCount = Math.max(0, ring.length - 1);
    const indices = new Uint32Array(triCount * 3);
    for (let i = 0; i < triCount; i += 1) {
      indices[i * 3] = center;
      indices[i * 3 + 1] = ring[i];
      indices[i * 3 + 2] = ring[i + 1];
    }

    const normals = new Float32Array(vertices.length);
    for (let t = 0; t < indices.length; t += 3) {
      const i0 = indices[t];
      const i1 = indices[t + 1];
      const i2 = indices[t + 2];
      const p0: Vec3 = [vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]];
      const p1: Vec3 = [vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]];
      const p2: Vec3 = [vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]];
      const nrm = normalize(cross(sub(p1, p0), sub(p2, p0)));

      for (const idx of [i0, i1, i2]) {
        normals[idx * 3] += nrm[0];
        normals[idx * 3 + 1] += nrm[1];
        normals[idx * 3 + 2] += nrm[2];
      }
    }

    for (let i = 0; i < n; i += 1) {
      const ni = normalize([normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]]);
      normals[i * 3] = ni[0];
      normals[i * 3 + 1] = ni[1];
      normals[i * 3 + 2] = ni[2];
    }

    return { vertices, indices, normals, uvs };
  }

  private convertMeshToGaussians(
    image: ImageData,
    mesh: FaceMesh3D,
    targetSplatCount: number,
    baseOpacity: number,
  ): GaussianSplatFace[] {
    const splats: GaussianSplatFace[] = [];
    if (mesh.indices.length < 3) return splats;

    const triCount = mesh.indices.length / 3;
    const splatsPerTri = Math.max(1, Math.round(targetSplatCount / triCount));

    for (let t = 0; t < mesh.indices.length; t += 3) {
      const i0 = mesh.indices[t];
      const i1 = mesh.indices[t + 1];
      const i2 = mesh.indices[t + 2];

      const p0: Vec3 = [mesh.vertices[i0 * 3], mesh.vertices[i0 * 3 + 1], mesh.vertices[i0 * 3 + 2]];
      const p1: Vec3 = [mesh.vertices[i1 * 3], mesh.vertices[i1 * 3 + 1], mesh.vertices[i1 * 3 + 2]];
      const p2: Vec3 = [mesh.vertices[i2 * 3], mesh.vertices[i2 * 3 + 1], mesh.vertices[i2 * 3 + 2]];

      const uv0: Vec2 = [mesh.uvs[i0 * 2], mesh.uvs[i0 * 2 + 1]];
      const uv1: Vec2 = [mesh.uvs[i1 * 2], mesh.uvs[i1 * 2 + 1]];
      const uv2: Vec2 = [mesh.uvs[i2 * 2], mesh.uvs[i2 * 2 + 1]];

      const e1 = sub(p1, p0);
      const e2 = sub(p2, p0);
      const normal = normalize(cross(e1, e2));
      const tangent = normalize(e1);
      const bitangent = normalize(cross(normal, tangent));
      const triArea = Math.max(EPS, 0.5 * Math.hypot(...cross(e1, e2)));

      for (let k = 0; k < splatsPerTri; k += 1) {
        const r1 = Math.random();
        const r2 = Math.random();
        const a = 1 - Math.sqrt(r1);
        const b = Math.sqrt(r1) * (1 - r2);
        const c = Math.sqrt(r1) * r2;

        const pos = add(add(mul(p0, a), mul(p1, b)), mul(p2, c));
        const uv: Vec2 = [
          uv0[0] * a + uv1[0] * b + uv2[0] * c,
          uv0[1] * a + uv1[1] * b + uv2[1] * c,
        ];

        const color = sampleBilinear(image, uv[0] * (image.width - 1), uv[1] * (image.height - 1));

        // Covariance-aware anisotropy: larger on tangent plane, thinner on normal.
        const tangentScale = Math.sqrt(triArea) * 0.18;
        const bitangentScale = Math.sqrt(triArea) * 0.13;
        const normalScale = Math.sqrt(triArea) * 0.03;

        splats.push({
          position: pos,
          rotation: quatFromBasis(tangent, bitangent, normal),
          scale: [tangentScale, bitangentScale, normalScale],
          opacity: clamp(baseOpacity, 0.05, 1),
          color,
        });
      }
    }

    return splats;
  }

  private buildVisualizationPayload(splats: GaussianSplatFace[]): FaceGaussianAvatar["visualization"] {
    const points: number[][] = [];
    const quats: number[][] = [];
    const scales: number[][] = [];
    const colors: number[][] = [];
    const opacity: number[] = [];

    const positions = new Float32Array(splats.length * 3);
    const quaternions = new Float32Array(splats.length * 4);
    const scaleBuffer = new Float32Array(splats.length * 3);
    const colorBuffer = new Float32Array(splats.length * 3);
    const opacities = new Float32Array(splats.length);

    for (let i = 0; i < splats.length; i += 1) {
      const s = splats[i];
      points.push([s.position[0], s.position[1], s.position[2]]);
      quats.push([s.rotation[0], s.rotation[1], s.rotation[2], s.rotation[3]]);
      scales.push([s.scale[0], s.scale[1], s.scale[2]]);
      colors.push([s.color[0], s.color[1], s.color[2]]);
      opacity.push(s.opacity);

      positions[i * 3] = s.position[0];
      positions[i * 3 + 1] = s.position[1];
      positions[i * 3 + 2] = s.position[2];
      quaternions[i * 4] = s.rotation[0];
      quaternions[i * 4 + 1] = s.rotation[1];
      quaternions[i * 4 + 2] = s.rotation[2];
      quaternions[i * 4 + 3] = s.rotation[3];
      scaleBuffer[i * 3] = s.scale[0];
      scaleBuffer[i * 3 + 1] = s.scale[1];
      scaleBuffer[i * 3 + 2] = s.scale[2];
      colorBuffer[i * 3] = s.color[0];
      colorBuffer[i * 3 + 1] = s.color[1];
      colorBuffer[i * 3 + 2] = s.color[2];
      opacities[i] = s.opacity;
    }

    return {
      supersplatPayload: { points, quats, scales, colors, opacity },
      webglBuffers: {
        positions,
        quaternions,
        scales: scaleBuffer,
        colors: colorBuffer,
        opacities,
      },
    };
  }
}
