import { useEffect, useMemo, useRef, useState } from "react";
import "./splatsWeb.css";
import { css as supersplatCss, html as supersplatHtml, js as supersplatJs } from "@playcanvas/supersplat-viewer";
import LoadPlySection from "./LoadPlySection";
import PlySplatViewerSection from "./PlySplatViewerSection";
import {
  faceAndBodyHeatmaps,
  histogram,
  kMeansClusters,
  parseGaussianPly,
  segmentRegions,
  type ClusterResult,
  type GaussianSplatData,
  type Heatmap2D,
  type Segmentation,
} from "./plyInspector";
import {
  applyTestPose,
  assignSkinningWeights,
  buildWeightDebugData,
  buildAnimationReadyAvatar,
  deformSplats,
  fitHumanoidSkeleton,
  prepareCanonicalAvatar,
  type AnimationReadyAvatar,
  type CanonicalAvatar,
  type DeformedSplats,
  type SkeletonBone,
  type SkeletonRig,
  type SkinningData,
  type TestControls,
  type WeightDebugData,
} from "@/simulations/splatsWeb/avatarPreparation";

type InspectorState = {
  sourceName: string;
  data: GaussianSplatData;
  segmentation: Segmentation;
  clusters: ClusterResult;
  canonical: CanonicalAvatar;
  bindRig: SkeletonRig;
  skinning: SkinningData;
  rigSampleCount: number;
  sourceCount: number;
  faceHeatmap: Heatmap2D;
  bodyHeatmap: Heatmap2D;
  rigSourceIndices: Uint32Array;
};

type ViewerSample = {
  positions: Float32Array;
  colors: Float32Array;
  sizes: Float32Array;
  alphas: Float32Array;
  quaternions: Float32Array;
  sourceIndices: Uint32Array;
  count: number;
  center: [number, number, number];
  viewScale: number;
};

type NormalViewerRigData = {
  rig: SkeletonRig;
  dominantBoneAt: (sourceIndex: number) => number;
  hasBoneInfluence: (boneId: number, sourceIndex: number) => boolean;
  weightForBone: (boneId: number, sourceIndex: number) => number;
  blendScoreAt: (sourceIndex: number) => number;
};

type InfluenceOverlayMode = "overlay" | "heatmap" | "wireframe" | "blend";
type ViewerDebugMode =
  | "normalAvatar"
  | "skeletonOverlay"
  | "boneInfluenceOverlay"
  | "weightHeatmap"
  | "covarianceOrientation"
  | "skinningDebug"
  | "bindPose"
  | "poseTest"
  | "transformDebug";

type UploadedPly = {
  sourceName: string;
  arrayBuffer: ArrayBuffer;
};

type ViewerBundle = {
  frameHtml: string;
  cleanupUrls: string[];
};

type ManualBoneEdit = {
  tx: number;
  ty: number;
  tz: number;
  rxDeg: number;
  ryDeg: number;
  rzDeg: number;
};

const EMPTY_MANUAL_EDIT: ManualBoneEdit = {
  tx: 0,
  ty: 0,
  tz: 0,
  rxDeg: 0,
  ryDeg: 0,
  rzDeg: 0,
};

const MAX_INTERACTIVE_SPLATS = 28000;
const MAX_CLUSTER_SPLATS = 22000;
const MAX_VIEWER_SPLATS = 36000;

function buildDefaultViewerSettingsJson() {
  return {
    version: 2,
    tonemapping: "none",
    highPrecisionRendering: false,
    background: {
      color: [0, 0, 0],
    },
    postEffectSettings: {
      sharpness: { enabled: false, amount: 0 },
      bloom: { enabled: false, intensity: 1, blurLevel: 2 },
      grading: { enabled: false, brightness: 0, contrast: 1, saturation: 1, tint: [1, 1, 1] },
      vignette: { enabled: false, intensity: 0.5, inner: 0.3, outer: 0.75, curvature: 1 },
      fringing: { enabled: false, intensity: 0.5 },
    },
    animTracks: [],
    cameras: [
      {
        initial: {
          position: [0, 1, -1],
          target: [0, 0, 0],
          fov: 60,
        },
      },
    ],
    annotations: [],
    startMode: "default",
  };
}

function buildSuperSplatViewerBundle(arrayBuffer: ArrayBuffer): ViewerBundle {
  const settingsBlobUrl = URL.createObjectURL(
    new Blob([JSON.stringify(buildDefaultViewerSettingsJson())], { type: "application/json" }),
  );
  const contentBlobUrl = URL.createObjectURL(new Blob([arrayBuffer], { type: "application/octet-stream" }));
  const cssBlobUrl = URL.createObjectURL(new Blob([supersplatCss], { type: "text/css" }));
  const jsBlobUrl = URL.createObjectURL(new Blob([supersplatJs], { type: "text/javascript" }));

  const htmlDoc = supersplatHtml
    .replace('./index.css', cssBlobUrl)
    .replace("'./index.js'", `'${jsBlobUrl}'`)
    .replace(
      "const settingsUrl = url.searchParams.has('settings') ? url.searchParams.get('settings') : './settings.json';",
      `const settingsUrl = '${settingsBlobUrl}';`,
    )
    .replace(
      "const contentUrl = url.searchParams.has('content') ? url.searchParams.get('content') : './scene.compressed.ply';",
      `const contentUrl = '${contentBlobUrl}';`,
    );

  return {
    frameHtml: htmlDoc,
    cleanupUrls: [settingsBlobUrl, contentBlobUrl, cssBlobUrl, jsBlobUrl],
  };
}

function revokeViewerBundle(bundle: ViewerBundle | null) {
  if (!bundle) return;
  for (const url of bundle.cleanupUrls) {
    URL.revokeObjectURL(url);
  }
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

function normalizeQuat(x: number, y: number, z: number, w: number): [number, number, number, number] {
  const n = Math.max(1e-8, Math.hypot(x, y, z, w));
  return [x / n, y / n, z / n, w / n];
}

function quatMultiply(
  ax: number,
  ay: number,
  az: number,
  aw: number,
  bx: number,
  by: number,
  bz: number,
  bw: number,
): [number, number, number, number] {
  return normalizeQuat(
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  );
}

function quatFromAxisAngleDeg(ax: number, ay: number, az: number, deg: number): [number, number, number, number] {
  const n = Math.max(1e-8, Math.hypot(ax, ay, az));
  const ux = ax / n;
  const uy = ay / n;
  const uz = az / n;
  const rad = (deg * Math.PI) / 180;
  const s = Math.sin(rad * 0.5);
  const c = Math.cos(rad * 0.5);
  return normalizeQuat(ux * s, uy * s, uz * s, c);
}

function quatFromEulerDeg(rxDeg: number, ryDeg: number, rzDeg: number): [number, number, number, number] {
  const qx = quatFromAxisAngleDeg(1, 0, 0, rxDeg);
  const qy = quatFromAxisAngleDeg(0, 1, 0, ryDeg);
  const qz = quatFromAxisAngleDeg(0, 0, 1, rzDeg);
  const qxy = quatMultiply(qx[0], qx[1], qx[2], qx[3], qy[0], qy[1], qy[2], qy[3]);
  return quatMultiply(qxy[0], qxy[1], qxy[2], qxy[3], qz[0], qz[1], qz[2], qz[3]);
}

function applyManualBoneEdits(rig: SkeletonRig, edits: Record<number, ManualBoneEdit>): SkeletonRig {
  const next: SkeletonRig = {
    bones: rig.bones.map((b) => ({ ...b })),
  };

  for (const bone of next.bones) {
    const edit = edits[bone.id] ?? EMPTY_MANUAL_EDIT;
    bone.localPosePosition = [
      bone.localBindPosition[0] + edit.tx,
      bone.localBindPosition[1] + edit.ty,
      bone.localBindPosition[2] + edit.tz,
    ];
    const qEdit = quatFromEulerDeg(edit.rxDeg, edit.ryDeg, edit.rzDeg);
    const qLocal = quatMultiply(
      qEdit[0],
      qEdit[1],
      qEdit[2],
      qEdit[3],
      bone.localBindRotation[0],
      bone.localBindRotation[1],
      bone.localBindRotation[2],
      bone.localBindRotation[3],
    );
    bone.localPoseRotation = qLocal;
  }

  const worldPos: Array<[number, number, number] | null> = new Array(next.bones.length).fill(null);
  const worldRot: Array<[number, number, number, number] | null> = new Array(next.bones.length).fill(null);

  const solve = (id: number) => {
    if (id < 0 || id >= next.bones.length) return;
    if (worldPos[id] && worldRot[id]) return;

    const b = next.bones[id];
    if (b.parentId < 0 || b.parentId >= next.bones.length) {
      worldPos[id] = [b.localPosePosition[0], b.localPosePosition[1], b.localPosePosition[2]];
      worldRot[id] = [b.localPoseRotation[0], b.localPoseRotation[1], b.localPoseRotation[2], b.localPoseRotation[3]];
    } else {
      solve(b.parentId);
      const pPos = worldPos[b.parentId] ?? [0, 0, 0];
      const pRot = worldRot[b.parentId] ?? [0, 0, 0, 1];
      const offset = rotateByQuatVec(
        b.localPosePosition[0],
        b.localPosePosition[1],
        b.localPosePosition[2],
        pRot[0],
        pRot[1],
        pRot[2],
        pRot[3],
      );
      worldPos[id] = [pPos[0] + offset.x, pPos[1] + offset.y, pPos[2] + offset.z];
      worldRot[id] = quatMultiply(
        pRot[0],
        pRot[1],
        pRot[2],
        pRot[3],
        b.localPoseRotation[0],
        b.localPoseRotation[1],
        b.localPoseRotation[2],
        b.localPoseRotation[3],
      );
    }

    const wp = worldPos[id] ?? [0, 0, 0];
    const wr = worldRot[id] ?? [0, 0, 0, 1];
    b.worldPosePosition = [wp[0], wp[1], wp[2]];
    b.worldPoseRotation = [wr[0], wr[1], wr[2], wr[3]];
  };

  for (let i = 0; i < next.bones.length; i += 1) solve(i);
  return next;
}

function basisQuat(canonical: CanonicalAvatar): [number, number, number, number] {
  const m00 = canonical.basis.right[0];
  const m10 = canonical.basis.right[1];
  const m20 = canonical.basis.right[2];
  const m01 = canonical.basis.up[0];
  const m11 = canonical.basis.up[1];
  const m21 = canonical.basis.up[2];
  const m02 = canonical.basis.forward[0];
  const m12 = canonical.basis.forward[1];
  const m22 = canonical.basis.forward[2];

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

  return normalizeQuat(x, y, z, w);
}

function canonicalToWorldPosition(canonical: CanonicalAvatar, x: number, y: number, z: number) {
  const sx = canonical.scaleFactors?.[0] ?? canonical.scaleFactor;
  const sy = canonical.scaleFactors?.[1] ?? canonical.scaleFactor;
  const sz = canonical.scaleFactors?.[2] ?? canonical.scaleFactor;
  const ux = x / Math.max(1e-8, sx);
  const uy = y / Math.max(1e-8, sy);
  const uz = z / Math.max(1e-8, sz);
  const wq = canonical.worldFromCanonicalQuat ?? basisQuat(canonical);
  const rotated = rotateByQuatVec(ux, uy, uz, wq[0], wq[1], wq[2], wq[3]);
  return {
    x: canonical.center[0] + rotated.x,
    y: canonical.center[1] + rotated.y,
    z: canonical.center[2] + rotated.z,
  };
}

function samplePositionsForClustering(data: GaussianSplatData): GaussianSplatData {
  if (data.count <= MAX_CLUSTER_SPLATS) return data;

  const step = Math.ceil(data.count / MAX_CLUSTER_SPLATS);
  const sampleCount = Math.ceil(data.count / step);
  const sampledPositions = new Float32Array(sampleCount * 3);

  let w = 0;
  for (let i = 0; i < data.count; i += step) {
    sampledPositions[w * 3] = data.positions[i * 3];
    sampledPositions[w * 3 + 1] = data.positions[i * 3 + 1];
    sampledPositions[w * 3 + 2] = data.positions[i * 3 + 2];
    w += 1;
  }

  return {
    ...data,
    count: sampleCount,
    positions: sampledPositions,
  };
}

function downsampleForRigging(canonical: CanonicalAvatar, segmentation: Segmentation): {
  canonical: CanonicalAvatar;
  segmentation: Segmentation;
  sampleCount: number;
  sourceIndices: Uint32Array;
} {
  const sourceCount = segmentation.labels.length;
  if (sourceCount <= MAX_INTERACTIVE_SPLATS) {
    const sourceIndices = new Uint32Array(sourceCount);
    for (let i = 0; i < sourceCount; i += 1) sourceIndices[i] = i;
    return { canonical, segmentation, sampleCount: sourceCount, sourceIndices };
  }

  const step = Math.ceil(sourceCount / MAX_INTERACTIVE_SPLATS);
  const sampleCount = Math.ceil(sourceCount / step);

  const positions = new Float32Array(sampleCount * 3);
  const quaternions = new Float32Array(sampleCount * 4);
  const scales = new Float32Array(sampleCount * 3);
  const labels = new Uint8Array(sampleCount);
  const sourceIndices = new Uint32Array(sampleCount);

  let w = 0;
  for (let i = 0; i < sourceCount; i += step) {
    positions[w * 3] = canonical.positions[i * 3];
    positions[w * 3 + 1] = canonical.positions[i * 3 + 1];
    positions[w * 3 + 2] = canonical.positions[i * 3 + 2];

    quaternions[w * 4] = canonical.quaternions[i * 4];
    quaternions[w * 4 + 1] = canonical.quaternions[i * 4 + 1];
    quaternions[w * 4 + 2] = canonical.quaternions[i * 4 + 2];
    quaternions[w * 4 + 3] = canonical.quaternions[i * 4 + 3];

    scales[w * 3] = canonical.scales[i * 3];
    scales[w * 3 + 1] = canonical.scales[i * 3 + 1];
    scales[w * 3 + 2] = canonical.scales[i * 3 + 2];

    labels[w] = segmentation.labels[i];
    sourceIndices[w] = i;
    w += 1;
  }

  return {
    canonical: {
      ...canonical,
      positions,
      quaternions,
      scales,
    },
    segmentation: {
      ...segmentation,
      labels,
      counts: {
        face: labels.reduce((acc, v) => acc + (v === 1 ? 1 : 0), 0),
        head: labels.reduce((acc, v) => acc + (v === 2 ? 1 : 0), 0),
        body: labels.reduce((acc, v) => acc + (v === 3 ? 1 : 0), 0),
      },
    },
    sampleCount,
    sourceIndices,
  };
}

function colorFromT(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, x * 1.6));
  const g = Math.round(255 * Math.min(1, 1.2 - Math.abs(x - 0.45) * 1.8));
  const b = Math.round(255 * Math.min(1, (1 - x) * 1.5));
  return [r, g, b];
}

function buildViewerSample(data: GaussianSplatData): ViewerSample {
  const sourceCount = data.count;
  if (sourceCount === 0) {
    return {
      positions: new Float32Array(0),
      colors: new Float32Array(0),
      sizes: new Float32Array(0),
      alphas: new Float32Array(0),
      quaternions: new Float32Array(0),
      sourceIndices: new Uint32Array(0),
      count: 0,
      center: [0, 0, 0],
      viewScale: 1,
    };
  }

  const step = Math.max(1, Math.ceil(sourceCount / MAX_VIEWER_SPLATS));
  const count = Math.ceil(sourceCount / step);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const alphas = new Float32Array(count);
  const quaternions = new Float32Array(count * 4);
  const sourceIndices = new Uint32Array(count);

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  let w = 0;
  for (let i = 0; i < sourceCount; i += step) {
    const x = data.positions[i * 3];
    const y = data.positions[i * 3 + 1];
    const z = data.positions[i * 3 + 2];

    positions[w * 3] = x;
    positions[w * 3 + 1] = y;
    positions[w * 3 + 2] = z;
    sourceIndices[w] = i;

    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;

    const r = clamp(0.5 + data.shDc[i * 3] * 0.48, 0, 1);
    const g = clamp(0.5 + data.shDc[i * 3 + 1] * 0.48, 0, 1);
    const b = clamp(0.5 + data.shDc[i * 3 + 2] * 0.48, 0, 1);
    colors[w * 3] = r;
    colors[w * 3 + 1] = g;
    colors[w * 3 + 2] = b;

    const sx = data.scalesWorld[i * 3];
    const sy = data.scalesWorld[i * 3 + 1];
    const sz = data.scalesWorld[i * 3 + 2];
    sizes[w] = (sx + sy + sz) / 3;
    alphas[w] = clamp(data.opacities[i], 0.08, 1);

    quaternions[w * 4] = data.quaternions[i * 4];
    quaternions[w * 4 + 1] = data.quaternions[i * 4 + 1];
    quaternions[w * 4 + 2] = data.quaternions[i * 4 + 2];
    quaternions[w * 4 + 3] = data.quaternions[i * 4 + 3];
    w += 1;
  }

  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  const extentX = maxX - minX;
  const extentY = maxY - minY;
  const extentZ = maxZ - minZ;
  const viewScale = 1 / Math.max(1e-6, Math.max(extentX, extentY, extentZ));

  return { positions, colors, sizes, alphas, quaternions, sourceIndices, count, center: [cx, cy, cz], viewScale };
}

function buildViewerSampleFromCanonical(
  canonical: CanonicalAvatar,
  sourceData: GaussianSplatData,
  sourceIndices: Uint32Array,
): ViewerSample {
  const sourceCount = Math.floor(canonical.positions.length / 3);
  if (sourceCount === 0) {
    return {
      positions: new Float32Array(0),
      colors: new Float32Array(0),
      sizes: new Float32Array(0),
      alphas: new Float32Array(0),
      quaternions: new Float32Array(0),
      sourceIndices: new Uint32Array(0),
      count: 0,
      center: [0, 0, 0],
      viewScale: 1,
    };
  }

  const positions = new Float32Array(sourceCount * 3);
  const colors = new Float32Array(sourceCount * 3);
  const sizes = new Float32Array(sourceCount);
  const alphas = new Float32Array(sourceCount);
  const quaternions = new Float32Array(sourceCount * 4);
  const sampledSourceIndices = new Uint32Array(sourceCount);
  const worldBasisQuat = canonical.worldFromCanonicalQuat ?? basisQuat(canonical);

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  for (let i = 0; i < sourceCount; i += 1) {
    const cx = canonical.positions[i * 3];
    const cy = canonical.positions[i * 3 + 1];
    const cz = canonical.positions[i * 3 + 2];
    const worldPos = canonicalToWorldPosition(canonical, cx, cy, cz);

    const sourceIndex = sourceIndices[i] ?? i;
    const clampedSourceIndex = Math.max(0, Math.min(sourceData.count - 1, sourceIndex));

    positions[i * 3] = worldPos.x;
    positions[i * 3 + 1] = worldPos.y;
    positions[i * 3 + 2] = worldPos.z;
    sampledSourceIndices[i] = clampedSourceIndex;

    if (worldPos.x < minX) minX = worldPos.x;
    if (worldPos.y < minY) minY = worldPos.y;
    if (worldPos.z < minZ) minZ = worldPos.z;
    if (worldPos.x > maxX) maxX = worldPos.x;
    if (worldPos.y > maxY) maxY = worldPos.y;
    if (worldPos.z > maxZ) maxZ = worldPos.z;

    colors[i * 3] = clamp(0.5 + sourceData.shDc[clampedSourceIndex * 3] * 0.48, 0, 1);
    colors[i * 3 + 1] = clamp(0.5 + sourceData.shDc[clampedSourceIndex * 3 + 1] * 0.48, 0, 1);
    colors[i * 3 + 2] = clamp(0.5 + sourceData.shDc[clampedSourceIndex * 3 + 2] * 0.48, 0, 1);

    const invScaleX = 1 / Math.max(1e-8, canonical.scaleFactors?.[0] ?? canonical.scaleFactor);
    const invScaleY = 1 / Math.max(1e-8, canonical.scaleFactors?.[1] ?? canonical.scaleFactor);
    const invScaleZ = 1 / Math.max(1e-8, canonical.scaleFactors?.[2] ?? canonical.scaleFactor);
    const sx = canonical.scales[i * 3] * invScaleX;
    const sy = canonical.scales[i * 3 + 1] * invScaleY;
    const sz = canonical.scales[i * 3 + 2] * invScaleZ;
    sizes[i] = (sx + sy + sz) / 3;
    alphas[i] = clamp(sourceData.opacities[clampedSourceIndex], 0.08, 1);

    const cqx = canonical.quaternions[i * 4];
    const cqy = canonical.quaternions[i * 4 + 1];
    const cqz = canonical.quaternions[i * 4 + 2];
    const cqw = canonical.quaternions[i * 4 + 3];
    const [wqx, wqy, wqz, wqw] = quatMultiply(
      worldBasisQuat[0],
      worldBasisQuat[1],
      worldBasisQuat[2],
      worldBasisQuat[3],
      cqx,
      cqy,
      cqz,
      cqw,
    );
    quaternions[i * 4] = wqx;
    quaternions[i * 4 + 1] = wqy;
    quaternions[i * 4 + 2] = wqz;
    quaternions[i * 4 + 3] = wqw;
  }

  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  const extentX = maxX - minX;
  const extentY = maxY - minY;
  const extentZ = maxZ - minZ;
  const viewScale = 1 / Math.max(1e-6, Math.max(extentX, extentY, extentZ));

  return {
    positions,
    colors,
    sizes,
    alphas,
    quaternions,
    sourceIndices: sampledSourceIndices,
    count: sourceCount,
    center: [cx, cy, cz],
    viewScale,
  };
}

function buildNormalViewerRigData(state: InspectorState, overlayRig: SkeletonRig): NormalViewerRigData {
  const splatCount = Math.floor(state.canonical.positions.length / 3);
  const influences = state.skinning.influencesPerSplat;
  const dominantBone = new Uint16Array(splatCount);
  const blendScore = new Float32Array(splatCount);

  for (let i = 0; i < splatCount; i += 1) {
    let bestW = -1;
    let bestB = 0;
    for (let j = 0; j < influences; j += 1) {
      const idx = i * influences + j;
      const w = state.skinning.weights[idx] ?? 0;
      const b = state.skinning.boneIds[idx] ?? 0;
      if (w > bestW) {
        bestW = w;
        bestB = b;
      }
    }
    dominantBone[i] = bestB;

    let top1 = 0;
    let top2 = 0;
    for (let j = 0; j < influences; j += 1) {
      const idx = i * influences + j;
      const w = state.skinning.weights[idx] ?? 0;
      if (w > top1) {
        top2 = top1;
        top1 = w;
      } else if (w > top2) {
        top2 = w;
      }
    }
    blendScore[i] = clamp(top2 / Math.max(1e-6, top1), 0, 1);
  }

  const sourceCount = state.data.count;
  const rawToRigIndex = new Int32Array(sourceCount);
  const rigToRaw = state.rigSourceIndices;
  if (splatCount === 0 || rigToRaw.length === 0 || sourceCount === 0) {
    rawToRigIndex.fill(-1);
  } else {
    let s = 0;
    for (let raw = 0; raw < sourceCount; raw += 1) {
      while (
        s + 1 < rigToRaw.length &&
        Math.abs((rigToRaw[s + 1] ?? sourceCount) - raw) <= Math.abs((rigToRaw[s] ?? 0) - raw)
      ) {
        s += 1;
      }
      rawToRigIndex[raw] = s;
    }
  }

  const resolveRigIndex = (sourceIndex: number) => {
    if (sourceCount <= 0) return -1;
    const clampedRaw = Math.max(0, Math.min(sourceCount - 1, sourceIndex));
    return rawToRigIndex[clampedRaw] ?? -1;
  };

  return {
    rig: overlayRig,
    dominantBoneAt: (sourceIndex: number) => {
      const rigIdx = resolveRigIndex(sourceIndex);
      return rigIdx >= 0 ? dominantBone[rigIdx] ?? 0 : 0;
    },
    hasBoneInfluence: (boneId: number, sourceIndex: number) => {
      const rigIdx = resolveRigIndex(sourceIndex);
      if (rigIdx < 0) return false;
      for (let j = 0; j < influences; j += 1) {
        const idx = rigIdx * influences + j;
        if ((state.skinning.boneIds[idx] ?? -1) === boneId && (state.skinning.weights[idx] ?? 0) > 0.001) {
          return true;
        }
      }
      return false;
    },
    weightForBone: (boneId: number, sourceIndex: number) => {
      const rigIdx = resolveRigIndex(sourceIndex);
      if (rigIdx < 0) return 0;
      let weight = 0;
      for (let j = 0; j < influences; j += 1) {
        const idx = rigIdx * influences + j;
        if ((state.skinning.boneIds[idx] ?? -1) === boneId) {
          weight += state.skinning.weights[idx] ?? 0;
        }
      }
      return clamp(weight, 0, 1);
    },
    blendScoreAt: (sourceIndex: number) => {
      const rigIdx = resolveRigIndex(sourceIndex);
      return rigIdx >= 0 ? blendScore[rigIdx] ?? 0 : 0;
    },
  };
}

function boneColor(boneId: number): [number, number, number] {
  const x = Math.abs(Math.sin((boneId + 1) * 12.9898));
  const y = Math.abs(Math.sin((boneId + 1) * 78.233));
  const z = Math.abs(Math.sin((boneId + 1) * 37.719));
  return [
    Math.round(80 + x * 170),
    Math.round(70 + y * 175),
    Math.round(85 + z * 160),
  ];
}

function rotateByQuatVec(vx: number, vy: number, vz: number, qx: number, qy: number, qz: number, qw: number) {
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  return {
    x: vx + qw * tx + (qy * tz - qz * ty),
    y: vy + qw * ty + (qz * tx - qx * tz),
    z: vz + qw * tz + (qx * ty - qy * tx),
  };
}

function drawSplatViewer(
  canvas: HTMLCanvasElement,
  sample: ViewerSample,
  yawDeg: number,
  pitchDeg: number,
  panX: number,
  panY: number,
  zoom: number,
  perspective: number,
  density: number,
  selectedBoneId: number | null,
  rigData: NormalViewerRigData | null,
  rigCanonical: CanonicalAvatar | null,
  debugMode: ViewerDebugMode,
  influenceMode: InfluenceOverlayMode,
  splatOpacity: number,
  skeletonOpacity: number,
  influenceIntensity: number,
  covarianceIntensity: number,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx || sample.count === 0) return;

  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const w = Math.max(300, Math.floor(canvas.clientWidth * dpr));
  const h = Math.max(220, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  ctx.fillStyle = "#030b14";
  ctx.fillRect(0, 0, w, h);

  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const stride = Math.max(1, Math.floor((101 - density) / 7));
  const cameraDistance = perspective;
  const focal = Math.min(w, h) * 0.9 * zoom;
  const cx = w * (0.5 + panX);
  const cyScr = h * (0.52 + panY);

  const project = (x: number, y: number, z: number) => {
    const xr = cy * x + sy * z;
    const zr = -sy * x + cy * z;
    const yr = cp * y - sp * zr;
    const zr2 = sp * y + cp * zr;
    const depth = zr2 + cameraDistance;
    if (depth <= 0.08) return null;
    const invDepth = 1 / depth;
    const px = cx + xr * focal * invDepth;
    const py = cyScr + yr * focal * invDepth;
    return { px, py, invDepth };
  };

  const toViewSpace = (x: number, y: number, z: number) => {
    return {
      x: (x - sample.center[0]) * sample.viewScale,
      y: (y - sample.center[1]) * sample.viewScale,
      z: (z - sample.center[2]) * sample.viewScale,
    };
  };

  const projectRaw = (x: number, y: number, z: number) => {
    const p = toViewSpace(x, y, z);
    return project(p.x, p.y, p.z);
  };

  const overlayBones = rigData?.rig.bones ?? [];
  const usePoseHierarchy =
    debugMode === "poseTest" ||
    overlayBones.some(
      (b) =>
        Math.abs(b.localPosePosition[0] - b.localBindPosition[0]) > 1e-6 ||
        Math.abs(b.localPosePosition[1] - b.localBindPosition[1]) > 1e-6 ||
        Math.abs(b.localPosePosition[2] - b.localBindPosition[2]) > 1e-6 ||
        Math.abs(b.localPoseRotation[0] - b.localBindRotation[0]) > 1e-6 ||
        Math.abs(b.localPoseRotation[1] - b.localBindRotation[1]) > 1e-6 ||
        Math.abs(b.localPoseRotation[2] - b.localBindRotation[2]) > 1e-6 ||
        Math.abs(b.localPoseRotation[3] - b.localBindRotation[3]) > 1e-6,
    );

  const canonicalWorldPos: Array<{ x: number; y: number; z: number }> = new Array(overlayBones.length);
  const canonicalWorldRot: Array<[number, number, number, number]> = new Array(overlayBones.length);
  const visited = new Uint8Array(overlayBones.length);

  const resolveCanonicalWorld = (boneId: number) => {
    if (boneId < 0 || boneId >= overlayBones.length) {
      return {
        pos: { x: 0, y: 0, z: 0 },
        rot: [0, 0, 0, 1] as [number, number, number, number],
      };
    }
    if (visited[boneId]) {
      return {
        pos: canonicalWorldPos[boneId],
        rot: canonicalWorldRot[boneId],
      };
    }

    const bone = overlayBones[boneId];
    const lp = usePoseHierarchy ? bone.localPosePosition : bone.localBindPosition;
    const lr = usePoseHierarchy ? bone.localPoseRotation : bone.localBindRotation;

    if (bone.parentId < 0 || bone.parentId >= overlayBones.length) {
      canonicalWorldPos[boneId] = { x: lp[0], y: lp[1], z: lp[2] };
      canonicalWorldRot[boneId] = [lr[0], lr[1], lr[2], lr[3]];
      visited[boneId] = 1;
      return {
        pos: canonicalWorldPos[boneId],
        rot: canonicalWorldRot[boneId],
      };
    }

    const parent = resolveCanonicalWorld(bone.parentId);
    const offset = rotateByQuatVec(lp[0], lp[1], lp[2], parent.rot[0], parent.rot[1], parent.rot[2], parent.rot[3]);
    canonicalWorldPos[boneId] = {
      x: parent.pos.x + offset.x,
      y: parent.pos.y + offset.y,
      z: parent.pos.z + offset.z,
    };
    canonicalWorldRot[boneId] = quatMultiply(
      parent.rot[0],
      parent.rot[1],
      parent.rot[2],
      parent.rot[3],
      lr[0],
      lr[1],
      lr[2],
      lr[3],
    );
    visited[boneId] = 1;
    return {
      pos: canonicalWorldPos[boneId],
      rot: canonicalWorldRot[boneId],
    };
  };

  for (let i = 0; i < overlayBones.length; i += 1) {
    resolveCanonicalWorld(i);
  }

  const headBone = overlayBones.find((b) => b.name.toLowerCase().includes("head"));
  const pelvisBone = overlayBones.find((b) => b.name.toLowerCase().includes("pelvis"));
  const headCanon = headBone ? canonicalWorldPos[headBone.id] : null;
  const pelvisCanon = pelvisBone ? canonicalWorldPos[pelvisBone.id] : null;
  const mappedHead =
    headCanon && rigCanonical
      ? canonicalToWorldPosition(rigCanonical, headCanon.x, headCanon.y, headCanon.z)
      : headCanon;
  const mappedPelvis =
    pelvisCanon && rigCanonical
      ? canonicalToWorldPosition(rigCanonical, pelvisCanon.x, pelvisCanon.y, pelvisCanon.z)
      : pelvisCanon;
  const flipRigY = Boolean(mappedHead && mappedPelvis && mappedHead.y < mappedPelvis.y);

  const boneWorldPosition = (bone: SkeletonBone) => {
    const cp = canonicalWorldPos[bone.id] ?? { x: bone.worldBindPosition[0], y: bone.worldBindPosition[1], z: bone.worldBindPosition[2] };
    const by = flipRigY ? -cp.y : cp.y;
    if (!rigCanonical) {
      return {
        x: cp.x,
        y: by,
        z: cp.z,
      };
    }
    return canonicalToWorldPosition(
      rigCanonical,
      cp.x,
      by,
      cp.z,
    );
  };

  const hasInfluenceOverlay =
    debugMode === "boneInfluenceOverlay" || debugMode === "weightHeatmap" || debugMode === "skinningDebug";
  const drawSkeleton =
    debugMode === "skeletonOverlay" ||
    debugMode === "boneInfluenceOverlay" ||
    debugMode === "bindPose" ||
    debugMode === "poseTest" ||
    debugMode === "transformDebug";
  const drawCovariance =
    debugMode === "covarianceOrientation" || (debugMode === "normalAvatar" && covarianceIntensity > 0.01);

  const projected: Array<{
    px: number;
    py: number;
    depth: number;
    r: number;
    g: number;
    b: number;
    alpha: number;
    radius: number;
    sourceIndex: number;
  }> = [];

  for (let i = 0; i < sample.count; i += stride) {
    const x = sample.positions[i * 3];
    const y = sample.positions[i * 3 + 1];
    const z = sample.positions[i * 3 + 2];

    const p = projectRaw(x, y, z);
    if (!p) continue;
    const { px, py, invDepth } = p;

    if (px < -20 || px > w + 20 || py < -20 || py > h + 20) continue;

    const sourceIndex = sample.sourceIndices[i] ?? i;

    let r = Math.round(sample.colors[i * 3] * 255);
    let g = Math.round(sample.colors[i * 3 + 1] * 255);
    let b = Math.round(sample.colors[i * 3 + 2] * 255);
    const alpha = clamp(sample.alphas[i] * (0.2 + invDepth * 0.35) * splatOpacity, 0.02, 0.6);
    const radius = clamp(sample.sizes[i] * sample.viewScale * invDepth * 1.55, 0.35, 2.2);

    if (drawCovariance && i % Math.max(1, stride * 3) === 0) {
      const qx = sample.quaternions[i * 4];
      const qy = sample.quaternions[i * 4 + 1];
      const qz = sample.quaternions[i * 4 + 2];
      const qw = sample.quaternions[i * 4 + 3];
      const axis = rotateByQuatVec(1, 0, 0, qx, qy, qz, qw);
      const covarianceLen = 0.025 / Math.max(1e-6, sample.viewScale);
      const sx = axis.x * covarianceLen;
      const sy = axis.y * covarianceLen;
      const sz = axis.z * covarianceLen;
      const p2 = projectRaw(x + sx, y + sy, z + sz);
      if (p2) {
        const covAlpha = clamp((0.15 + invDepth * 0.2) * covarianceIntensity, 0.04, 0.75);
        ctx.strokeStyle = `rgba(145, 207, 255, ${covAlpha.toFixed(3)})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(p2.px, p2.py);
        ctx.stroke();
      }
    }

    projected.push({ px, py, depth: invDepth, r, g, b, alpha, radius, sourceIndex });
  }

  projected.sort((a, b) => a.depth - b.depth);
  for (const p of projected) {
    ctx.fillStyle = `rgba(${p.r}, ${p.g}, ${p.b}, ${p.alpha.toFixed(3)})`;
    if (p.radius <= 0.6) {
      ctx.fillRect(p.px, p.py, 1.4, 1.4);
    } else {
      ctx.beginPath();
      ctx.arc(p.px, p.py, p.radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (hasInfluenceOverlay && rigData && selectedBoneId !== null) {
    for (const p of projected) {
      const wBone = rigData.weightForBone(selectedBoneId, p.sourceIndex);
      if (wBone <= 0.0005) {
        if (influenceMode !== "wireframe") {
          const dimAlpha = clamp(0.04 * influenceIntensity, 0, 0.15);
          ctx.fillStyle = `rgba(6, 10, 15, ${dimAlpha.toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(p.px, p.py, p.radius * 0.95, 0, Math.PI * 2);
          ctx.fill();
        }
        continue;
      }

      if (influenceMode === "overlay") {
        const alpha = clamp((0.22 + wBone * 0.58) * influenceIntensity, 0.04, 0.9);
        ctx.fillStyle = `rgba(255, 165, 72, ${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p.px, p.py, p.radius * (1.05 + wBone * 0.5), 0, Math.PI * 2);
        ctx.fill();
      } else if (influenceMode === "heatmap") {
        const [hr, hg, hb] = colorFromT(wBone);
        const alpha = clamp((0.15 + wBone * 0.62) * influenceIntensity, 0.05, 0.88);
        ctx.fillStyle = `rgba(${hr}, ${hg}, ${hb}, ${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p.px, p.py, p.radius * (1 + wBone * 0.45), 0, Math.PI * 2);
        ctx.fill();
      } else if (influenceMode === "wireframe") {
        const alpha = clamp((0.14 + wBone * 0.58) * influenceIntensity, 0.06, 0.8);
        ctx.strokeStyle = `rgba(255, 188, 94, ${alpha.toFixed(3)})`;
        ctx.lineWidth = clamp(0.6 + wBone * 1.6, 0.6, 2.3);
        ctx.beginPath();
        ctx.arc(p.px, p.py, p.radius * (1.05 + wBone * 0.5), 0, Math.PI * 2);
        ctx.stroke();
      } else {
        const dominant = rigData.dominantBoneAt(p.sourceIndex);
        const blend = rigData.blendScoreAt(p.sourceIndex);
        const [br, bg, bb] = boneColor(dominant);
        const [wr, wg, wb] = colorFromT(wBone);
        const mix = clamp(blend * 0.7 + wBone * 0.3, 0, 1);
        const rr = Math.round(br * (1 - mix) + wr * mix);
        const rg = Math.round(bg * (1 - mix) + wg * mix);
        const rb = Math.round(bb * (1 - mix) + wb * mix);
        const alpha = clamp((0.12 + wBone * 0.55) * influenceIntensity, 0.05, 0.88);
        ctx.fillStyle = `rgba(${rr}, ${rg}, ${rb}, ${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p.px, p.py, p.radius * (1 + mix * 0.45), 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  if (!rigData || !drawSkeleton) return;

  const bones = rigData.rig.bones;

  if (debugMode === "transformDebug") {
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 1.5;
    for (const bone of bones) {
      if (bone.parentId < 0) continue;
      const parentCanon = canonicalWorldPos[bone.parentId];
      const childCanon = canonicalWorldPos[bone.id];
      if (parentCanon && childCanon) {
        const aCanon = project(parentCanon.x, flipRigY ? -parentCanon.y : parentCanon.y, parentCanon.z);
        const bCanon = project(childCanon.x, flipRigY ? -childCanon.y : childCanon.y, childCanon.z);
        if (aCanon && bCanon) {
          ctx.strokeStyle = "rgba(92, 255, 138, 0.85)";
          ctx.beginPath();
          ctx.moveTo(aCanon.px, aCanon.py);
          ctx.lineTo(bCanon.px, bCanon.py);
          ctx.stroke();
        }
      }

      const parentWorld = boneWorldPosition(bones[bone.parentId]);
      const childWorld = boneWorldPosition(bone);
      const aWorld = projectRaw(parentWorld.x, parentWorld.y, parentWorld.z);
      const bWorld = projectRaw(childWorld.x, childWorld.y, childWorld.z);
      if (aWorld && bWorld) {
        ctx.strokeStyle = "rgba(255, 102, 102, 0.88)";
        ctx.beginPath();
        ctx.moveTo(aWorld.px, aWorld.py);
        ctx.lineTo(bWorld.px, bWorld.py);
        ctx.stroke();
      }
    }

    for (const bone of bones) {
      const cp = canonicalWorldPos[bone.id];
      if (cp) {
        const pCanon = project(cp.x, flipRigY ? -cp.y : cp.y, cp.z);
        if (pCanon) {
          ctx.fillStyle = "rgba(92, 255, 138, 0.92)";
          ctx.beginPath();
          ctx.arc(pCanon.px, pCanon.py, 2.8, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      const wp = boneWorldPosition(bone);
      const pWorld = projectRaw(wp.x, wp.y, wp.z);
      if (pWorld) {
        ctx.fillStyle = "rgba(255, 102, 102, 0.95)";
        ctx.beginPath();
        ctx.arc(pWorld.px, pWorld.py, 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    return;
  }

  ctx.shadowBlur = 8;
  ctx.shadowColor = `rgba(120, 225, 255, ${clamp(skeletonOpacity * 0.75, 0.1, 0.8).toFixed(3)})`;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = 1.8;
  for (const bone of bones) {
    if (bone.parentId < 0) continue;
    const parent = bones[bone.parentId];
    const parentPos = boneWorldPosition(parent);
    const childPos = boneWorldPosition(bone);
    const a = projectRaw(parentPos.x, parentPos.y, parentPos.z);
    const b = projectRaw(childPos.x, childPos.y, childPos.z);
    if (!a || !b) continue;
    const boneAlpha = clamp(skeletonOpacity * (selectedBoneId === bone.id ? 1 : 0.8), 0.08, 1);
    ctx.strokeStyle = selectedBoneId === bone.id ? `rgba(255, 188, 96, ${boneAlpha.toFixed(3)})` : `rgba(152, 220, 255, ${boneAlpha.toFixed(3)})`;
    ctx.beginPath();
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
    ctx.stroke();
  }

  for (const bone of bones) {
    const bWorld = boneWorldPosition(bone);
    const bx = bWorld.x;
    const by = bWorld.y;
    const bz = bWorld.z;
    const p = projectRaw(bx, by, bz);
    if (!p) continue;
    const crow = canonicalWorldRot[bone.id] ?? [0, 0, 0, 1];
    const qx = crow[0];
    const qy = crow[1];
    const qz = crow[2];
    const qw = crow[3];
    const ax = rotateByQuatVec(1, 0, 0, qx, qy, qz, qw);
    const ay = rotateByQuatVec(0, 1, 0, qx, qy, qz, qw);
    const az = rotateByQuatVec(0, 0, 1, qx, qy, qz, qw);
    const len = 0.035 / Math.max(1e-6, sample.viewScale);

    const pX = projectRaw(bx + ax.x * len, by + ax.y * len, bz + ax.z * len);
    const pY = projectRaw(bx + ay.x * len, by + ay.y * len, bz + ay.z * len);
    const pZ = projectRaw(bx + az.x * len, by + az.y * len, bz + az.z * len);

    if (pX) {
      ctx.strokeStyle = `rgba(255, 100, 100, ${clamp(skeletonOpacity * 0.92, 0.08, 1).toFixed(3)})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(p.px, p.py);
      ctx.lineTo(pX.px, pX.py);
      ctx.stroke();
    }
    if (pY) {
      ctx.strokeStyle = `rgba(111, 248, 126, ${clamp(skeletonOpacity * 0.92, 0.08, 1).toFixed(3)})`;
      ctx.beginPath();
      ctx.moveTo(p.px, p.py);
      ctx.lineTo(pY.px, pY.py);
      ctx.stroke();
    }
    if (pZ) {
      ctx.strokeStyle = `rgba(112, 175, 255, ${clamp(skeletonOpacity * 0.92, 0.08, 1).toFixed(3)})`;
      ctx.beginPath();
      ctx.moveTo(p.px, p.py);
      ctx.lineTo(pZ.px, pZ.py);
      ctx.stroke();
    }
  }

  for (const bone of bones) {
    const bWorld = boneWorldPosition(bone);
    const p = projectRaw(bWorld.x, bWorld.y, bWorld.z);
    if (!p) continue;
    const isSelected = selectedBoneId === bone.id;
    const jointAlpha = clamp(skeletonOpacity * (isSelected ? 1 : 0.9), 0.08, 1);
    ctx.fillStyle = isSelected ? `rgba(255, 178, 88, ${jointAlpha.toFixed(3)})` : `rgba(132, 208, 255, ${jointAlpha.toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(p.px, p.py, isSelected ? 4.6 : 3.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.shadowBlur = 0;
}

function drawHeatmap(canvas: HTMLCanvasElement, heatmap: Heatmap2D, colorA: string, colorB: string) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const { width, height, values } = heatmap;
  canvas.width = width;
  canvas.height = height;

  let max = 0;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] > max) max = values[i];
  }
  max = Math.max(1, max);

  const image = ctx.createImageData(width, height);
  const rgbaA = parseCssColor(colorA);
  const rgbaB = parseCssColor(colorB);

  for (let i = 0; i < values.length; i += 1) {
    const t = Math.min(1, values[i] / max);
    const r = Math.round(rgbaA[0] * (1 - t) + rgbaB[0] * t);
    const g = Math.round(rgbaA[1] * (1 - t) + rgbaB[1] * t);
    const b = Math.round(rgbaA[2] * (1 - t) + rgbaB[2] * t);

    const base = i * 4;
    image.data[base] = r;
    image.data[base + 1] = g;
    image.data[base + 2] = b;
    image.data[base + 3] = 255;
  }

  ctx.putImageData(image, 0, 0);
}

function parseCssColor(input: string): [number, number, number] {
  if (input.startsWith("#") && input.length === 7) {
    return [
      Number.parseInt(input.slice(1, 3), 16),
      Number.parseInt(input.slice(3, 5), 16),
      Number.parseInt(input.slice(5, 7), 16),
    ];
  }
  return [20, 20, 20];
}

function drawOrientation(canvas: HTMLCanvasElement, positions: Float32Array, quaternions: Float32Array) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const vectors: Array<{ x: number; y: number; dx: number; dy: number }> = [];
  const count = Math.floor(positions.length / 3);
  const step = Math.max(1, Math.floor(count / 1600));
  for (let i = 0; i < count; i += step) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const qx = quaternions[i * 4];
    const qy = quaternions[i * 4 + 1];
    const qz = quaternions[i * 4 + 2];
    const qw = quaternions[i * 4 + 3];

    const m02 = 2 * (qx * qz + qw * qy);
    const m12 = 2 * (qy * qz - qw * qx);
    vectors.push({ x, y, dx: m02, dy: m12 });
  }

  const w = 460;
  const h = 280;
  canvas.width = w;
  canvas.height = h;

  ctx.fillStyle = "#071726";
  ctx.fillRect(0, 0, w, h);

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (const v of vectors) {
    if (v.x < minX) minX = v.x;
    if (v.x > maxX) maxX = v.x;
    if (v.y < minY) minY = v.y;
    if (v.y > maxY) maxY = v.y;
  }

  const sx = Math.max(1e-6, maxX - minX);
  const sy = Math.max(1e-6, maxY - minY);
  const scale = 10;

  ctx.strokeStyle = "rgba(131, 225, 255, 0.65)";
  ctx.lineWidth = 1;

  for (const v of vectors) {
    const px = ((v.x - minX) / sx) * (w - 10) + 5;
    const py = (1 - (v.y - minY) / sy) * (h - 10) + 5;
    const qx = px + v.dx * scale;
    const qy = py - v.dy * scale;

    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(qx, qy);
    ctx.stroke();
  }
}

function drawDeformation(canvas: HTMLCanvasElement, bindPositions: Float32Array, posedPositions: Float32Array) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const w = 460;
  const h = 280;
  canvas.width = w;
  canvas.height = h;

  ctx.fillStyle = "#06111a";
  ctx.fillRect(0, 0, w, h);

  const count = Math.floor(bindPositions.length / 3);
  const step = Math.max(1, Math.floor(count / 3500));
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (let i = 0; i < count; i += step) {
    const x = bindPositions[i * 3];
    const y = bindPositions[i * 3 + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const sx = Math.max(1e-6, maxX - minX);
  const sy = Math.max(1e-6, maxY - minY);

  ctx.fillStyle = "rgba(108, 157, 187, 0.45)";
  for (let i = 0; i < count; i += step) {
    const x = ((bindPositions[i * 3] - minX) / sx) * (w - 10) + 5;
    const y = (1 - (bindPositions[i * 3 + 1] - minY) / sy) * (h - 10) + 5;
    ctx.fillRect(x, y, 1.5, 1.5);
  }

  ctx.fillStyle = "rgba(255, 145, 77, 0.75)";
  for (let i = 0; i < count; i += step) {
    const x = ((posedPositions[i * 3] - minX) / sx) * (w - 10) + 5;
    const y = (1 - (posedPositions[i * 3 + 1] - minY) / sy) * (h - 10) + 5;
    ctx.fillRect(x, y, 1.5, 1.5);
  }
}

function drawWeightDebug(
  canvas: HTMLCanvasElement,
  positions: Float32Array,
  debug: WeightDebugData,
  mode: "bone" | "weight" | "blend",
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const w = 460;
  const h = 280;
  canvas.width = w;
  canvas.height = h;

  ctx.fillStyle = "#08111a";
  ctx.fillRect(0, 0, w, h);

  const count = Math.floor(positions.length / 3);
  const step = Math.max(1, Math.floor(count / 6000));
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (let i = 0; i < count; i += step) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const sx = Math.max(1e-6, maxX - minX);
  const sy = Math.max(1e-6, maxY - minY);

  for (let i = 0; i < count; i += step) {
    const x = ((positions[i * 3] - minX) / sx) * (w - 10) + 5;
    const y = (1 - (positions[i * 3 + 1] - minY) / sy) * (h - 10) + 5;

    let r = 220;
    let g = 220;
    let b = 220;
    if (mode === "bone") {
      const bid = debug.dominantBone[i];
      const base = bid * 3;
      if (base + 2 < debug.boneColors.length) {
        r = Math.round(debug.boneColors[base] * 255);
        g = Math.round(debug.boneColors[base + 1] * 255);
        b = Math.round(debug.boneColors[base + 2] * 255);
      }
    } else if (mode === "weight") {
      [r, g, b] = colorFromT(debug.dominantWeight[i]);
    } else {
      [r, g, b] = colorFromT(debug.blendScore[i]);
    }

    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    ctx.fillRect(x, y, 1.6, 1.6);
  }
}

function MiniHistogram({ title, bins }: { title: string; bins: number[] }) {
  const max = Math.max(1, ...bins);

  return (
    <div className="mini-hist">
      <div className="mini-hist-title">{title}</div>
      <div className="mini-hist-bars">
        {bins.map((v, i) => {
          const h = Math.max(2, (v / max) * 84);
          return <span key={`${title}-${i}`} style={{ height: `${h}px` }} />;
        })}
      </div>
    </div>
  );
}

async function inspectArrayBuffer(sourceName: string, arrayBuffer: ArrayBuffer, clusterCount: number) {
  const data = parseGaussianPly(arrayBuffer);
  const segmentation = segmentRegions(data);
  const clusteringInput = samplePositionsForClustering(data);
  const clusters = kMeansClusters(clusteringInput, clusterCount, 6);
  const heatmaps = faceAndBodyHeatmaps(data, segmentation);
  const canonicalFull = prepareCanonicalAvatar(data, segmentation);
  const sampled = downsampleForRigging(canonicalFull, segmentation);
  const bindRig = fitHumanoidSkeleton(sampled.canonical);
  const skinning = assignSkinningWeights(sampled.canonical, bindRig, sampled.segmentation);

  return {
    sourceName,
    data,
    segmentation,
    clusters,
    canonical: sampled.canonical,
    bindRig,
    skinning,
    rigSampleCount: sampled.sampleCount,
    rigSourceIndices: sampled.sourceIndices,
    sourceCount: data.count,
    faceHeatmap: heatmaps.face,
    bodyHeatmap: heatmaps.body,
  } satisfies InspectorState;
}

export default function SplatWebSim3D() {
  const [activeSection, setActiveSection] = useState<"load" | "view">("load");
  const [clusterCount, setClusterCount] = useState(8);
  const [debugMode, setDebugMode] = useState<"bone" | "weight" | "blend">("bone");
  const [controls, setControls] = useState<TestControls>({
    armRotateDeg: 0,
    spineBendDeg: 0,
    headRotateDeg: 0,
    torsoTwistDeg: 0,
  });
  const [state, setState] = useState<InspectorState | null>(null);
  const [uploadedPly, setUploadedPly] = useState<UploadedPly | null>(null);
  const [viewerData, setViewerData] = useState<GaussianSplatData | null>(null);
  const [viewerBundle, setViewerBundle] = useState<ViewerBundle | null>(null);
  const [viewerMode, setViewerMode] = useState<"none" | "normal" | "supersplat">("none");
  const [selectedBoneId, setSelectedBoneId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [poseError, setPoseError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [viewerYawDeg, setViewerYawDeg] = useState(0);
  const [viewerPitchDeg, setViewerPitchDeg] = useState(0);
  const [viewerPanX, setViewerPanX] = useState(0);
  const [viewerPanY, setViewerPanY] = useState(0);
  const [viewerZoom, setViewerZoom] = useState(1.3);
  const [viewerPerspective, setViewerPerspective] = useState(1.7);
  const [viewerDensity, setViewerDensity] = useState(78);
  const [viewerInfluenceMode, setViewerInfluenceMode] = useState<InfluenceOverlayMode>("overlay");
  const [viewerDebugMode, setViewerDebugMode] = useState<ViewerDebugMode>("normalAvatar");
  const [viewerSplatOpacity, setViewerSplatOpacity] = useState(1);
  const [viewerSkeletonOpacity, setViewerSkeletonOpacity] = useState(0.8);
  const [viewerInfluenceIntensity, setViewerInfluenceIntensity] = useState(0.75);
  const [viewerCovarianceIntensity, setViewerCovarianceIntensity] = useState(0.6);
  const [viewerPoseMode, setViewerPoseMode] = useState<"bind" | "deformed">("bind");
  const [manualBoneEdits, setManualBoneEdits] = useState<Record<number, ManualBoneEdit>>({});
  const viewerDrag = useRef<{ active: boolean; x: number; y: number; mode: "orbit" | "pan" }>({
    active: false,
    x: 0,
    y: 0,
    mode: "orbit",
  });
  const faceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bodyCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const orientationCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const deformationCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const weightDebugCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const viewerCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const posed = useMemo(() => {
    if (!state) return null;
    try {
      const rigPose = applyTestPose(state.bindRig, controls);
      const deformed = deformSplats(state.canonical, state.bindRig, rigPose, state.skinning);
      const weightDebug = buildWeightDebugData(state.skinning, state.bindRig.bones.length);
      return {
        ok: true,
        rigPose,
        deformed,
        weightDebug,
      } as const;
    } catch (e) {
      return {
        ok: false,
        message: e instanceof Error ? e.message : "Unknown pose deformation failure",
      } as const;
    }
  }, [state, controls]);

  useEffect(() => {
    if (!posed || posed.ok) {
      setPoseError(null);
      return;
    }
    setPoseError(posed.message);
  }, [posed]);

  const scaleHist = useMemo(() => {
    if (!state) return [];
    const mags = new Float32Array(state.data.count);
    for (let i = 0; i < state.data.count; i += 1) {
      const sx = state.data.scalesWorld[i * 3];
      const sy = state.data.scalesWorld[i * 3 + 1];
      const sz = state.data.scalesWorld[i * 3 + 2];
      mags[i] = (sx + sy + sz) / 3;
    }
    return histogram(mags, 36).bins;
  }, [state]);

  const opacityHist = useMemo(() => (state ? histogram(state.data.opacities, 36).bins : []), [state]);
  const covHist = useMemo(() => (state ? histogram(state.data.covarianceTrace, 36).bins : []), [state]);
  const viewerSource = viewerData ?? state?.data ?? null;
  const viewerUsePoseRig = useMemo(() => {
    if (viewerDebugMode === "bindPose") return false;
    if (viewerDebugMode === "poseTest") return true;
    return viewerPoseMode === "deformed";
  }, [viewerDebugMode, viewerPoseMode]);
  const viewerOverlayRig = useMemo(() => {
    if (!state) return null;
    const baseRig = viewerUsePoseRig && posed && posed.ok ? posed.rigPose : state.bindRig;
    return applyManualBoneEdits(baseRig, manualBoneEdits);
  }, [state, viewerUsePoseRig, posed, manualBoneEdits]);
  const viewerRigData = useMemo(
    () => (state && viewerOverlayRig ? buildNormalViewerRigData(state, viewerOverlayRig) : null),
    [state, viewerOverlayRig],
  );
  const viewerSample = useMemo(() => {
    if (viewerSource) return buildViewerSample(viewerSource);
    return null;
  }, [viewerSource]);

  const viewerDebugStats = useMemo(() => {
    if (!viewerRigData || selectedBoneId === null || !viewerSample) {
      return {
        selectedBoneName: selectedBoneId === null ? "None" : `Bone ${selectedBoneId}`,
        affectedSplatCount: 0,
        averageWeight: 0,
      };
    }

    const name = viewerRigData.rig.bones.find((b) => b.id === selectedBoneId)?.name ?? `Bone ${selectedBoneId}`;
    let count = 0;
    let totalWeight = 0;
    for (let i = 0; i < viewerSample.count; i += 1) {
      const sourceIndex = viewerSample.sourceIndices[i] ?? i;
      const w = viewerRigData.weightForBone(selectedBoneId, sourceIndex);
      if (w > 0.001) {
        count += 1;
        totalWeight += w;
      }
    }
    return {
      selectedBoneName: name,
      affectedSplatCount: count,
      averageWeight: count > 0 ? totalWeight / count : 0,
    };
  }, [viewerRigData, selectedBoneId, viewerSample]);

  const selectedBoneEdit = useMemo(() => {
    if (selectedBoneId === null) return EMPTY_MANUAL_EDIT;
    return manualBoneEdits[selectedBoneId] ?? EMPTY_MANUAL_EDIT;
  }, [selectedBoneId, manualBoneEdits]);

  useEffect(() => {
    if (!state || !posed || !posed.ok || !faceCanvasRef.current || !bodyCanvasRef.current || !orientationCanvasRef.current) {
      return;
    }
    drawHeatmap(faceCanvasRef.current, state.faceHeatmap, "#08121b", "#f97316");
    drawHeatmap(bodyCanvasRef.current, state.bodyHeatmap, "#08131f", "#38bdf8");
    drawOrientation(orientationCanvasRef.current, posed.deformed.positions, posed.deformed.quaternions);
    if (deformationCanvasRef.current) {
      drawDeformation(deformationCanvasRef.current, state.canonical.positions, posed.deformed.positions);
    }
    if (weightDebugCanvasRef.current) {
      drawWeightDebug(weightDebugCanvasRef.current, state.canonical.positions, posed.weightDebug, debugMode);
    }
  }, [state, posed, debugMode]);

  useEffect(() => {
    return () => {
      revokeViewerBundle(viewerBundle);
    };
  }, [viewerBundle]);

  useEffect(() => {
    if (viewerMode !== "normal" || !viewerSample || !viewerCanvasRef.current) return;
    drawSplatViewer(
      viewerCanvasRef.current,
      viewerSample,
      viewerYawDeg,
      viewerPitchDeg,
      viewerPanX,
      viewerPanY,
      viewerZoom,
      viewerPerspective,
      viewerDensity,
      selectedBoneId,
      viewerRigData,
      state?.canonical ?? null,
      viewerDebugMode,
      viewerInfluenceMode,
      viewerSplatOpacity,
      viewerSkeletonOpacity,
      viewerInfluenceIntensity,
      viewerCovarianceIntensity,
    );
  }, [
    viewerMode,
    viewerSample,
    viewerYawDeg,
    viewerPitchDeg,
    viewerPanX,
    viewerPanY,
    viewerZoom,
    viewerPerspective,
    viewerDensity,
    selectedBoneId,
    viewerRigData,
    state,
    viewerDebugMode,
    viewerInfluenceMode,
    viewerSplatOpacity,
    viewerSkeletonOpacity,
    viewerInfluenceIntensity,
    viewerCovarianceIntensity,
  ]);

  useEffect(() => {
    if (viewerMode !== "normal" || !viewerSample || !viewerCanvasRef.current) return;
    const handle = () => {
      if (!viewerCanvasRef.current) return;
      drawSplatViewer(
        viewerCanvasRef.current,
        viewerSample,
        viewerYawDeg,
        viewerPitchDeg,
        viewerPanX,
        viewerPanY,
        viewerZoom,
        viewerPerspective,
        viewerDensity,
        selectedBoneId,
        viewerRigData,
        state?.canonical ?? null,
        viewerDebugMode,
        viewerInfluenceMode,
        viewerSplatOpacity,
        viewerSkeletonOpacity,
        viewerInfluenceIntensity,
        viewerCovarianceIntensity,
      );
    };
    window.addEventListener("resize", handle);
    return () => window.removeEventListener("resize", handle);
  }, [
    viewerMode,
    viewerSample,
    viewerYawDeg,
    viewerPitchDeg,
    viewerPanX,
    viewerPanY,
    viewerZoom,
    viewerPerspective,
    viewerDensity,
    selectedBoneId,
    viewerRigData,
    state,
    viewerDebugMode,
    viewerInfluenceMode,
    viewerSplatOpacity,
    viewerSkeletonOpacity,
    viewerInfluenceIntensity,
    viewerCovarianceIntensity,
  ]);

  const onFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const buffer = await file.arrayBuffer();
      setUploadedPly({ sourceName: file.name, arrayBuffer: buffer });
      setState(null);
      setViewerData(null);
      setViewerBundle((prev) => {
        revokeViewerBundle(prev);
        return null;
      });
      setViewerMode("none");
      setSelectedBoneId(null);
      setManualBoneEdits({});
      setPoseError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to read selected PLY file.");
    } finally {
      setIsLoading(false);
    }
  };

  const onAnalyze = async () => {
    if (!uploadedPly) return;
    setIsLoading(true);
    setError(null);
    try {
      const next = await inspectArrayBuffer(uploadedPly.sourceName, uploadedPly.arrayBuffer, clusterCount);
      setState(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to analyze uploaded PLY file.");
    } finally {
      setIsLoading(false);
    }
  };

  const onLoadNormalViewer = async () => {
    if (!uploadedPly) return;
    setIsLoading(true);
    setError(null);
    try {
      const parsed = parseGaussianPly(uploadedPly.arrayBuffer);
      setViewerData(parsed);
      setViewerMode("normal");
      setViewerPoseMode("bind");
      setViewerInfluenceMode("overlay");
      setViewerDebugMode("normalAvatar");
      setViewerSplatOpacity(1);
      setViewerSkeletonOpacity(0.8);
      setViewerInfluenceIntensity(0.75);
      setViewerCovarianceIntensity(0.6);
      setViewerPanX(0);
      setViewerPanY(0);
      setViewerZoom(1.3);
      setViewerPerspective(1.7);
      if (state?.bindRig?.bones?.length) {
        setSelectedBoneId(state.bindRig.bones[0].id);
      } else {
        setSelectedBoneId(null);
      }
      setManualBoneEdits({});
      setViewerBundle((prev) => {
        revokeViewerBundle(prev);
        return null;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to prepare normal viewer from uploaded PLY file.");
    } finally {
      setIsLoading(false);
    }
  };

  const onLoadSuperSplatViewer = async () => {
    if (!uploadedPly) return;
    setIsLoading(true);
    setError(null);
    try {
      const parsed = parseGaussianPly(uploadedPly.arrayBuffer);
      setViewerData(parsed);
      setViewerMode("supersplat");
      setViewerBundle((prev) => {
        revokeViewerBundle(prev);
        return buildSuperSplatViewerBundle(uploadedPly.arrayBuffer);
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to prepare SuperSplat viewer from uploaded PLY file.");
    } finally {
      setIsLoading(false);
    }
  };

  const onViewerPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const isPan = event.button === 2 || event.shiftKey;
    viewerDrag.current = { active: true, x: event.clientX, y: event.clientY, mode: isPan ? "pan" : "orbit" };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onViewerPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!viewerDrag.current.active) return;
    const dx = event.clientX - viewerDrag.current.x;
    const dy = event.clientY - viewerDrag.current.y;
    viewerDrag.current.x = event.clientX;
    viewerDrag.current.y = event.clientY;
    if (viewerDrag.current.mode === "orbit") {
      setViewerYawDeg((prev) => prev + dx * 0.36);
      setViewerPitchDeg((prev) => clamp(prev + dy * 0.32, -88, 88));
    } else {
      setViewerPanX((prev) => clamp(prev + dx / 900, -0.45, 0.45));
      setViewerPanY((prev) => clamp(prev + dy / 900, -0.45, 0.45));
    }
  };

  const onViewerPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    viewerDrag.current.active = false;
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // no-op
    }
  };

  const onViewerWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const delta = Math.sign(event.deltaY);
    setViewerZoom((prev) => clamp(prev - delta * 0.07, 0.45, 4));
  };

  const onExport = () => {
    if (!state || !posed || !posed.ok) return;
    const exportPayload = buildAnimationReadyAvatar(state.sourceName, state.canonical, posed.rigPose, state.skinning);
    const blob = new Blob([JSON.stringify(exportPayload)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "siyun-splat-avatar-metadata.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="splat-inspector-wrap">
      <header className="inspector-header">
        <h1>Gaussian Splat PLY Inspector + Skeleton Binder</h1>
        <p>
          Loads and analyzes Siyun splats, segments face/head/body, clusters regions, and exports animation-ready
          binding metadata without regenerating or animating splats.
        </p>
      </header>

      <section className="upload-step">
        <h2>Step 1: Upload .ply (Common for all operations)</h2>
        <p>Upload once, then either analyze and generate JSON or open full-width splat viewer below.</p>
        <label className="file-input-label">
          <span>Load .ply</span>
          <input type="file" accept=".ply" onChange={onFile} />
        </label>
        <div className="upload-meta">
          <span>Uploaded file: {uploadedPly ? uploadedPly.sourceName : "None"}</span>
        </div>
      </section>

      <section className="workflow-sections">
        <div className="section-switch">
          <button type="button" data-active={activeSection === "load"} onClick={() => setActiveSection("load")}>
            Open Load PLY Section
          </button>
          <button type="button" data-active={activeSection === "view"} onClick={() => setActiveSection("view")}>
            Open View Section
          </button>
        </div>

        {activeSection === "load" && (
          <LoadPlySection
            hasUpload={Boolean(uploadedPly)}
            uploadedSourceName={uploadedPly?.sourceName ?? null}
            clusterCount={clusterCount}
            onClusterChange={setClusterCount}
            onAnalyze={onAnalyze}
            onExport={onExport}
            canAnalyze={Boolean(uploadedPly)}
            canExport={Boolean(state && posed && posed.ok)}
          />
        )}

        {activeSection === "view" && (
          <PlySplatViewerSection
            hasUpload={Boolean(uploadedPly)}
            hasViewerData={Boolean(viewerSource)}
            uploadedSourceName={uploadedPly?.sourceName ?? null}
            onLoadNormalViewer={onLoadNormalViewer}
            onLoadSuperSplatViewer={onLoadSuperSplatViewer}
            viewerMode={viewerMode}
            viewerYawDeg={viewerYawDeg}
            viewerPitchDeg={viewerPitchDeg}
            viewerPanX={viewerPanX}
            viewerPanY={viewerPanY}
            viewerZoom={viewerZoom}
            viewerPerspective={viewerPerspective}
            viewerDensity={viewerDensity}
            viewerDebugMode={viewerDebugMode}
            viewerInfluenceMode={viewerInfluenceMode}
            viewerPoseMode={viewerPoseMode}
            viewerSampleCount={viewerSample?.count ?? 0}
            bones={state?.bindRig.bones ?? []}
            selectedBoneId={selectedBoneId}
            onSelectBone={setSelectedBoneId}
            selectedBoneName={viewerDebugStats.selectedBoneName}
            affectedSplatCount={viewerDebugStats.affectedSplatCount}
            averageWeight={viewerDebugStats.averageWeight}
            onZoomChange={setViewerZoom}
            onPerspectiveChange={setViewerPerspective}
            onDebugModeChange={setViewerDebugMode}
            onInfluenceModeChange={setViewerInfluenceMode}
            onPoseModeChange={setViewerPoseMode}
            selectedBoneEdit={selectedBoneEdit}
            onSelectedBoneEditChange={(field, value) => {
              if (selectedBoneId === null) return;
              setManualBoneEdits((prev) => {
                const current = prev[selectedBoneId] ?? EMPTY_MANUAL_EDIT;
                return {
                  ...prev,
                  [selectedBoneId]: {
                    ...current,
                    [field]: value,
                  },
                };
              });
            }}
            onResetSelectedBonePose={() => {
              if (selectedBoneId === null) return;
              setManualBoneEdits((prev) => {
                const next = { ...prev };
                delete next[selectedBoneId];
                return next;
              });
            }}
            onResetAllBonePose={() => setManualBoneEdits({})}
            splatOpacity={viewerSplatOpacity}
            skeletonOpacity={viewerSkeletonOpacity}
            influenceIntensity={viewerInfluenceIntensity}
            covarianceIntensity={viewerCovarianceIntensity}
            onSplatOpacityChange={setViewerSplatOpacity}
            onSkeletonOpacityChange={setViewerSkeletonOpacity}
            onInfluenceIntensityChange={setViewerInfluenceIntensity}
            onCovarianceIntensityChange={setViewerCovarianceIntensity}
            onPanReset={() => {
              setViewerPanX(0);
              setViewerPanY(0);
            }}
            onDensityChange={setViewerDensity}
            onReset={() => {
              setViewerYawDeg(0);
              setViewerPitchDeg(0);
              setViewerPanX(0);
              setViewerPanY(0);
              setViewerZoom(1.3);
              setViewerPerspective(1.7);
              setViewerDensity(78);
              setViewerDebugMode("normalAvatar");
              setViewerInfluenceMode("overlay");
              setViewerSplatOpacity(1);
              setViewerSkeletonOpacity(0.8);
              setViewerInfluenceIntensity(0.75);
              setViewerCovarianceIntensity(0.6);
              setManualBoneEdits({});
            }}
            canvasRef={viewerCanvasRef}
            onPointerDown={onViewerPointerDown}
            onPointerMove={onViewerPointerMove}
            onPointerUp={onViewerPointerUp}
            onWheel={onViewerWheel}
            viewerHtml={viewerBundle?.frameHtml ?? null}
          />
        )}
      </section>

      {isLoading && <div className="status-line">Analyzing splat structure and preparing skeleton bindings...</div>}
      {error && <div className="status-line error">{error}</div>}
      {poseError && <div className="status-line error">Realtime deformation fallback: {poseError}</div>}

      {state && activeSection === "load" && (
        <>
          <section className="summary-grid">
            <article>
              <h2>Input</h2>
              <p>
                <strong>Source:</strong> {state.sourceName}
              </p>
              <p>
                <strong>Splat Count:</strong> {state.data.count.toLocaleString()}
              </p>
              <p>
                <strong>Interactive Rig Sample:</strong> {state.rigSampleCount.toLocaleString()} / {state.sourceCount.toLocaleString()}
              </p>
              <p>
                <strong>Attributes:</strong> {state.data.propertyNames.join(", ")}
              </p>
            </article>

            <article>
              <h2>Segmentation</h2>
              <p>
                <strong>Face:</strong> {state.segmentation.counts.face.toLocaleString()}
              </p>
              <p>
                <strong>Head:</strong> {state.segmentation.counts.head.toLocaleString()}
              </p>
              <p>
                <strong>Body:</strong> {state.segmentation.counts.body.toLocaleString()}
              </p>
            </article>

            <article>
              <h2>Canonical + Rig</h2>
              <p>
                <strong>Normalized Height:</strong> {state.canonical.height.toFixed(2)}
              </p>
              <p>
                <strong>Bone Count:</strong> {state.bindRig.bones.length}
              </p>
              <p>
                <strong>Influences per Splat:</strong> {state.skinning.influencesPerSplat}
              </p>
              <p>
                <strong>Forward:</strong> {state.canonical.basis.forward.map((v: number) => v.toFixed(3)).join(", ")}
              </p>
              <p>
                <strong>Realtime Deformation:</strong> Stabilized skinning + covariance-aware rotation
              </p>
            </article>
          </section>

          <section className="pose-controls-grid">
            <article>
              <h2>Test Deformation Controls</h2>
              <label>
                <span>Rotate Arms: {controls.armRotateDeg.toFixed(0)} deg</span>
                <input
                  type="range"
                  min={-75}
                  max={75}
                  value={controls.armRotateDeg}
                  onChange={(e) =>
                    setControls((prev: TestControls) => ({ ...prev, armRotateDeg: Number(e.target.value) }))
                  }
                />
              </label>
              <label>
                <span>Bend Spine: {controls.spineBendDeg.toFixed(0)} deg</span>
                <input
                  type="range"
                  min={-35}
                  max={35}
                  value={controls.spineBendDeg}
                  onChange={(e) =>
                    setControls((prev: TestControls) => ({ ...prev, spineBendDeg: Number(e.target.value) }))
                  }
                />
              </label>
              <label>
                <span>Rotate Head: {controls.headRotateDeg.toFixed(0)} deg</span>
                <input
                  type="range"
                  min={-75}
                  max={75}
                  value={controls.headRotateDeg}
                  onChange={(e) =>
                    setControls((prev: TestControls) => ({ ...prev, headRotateDeg: Number(e.target.value) }))
                  }
                />
              </label>
              <label>
                <span>Twist Torso: {controls.torsoTwistDeg.toFixed(0)} deg</span>
                <input
                  type="range"
                  min={-65}
                  max={65}
                  value={controls.torsoTwistDeg}
                  onChange={(e) =>
                    setControls((prev: TestControls) => ({ ...prev, torsoTwistDeg: Number(e.target.value) }))
                  }
                />
              </label>
            </article>
            <article>
              <h2>Deformation Preview</h2>
              <p>Blue points: canonical bind pose. Orange points: current test deformation.</p>
              <canvas ref={deformationCanvasRef} className="deformation-canvas" />
            </article>
          </section>

          {posed && posed.ok && (
            <section className="weight-debug-grid">
            <article>
              <h2>Visual Weight Debugging</h2>
              <p>Switch between dominant bone color, weight heatmap, and region blending influence.</p>
              <div className="debug-mode-switch">
                <button type="button" onClick={() => setDebugMode("bone")} data-active={debugMode === "bone"}>
                  Bone Colors
                </button>
                <button type="button" onClick={() => setDebugMode("weight")} data-active={debugMode === "weight"}>
                  Weight Heatmap
                </button>
                <button type="button" onClick={() => setDebugMode("blend")} data-active={debugMode === "blend"}>
                  Blend Score
                </button>
              </div>
              <canvas ref={weightDebugCanvasRef} className="deformation-canvas" />
            </article>
            </section>
          )}

          <section className="hist-grid">
            <MiniHistogram title="Scale Distribution" bins={scaleHist} />
            <MiniHistogram title="Opacity Distribution" bins={opacityHist} />
            <MiniHistogram title="Covariance Trace" bins={covHist} />
          </section>

          <section className="viz-grid">
            <figure>
              <figcaption>Face Region Density (X/Y heatmap)</figcaption>
              <canvas ref={faceCanvasRef} />
            </figure>
            <figure>
              <figcaption>Body Region Density (X/Y heatmap)</figcaption>
              <canvas ref={bodyCanvasRef} />
            </figure>
            <figure className="orientation-plot">
              <figcaption>Splat Orientation After Bone/Covariance Deformation (XY projection)</figcaption>
              <canvas ref={orientationCanvasRef} />
            </figure>
          </section>

          <section className="clusters-section">
            <h2>Clustering Tools</h2>
            <p>Centroid and per-cluster population from K-Means over splat positions.</p>
            <div className="cluster-list">
              {state.clusters.centroids.map((c, i) => (
                <div key={`cluster-${i}`} className="cluster-item">
                  <span className="cluster-name">Cluster {i}</span>
                  <span>Count: {state.clusters.counts[i].toLocaleString()}</span>
                  <span>
                    Center: {c[0].toFixed(3)}, {c[1].toFixed(3)}, {c[2].toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
