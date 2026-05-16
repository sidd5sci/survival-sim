import type { SkeletonBone } from "@/simulations/splatsWeb/avatarPreparation";
import type { PointerEventHandler, RefObject, WheelEventHandler } from "react";

type PlySplatViewerSectionProps = {
  hasUpload: boolean;
  hasViewerData: boolean;
  uploadedSourceName: string | null;
  onLoadNormalViewer: () => void;
  onLoadSuperSplatViewer: () => void;
  viewerMode: "none" | "normal" | "supersplat";
  viewerYawDeg: number;
  viewerPitchDeg: number;
  viewerPanX: number;
  viewerPanY: number;
  viewerZoom: number;
  viewerPerspective: number;
  viewerDensity: number;
  viewerDebugMode:
    | "normalAvatar"
    | "skeletonOverlay"
    | "boneInfluenceOverlay"
    | "weightHeatmap"
    | "covarianceOrientation"
    | "skinningDebug"
    | "bindPose"
    | "poseTest"
    | "transformDebug";
  viewerInfluenceMode: "overlay" | "heatmap" | "wireframe" | "blend";
  viewerPoseMode: "bind" | "deformed";
  viewerSampleCount: number;
  bones: SkeletonBone[];
  selectedBoneId: number | null;
  onSelectBone: (boneId: number | null) => void;
  selectedBoneName: string;
  affectedSplatCount: number;
  averageWeight: number;
  onZoomChange: (value: number) => void;
  onPerspectiveChange: (value: number) => void;
  onDebugModeChange: (
    mode:
      | "normalAvatar"
      | "skeletonOverlay"
      | "boneInfluenceOverlay"
      | "weightHeatmap"
      | "covarianceOrientation"
      | "skinningDebug"
      | "bindPose"
      | "poseTest"
      | "transformDebug",
  ) => void;
  onInfluenceModeChange: (mode: "overlay" | "heatmap" | "wireframe" | "blend") => void;
  onPoseModeChange: (mode: "bind" | "deformed") => void;
  selectedBoneEdit: {
    tx: number;
    ty: number;
    tz: number;
    rxDeg: number;
    ryDeg: number;
    rzDeg: number;
  };
  onSelectedBoneEditChange: (field: "tx" | "ty" | "tz" | "rxDeg" | "ryDeg" | "rzDeg", value: number) => void;
  onResetSelectedBonePose: () => void;
  onResetAllBonePose: () => void;
  splatOpacity: number;
  skeletonOpacity: number;
  influenceIntensity: number;
  covarianceIntensity: number;
  onSplatOpacityChange: (value: number) => void;
  onSkeletonOpacityChange: (value: number) => void;
  onInfluenceIntensityChange: (value: number) => void;
  onCovarianceIntensityChange: (value: number) => void;
  onPanReset: () => void;
  onDensityChange: (value: number) => void;
  onReset: () => void;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  onPointerDown: PointerEventHandler<HTMLCanvasElement>;
  onPointerMove: PointerEventHandler<HTMLCanvasElement>;
  onPointerUp: PointerEventHandler<HTMLCanvasElement>;
  onWheel: WheelEventHandler<HTMLCanvasElement>;
  viewerHtml: string | null;
};

export default function PlySplatViewerSection({
  hasUpload,
  hasViewerData,
  uploadedSourceName,
  onLoadNormalViewer,
  onLoadSuperSplatViewer,
  viewerMode,
  viewerYawDeg,
  viewerPitchDeg,
  viewerPanX,
  viewerPanY,
  viewerZoom,
  viewerPerspective,
  viewerDensity,
  viewerDebugMode,
  viewerInfluenceMode,
  viewerPoseMode,
  viewerSampleCount,
  bones,
  selectedBoneId,
  onSelectBone,
  selectedBoneName,
  affectedSplatCount,
  averageWeight,
  onZoomChange,
  onPerspectiveChange,
  onDebugModeChange,
  onInfluenceModeChange,
  onPoseModeChange,
  selectedBoneEdit,
  onSelectedBoneEditChange,
  onResetSelectedBonePose,
  onResetAllBonePose,
  splatOpacity,
  skeletonOpacity,
  influenceIntensity,
  covarianceIntensity,
  onSplatOpacityChange,
  onSkeletonOpacityChange,
  onInfluenceIntensityChange,
  onCovarianceIntensityChange,
  onPanReset,
  onDensityChange,
  onReset,
  canvasRef,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onWheel,
  viewerHtml,
}: PlySplatViewerSectionProps) {
  return (
    <article className="workflow-card workflow-card-full">
      <h2>Section 2B: View .ply (Full Width Splat Preview)</h2>
      <p>Choose either normal viewer or SuperSplat viewer for the uploaded file.</p>
      <p>{hasUpload ? `Ready: ${uploadedSourceName}` : "Upload a .ply in Step 1 first."}</p>
      <div className="viewer-controls">
        <button type="button" onClick={onLoadNormalViewer} disabled={!hasUpload}>
          Open Normal Viewer
        </button>
        <button type="button" onClick={onLoadSuperSplatViewer} disabled={!hasUpload}>
          Open In SuperSplat Viewer
        </button>
        {viewerMode === "normal" && (
          <>
            <label>
              <span>Select Bone</span>
              <select
                value={selectedBoneId === null ? "" : String(selectedBoneId)}
                onChange={(e) => {
                  const val = e.target.value;
                  onSelectBone(val === "" ? null : Number(val));
                }}
              >
                <option value="">No Highlight</option>
                {bones.map((bone) => (
                  <option key={bone.id} value={bone.id}>
                    {bone.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Viewer Mode</span>
              <select
                value={viewerDebugMode}
                onChange={(e) =>
                  onDebugModeChange(
                    e.target.value as
                      | "normalAvatar"
                      | "skeletonOverlay"
                      | "boneInfluenceOverlay"
                      | "weightHeatmap"
                      | "covarianceOrientation"
                      | "skinningDebug"
                      | "bindPose"
                      | "poseTest"
                      | "transformDebug",
                  )
                }
              >
                <option value="normalAvatar">Normal Avatar</option>
                <option value="skeletonOverlay">Skeleton Overlay</option>
                <option value="boneInfluenceOverlay">Bone Influence Overlay</option>
                <option value="weightHeatmap">Weight Heatmap</option>
                <option value="covarianceOrientation">Covariance Orientation</option>
                <option value="skinningDebug">Skinning Debug</option>
                <option value="bindPose">Bind Pose</option>
                <option value="poseTest">Pose Test</option>
                <option value="transformDebug">Transform Debug (Green/Red)</option>
              </select>
            </label>
            <label>
              <span>Pose Source</span>
              <select value={viewerPoseMode} onChange={(e) => onPoseModeChange(e.target.value as "bind" | "deformed")}>
                <option value="bind">Bind Pose</option>
                <option value="deformed">Deformed Pose</option>
              </select>
            </label>
            <label>
              <span>Influence Mode</span>
              <select
                value={viewerInfluenceMode}
                onChange={(e) => onInfluenceModeChange(e.target.value as "overlay" | "heatmap" | "wireframe" | "blend")}
              >
                <option value="overlay">Overlay Highlight</option>
                <option value="heatmap">Heatmap Glow</option>
                <option value="wireframe">Wireframe Influence</option>
                <option value="blend">Blended Skinning Weights</option>
              </select>
            </label>
            <label>
              <span>Splat Opacity: {Math.round(splatOpacity * 100)}%</span>
              <input
                type="range"
                min={0.15}
                max={1.2}
                step={0.01}
                value={splatOpacity}
                onChange={(e) => onSplatOpacityChange(Number(e.target.value))}
              />
            </label>
            <label>
              <span>Skeleton Opacity: {Math.round(skeletonOpacity * 100)}%</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={skeletonOpacity}
                onChange={(e) => onSkeletonOpacityChange(Number(e.target.value))}
              />
            </label>
            <label>
              <span>Influence Intensity: {Math.round(influenceIntensity * 100)}%</span>
              <input
                type="range"
                min={0}
                max={1.4}
                step={0.01}
                value={influenceIntensity}
                onChange={(e) => onInfluenceIntensityChange(Number(e.target.value))}
              />
            </label>
            <label>
              <span>Covariance Intensity: {Math.round(covarianceIntensity * 100)}%</span>
              <input
                type="range"
                min={0}
                max={1.4}
                step={0.01}
                value={covarianceIntensity}
                onChange={(e) => onCovarianceIntensityChange(Number(e.target.value))}
              />
            </label>
            <label>
              <span>Zoom: {viewerZoom.toFixed(2)}x</span>
              <input
                type="range"
                min={0.7}
                max={2.6}
                step={0.01}
                value={viewerZoom}
                onChange={(e) => onZoomChange(Number(e.target.value))}
              />
            </label>
            <label>
              <span>Perspective: {viewerPerspective.toFixed(2)}</span>
              <input
                type="range"
                min={1.0}
                max={3.2}
                step={0.01}
                value={viewerPerspective}
                onChange={(e) => onPerspectiveChange(Number(e.target.value))}
              />
            </label>
            <label>
              <span>Density: {viewerDensity}%</span>
              <input
                type="range"
                min={20}
                max={100}
                value={viewerDensity}
                onChange={(e) => onDensityChange(Number(e.target.value))}
              />
            </label>
            <button type="button" onClick={onReset} disabled={!hasViewerData}>
              Reset View
            </button>
            <button type="button" onClick={onPanReset} disabled={!hasViewerData}>
              Center Pan
            </button>
          </>
        )}
      </div>

      {viewerMode === "normal" && hasViewerData && (
        <>
          <div className="viewer-meta">
            <span>Yaw: {viewerYawDeg.toFixed(1)} deg</span>
            <span>Pitch: {viewerPitchDeg.toFixed(1)} deg</span>
            <span>Pan: {viewerPanX.toFixed(2)}, {viewerPanY.toFixed(2)}</span>
            <span>Viewer sample: {viewerSampleCount.toLocaleString()}</span>
            <span>Skeleton bones: {bones.length}</span>
          </div>
          <div className="viewer-stats-grid">
            <div>
              <strong>Selected Bone:</strong> {selectedBoneName}
            </div>
            <div>
              <strong>Affected Splats:</strong> {affectedSplatCount.toLocaleString()}
            </div>
            <div>
              <strong>Average Weight:</strong> {averageWeight.toFixed(3)}
            </div>
          </div>
          {bones.length > 0 && (
            <p className="viewer-note">
              Left drag: orbit. Right drag or Shift+drag: pan. Wheel: zoom. Bone axes are RGB (X red, Y green, Z blue).
            </p>
          )}
          <div className="viewer-canvas-wrap">
            <canvas
              ref={canvasRef}
              className="splat-viewer-canvas"
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              onPointerLeave={onPointerUp}
              onWheel={onWheel}
              onContextMenu={(e) => e.preventDefault()}
            />
            {selectedBoneId !== null && (
              <div className="viewer-overlay-panel">
                <h4>Selected Bone Controls</h4>
                <label>
                  <span>Move X: {selectedBoneEdit.tx.toFixed(3)}</span>
                  <input
                    type="range"
                    min={-0.5}
                    max={0.5}
                    step={0.001}
                    value={selectedBoneEdit.tx}
                    onChange={(e) => onSelectedBoneEditChange("tx", Number(e.target.value))}
                  />
                </label>
                <label>
                  <span>Move Y: {selectedBoneEdit.ty.toFixed(3)}</span>
                  <input
                    type="range"
                    min={-0.5}
                    max={0.5}
                    step={0.001}
                    value={selectedBoneEdit.ty}
                    onChange={(e) => onSelectedBoneEditChange("ty", Number(e.target.value))}
                  />
                </label>
                <label>
                  <span>Move Z: {selectedBoneEdit.tz.toFixed(3)}</span>
                  <input
                    type="range"
                    min={-0.5}
                    max={0.5}
                    step={0.001}
                    value={selectedBoneEdit.tz}
                    onChange={(e) => onSelectedBoneEditChange("tz", Number(e.target.value))}
                  />
                </label>
                <label>
                  <span>Rotate X: {selectedBoneEdit.rxDeg.toFixed(1)} deg</span>
                  <input
                    type="range"
                    min={-360}
                    max={360}
                    step={0.1}
                    value={selectedBoneEdit.rxDeg}
                    onChange={(e) => onSelectedBoneEditChange("rxDeg", Number(e.target.value))}
                  />
                </label>
                <label>
                  <span>Rotate Y: {selectedBoneEdit.ryDeg.toFixed(1)} deg</span>
                  <input
                    type="range"
                    min={-360}
                    max={360}
                    step={0.1}
                    value={selectedBoneEdit.ryDeg}
                    onChange={(e) => onSelectedBoneEditChange("ryDeg", Number(e.target.value))}
                  />
                </label>
                <label>
                  <span>Rotate Z: {selectedBoneEdit.rzDeg.toFixed(1)} deg</span>
                  <input
                    type="range"
                    min={-360}
                    max={360}
                    step={0.1}
                    value={selectedBoneEdit.rzDeg}
                    onChange={(e) => onSelectedBoneEditChange("rzDeg", Number(e.target.value))}
                  />
                </label>
                <div className="viewer-overlay-actions">
                  <button type="button" onClick={onResetSelectedBonePose}>
                    Reset Bone
                  </button>
                  <button type="button" onClick={onResetAllBonePose}>
                    Reset All
                  </button>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {viewerMode === "supersplat" && hasViewerData && viewerHtml && (
        <iframe
          className="supersplat-frame"
          srcDoc={viewerHtml}
          title="SuperSplat Viewer"
          allow="fullscreen; xr-spatial-tracking"
        />
      )}
    </article>
  );
}
