type LoadPlySectionProps = {
  hasUpload: boolean;
  uploadedSourceName: string | null;
  clusterCount: number;
  onClusterChange: (value: number) => void;
  onAnalyze: () => void;
  onExport: () => void;
  canAnalyze: boolean;
  canExport: boolean;
};

export default function LoadPlySection({
  hasUpload,
  uploadedSourceName,
  clusterCount,
  onClusterChange,
  onAnalyze,
  onExport,
  canAnalyze,
  canExport,
}: LoadPlySectionProps) {
  return (
    <article className="workflow-card">
      <h2>Section 2A: Analyze and Generate JSON</h2>
      <p>Run analysis on the uploaded .ply, then generate animation-ready metadata JSON.</p>
      <p>{hasUpload ? `Ready: ${uploadedSourceName}` : "Upload a .ply in Step 1 first."}</p>
      <div className="inspector-controls">
        <label className="cluster-label">
          <span>Clusters: {clusterCount}</span>
          <input
            type="range"
            min={4}
            max={20}
            value={clusterCount}
            onChange={(e) => onClusterChange(Number(e.target.value))}
          />
        </label>

        <button type="button" onClick={onAnalyze} disabled={!canAnalyze}>
          Analyze Uploaded PLY
        </button>

        <button type="button" onClick={onExport} disabled={!canExport}>
          Generate + Download JSON
        </button>
      </div>
    </article>
  );
}
