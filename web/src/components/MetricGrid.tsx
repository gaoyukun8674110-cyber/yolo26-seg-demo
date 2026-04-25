import type { Metrics } from "../types";

type MetricGridProps = {
  metrics: Metrics | null;
};

export default function MetricGrid({ metrics }: MetricGridProps) {
  if (!metrics) {
    return <p className="loading-copy">Loading metrics...</p>;
  }

  return (
    <div className="metric-grid">
      <div className="metric-card">
        <span className="metric-label">Dataset</span>
        <strong className="metric-value">{metrics.total_samples} samples</strong>
      </div>
      <div className="metric-card">
        <span className="metric-label">Train / Val / Test</span>
        <strong className="metric-value">
          {metrics.train_samples} / {metrics.val_samples} / {metrics.test_samples}
        </strong>
      </div>
      <div className="metric-card">
        <span className="metric-label">Good / Defect</span>
        <strong className="metric-value">
          {metrics.good_samples} / {metrics.defect_samples}
        </strong>
      </div>
      <div className="metric-card">
        <span className="metric-label">Runtime</span>
        <strong className="metric-value">
          {metrics.latency_ms} ms · {metrics.model_status}
        </strong>
      </div>
    </div>
  );
}
