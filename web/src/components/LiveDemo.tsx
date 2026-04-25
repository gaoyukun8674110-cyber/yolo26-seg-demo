import { useState } from "react";

import { uploadImage } from "../api";
import type { PredictResponse } from "../types";

export default function LiveDemo() {
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const result = await uploadImage(file);
      setPrediction(result);
    } catch (caughtError) {
      const message = caughtError instanceof Error ? caughtError.message : "Upload failed";
      setError(message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="live-demo">
      <label className="upload-label" htmlFor="inspection-upload">
        Upload inspection image
      </label>
      <input
        id="inspection-upload"
        className="upload-input"
        type="file"
        accept="image/png,image/jpeg"
        onChange={handleFileChange}
      />
      <p className="upload-help">
        {uploading ? "Running inference..." : "This local demo stores the returned overlay under /artifacts/generated."}
      </p>
      {error ? <p className="error-copy">{error}</p> : null}
      {prediction ? (
        <div className="prediction-panel">
          <div>
            <p className="prediction-title">
              {prediction.has_defect ? "Potential defect detected" : "No defect detected in demo mode"}
            </p>
            <p className="prediction-meta">
              Confidence {prediction.confidence} · {prediction.latency_ms} ms
            </p>
          </div>
          <img
            src={prediction.overlay_url}
            alt="Prediction overlay"
            className="prediction-image"
          />
        </div>
      ) : null}
    </div>
  );
}
