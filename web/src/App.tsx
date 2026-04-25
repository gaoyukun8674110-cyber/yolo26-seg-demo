import { useEffect, useState } from "react";

import { fetchExamples, fetchMetrics } from "./api";
import ExampleGrid from "./components/ExampleGrid";
import LiveDemo from "./components/LiveDemo";
import MetricGrid from "./components/MetricGrid";
import "./styles.css";
import type { ExampleItem, Metrics } from "./types";

export default function App() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [examples, setExamples] = useState<ExampleItem[]>([]);

  useEffect(() => {
    let active = true;

    async function loadPageData() {
      try {
        const [metricsPayload, examplesPayload] = await Promise.all([
          fetchMetrics(),
          fetchExamples(),
        ]);
        if (!active) {
          return;
        }
        setMetrics(metricsPayload);
        setExamples(examplesPayload.items);
      } catch (error) {
        console.error(error);
      }
    }

    void loadPageData();

    return () => {
      active = false;
    };
  }, []);

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <p className="eyebrow">YOLO26 / MVTec / Resume Demo</p>
        <h1>Industrial Defect Segmentation</h1>
        <p className="hero-copy">
          A lightweight industrial-vision showcase that reframes MVTec AD into a
          supervised segmentation workflow for YOLO26.
        </p>
        <MetricGrid metrics={metrics} />
      </section>

      <section className="content-grid">
        <article className="panel">
          <p className="panel-index">01</p>
          <h2>Data Reframing</h2>
          <p>
            Ground-truth masks are converted into YOLO segmentation labels and
            reorganized into train, validation, and test splits.
          </p>
        </article>

        <article className="panel">
          <p className="panel-index">02</p>
          <h2>Model Results</h2>
          <p className="results-copy">
            The API serves sample images and dataset summaries so the page can
            function before a trained YOLO26 segmentation weight is attached.
          </p>
          <ExampleGrid items={examples} />
        </article>

        <article className="panel panel-wide">
          <p className="panel-index">03</p>
          <h2>Live Demo</h2>
          <LiveDemo />
        </article>
      </section>
    </main>
  );
}
