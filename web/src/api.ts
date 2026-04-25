import type { ExampleItem, Metrics, PredictResponse } from "./types";

function buildUrl(path: string): string {
  return new URL(path, window.location.origin).toString();
}

export async function fetchMetrics(): Promise<Metrics> {
  const response = await fetch(buildUrl("/metrics"));
  if (!response.ok) {
    throw new Error("Failed to load metrics");
  }
  return (await response.json()) as Metrics;
}

export async function fetchExamples(): Promise<{ items: ExampleItem[] }> {
  const response = await fetch(buildUrl("/examples"));
  if (!response.ok) {
    throw new Error("Failed to load examples");
  }
  return (await response.json()) as { items: ExampleItem[] };
}

export async function uploadImage(file: File): Promise<PredictResponse> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(buildUrl("/predict"), {
    method: "POST",
    body,
  });
  if (!response.ok) {
    throw new Error("Failed to run prediction");
  }
  return (await response.json()) as PredictResponse;
}
