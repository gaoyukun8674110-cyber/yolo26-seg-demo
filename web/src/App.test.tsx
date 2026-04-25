import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

describe("App", () => {
  it("renders the core showcase sections", () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/metrics")) {
          return new Response(
            JSON.stringify({
              task: "defect-segmentation",
              total_samples: 978,
              train_samples: 692,
              val_samples: 143,
              test_samples: 143,
              defect_samples: 265,
              good_samples: 713,
              latency_ms: 0,
              model_status: "demo-mode"
            }),
          );
        }

        if (url.endsWith("/examples")) {
          return new Response(JSON.stringify({ items: [] }));
        }

        return new Response("Not found", { status: 404 });
      }),
    );
    render(<App />);

    expect(screen.getByText(/Industrial Defect Segmentation/i)).toBeInTheDocument();
    expect(screen.getByText(/Data Reframing/i)).toBeInTheDocument();
    expect(screen.getByText(/Live Demo/i)).toBeInTheDocument();
  });
});

describe("App API integration", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    fetchMock.mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/metrics")) {
        return new Response(
          JSON.stringify({
            task: "defect-segmentation",
            total_samples: 978,
            train_samples: 692,
            val_samples: 143,
            test_samples: 143,
            defect_samples: 265,
            good_samples: 713,
            latency_ms: 0,
            model_status: "demo-mode"
          }),
        );
      }

      if (url.endsWith("/examples")) {
        return new Response(
          JSON.stringify({
            items: [
              {
                id: "bottle-source-sample",
                category: "bottle",
                status: "converted source sample",
                image: "/artifacts/examples/bottle-source-sample.png"
              }
            ]
          }),
        );
      }

      if (url.endsWith("/predict") && init?.method === "POST") {
        return new Response(
          JSON.stringify({
            has_defect: false,
            confidence: 0,
            latency_ms: 0,
            overlay_url: "/artifacts/generated/demo-overlay.png"
          }),
        );
      }

      return new Response("Not found", { status: 404 });
    });

    vi.stubGlobal("fetch", fetchMock);
  });

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

  it("renders fetched metrics, examples, and upload results", async () => {
    render(<App />);

    expect((await screen.findAllByText(/978 samples/i)).length).toBeGreaterThan(0);
    expect((await screen.findAllByText(/bottle · converted source sample/i)).length).toBeGreaterThan(0);

    const file = new File(["demo"], "demo.png", { type: "image/png" });
    fireEvent.change(screen.getByLabelText(/Upload inspection image/i), {
      target: { files: [file] }
    });

    expect(await screen.findByText(/No defect detected in demo mode/i)).toBeInTheDocument();
    expect(await screen.findByAltText(/Prediction overlay/i)).toHaveAttribute(
      "src",
      "/artifacts/generated/demo-overlay.png",
    );
  });
});
