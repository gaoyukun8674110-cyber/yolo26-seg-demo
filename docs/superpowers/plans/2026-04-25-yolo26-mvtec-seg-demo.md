# YOLO26 MVTec Seg Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resume-focused YOLO26 industrial defect segmentation project that converts MVTec AD into a supervised segmentation dataset, exposes a lightweight FastAPI inference API, and presents results in a single-page web demo.

**Architecture:** The repository is a small monorepo with three focused areas: `scripts/` for dataset conversion and artifact generation, `api/` for inference and static result endpoints, and `web/` for a React/Vite showcase UI. The API reads precomputed metrics/examples from JSON artifacts and exposes a single-image `/predict` endpoint backed by an injectable inference service so the code is testable without a real model file.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, Pytest, NumPy, OpenCV, Pillow, Ultralytics, React, Vite, TypeScript, Vitest, Testing Library

---

### Task 1: Bootstrap The API And Health Check

**Files:**
- Create: `.gitignore`
- Create: `api/requirements.txt`
- Create: `api/app/__init__.py`
- Create: `api/app/core/settings.py`
- Create: `api/app/api/__init__.py`
- Create: `api/app/api/routes.py`
- Create: `api/app/main.py`
- Create: `api/tests/conftest.py`
- Create: `api/tests/test_health.py`

- [ ] **Step 1: Write the failing health test**

```python
from fastapi.testclient import TestClient

from api.app.main import create_app


def test_health_returns_ok_payload() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "yolo26-mvtec-seg-demo"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest api/tests/test_health.py -q`

Expected: FAIL with `ModuleNotFoundError` or import failure because `api.app.main` does not exist yet.

- [ ] **Step 3: Write the minimal API bootstrap**

`.gitignore`
```gitignore
__pycache__/
.pytest_cache/
.venv/
node_modules/
dist/
coverage/
artifacts/generated/
data/processed/
models/checkpoints/
models/exports/
```

`api/requirements.txt`
```text
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9
numpy==2.1.1
opencv-python-headless==4.10.0.84
Pillow==10.4.0
pytest==8.3.3
httpx==0.27.2
```

`api/app/core/settings.py`
```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    service_name: str = "yolo26-mvtec-seg-demo"
    project_root: Path = Path(__file__).resolve().parents[3]


def get_settings() -> Settings:
    return Settings()
```

`api/app/api/routes.py`
```python
from fastapi import APIRouter

from api.app.core.settings import get_settings

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "service": settings.service_name}
```

`api/app/main.py`
```python
from fastapi import FastAPI

from api.app.api.routes import router
from api.app.core.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.service_name)
    app.include_router(router)
    return app


app = create_app()
```

`api/tests/conftest.py`
```python
from fastapi.testclient import TestClient

from api.app.main import create_app


def build_client() -> TestClient:
    return TestClient(create_app())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest api/tests/test_health.py -q`

Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add .gitignore api/requirements.txt api/app api/tests
git commit -m "feat: bootstrap fastapi health endpoint"
```

### Task 2: Add Static Metrics And Example Endpoints

**Files:**
- Create: `artifacts/metrics/summary.json`
- Create: `artifacts/examples/gallery.json`
- Create: `api/app/services/content_store.py`
- Create: `api/tests/test_content.py`
- Modify: `api/app/api/routes.py`

- [ ] **Step 1: Write failing tests for `/metrics` and `/examples`**

```python
from fastapi.testclient import TestClient

from api.app.main import create_app


def test_metrics_returns_summary_payload() -> None:
    client = TestClient(create_app())

    response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "defect-segmentation"
    assert "latency_ms" in payload


def test_examples_returns_gallery_items() -> None:
    client = TestClient(create_app())

    response = client.get("/examples")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["items"], list)
    assert payload["items"][0]["category"] == "bottle"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest api/tests/test_content.py -q`

Expected: FAIL with `404 != 200` because the routes do not exist yet.

- [ ] **Step 3: Implement the minimal content store and routes**

`artifacts/metrics/summary.json`
```json
{
  "task": "defect-segmentation",
  "categories": ["bottle", "capsule", "metal_nut"],
  "model": "yolo26s-seg",
  "latency_ms": 42.5,
  "dice": 0.81,
  "iou": 0.69
}
```

`artifacts/examples/gallery.json`
```json
{
  "items": [
    {
      "id": "bottle_broken_large_001",
      "category": "bottle",
      "status": "success",
      "image": "/artifacts/examples/bottle_broken_large_001_overlay.png"
    }
  ]
}
```

`api/app/services/content_store.py`
```python
import json
from pathlib import Path
from typing import Any


class ContentStore:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    def read_metrics(self) -> dict[str, Any]:
        metrics_path = self._project_root / "artifacts" / "metrics" / "summary.json"
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def read_examples(self) -> dict[str, Any]:
        examples_path = self._project_root / "artifacts" / "examples" / "gallery.json"
        return json.loads(examples_path.read_text(encoding="utf-8"))
```

`api/app/api/routes.py`
```python
from fastapi import APIRouter

from api.app.core.settings import get_settings
from api.app.services.content_store import ContentStore

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "service": settings.service_name}


@router.get("/metrics")
def metrics() -> dict:
    settings = get_settings()
    return ContentStore(settings.project_root).read_metrics()


@router.get("/examples")
def examples() -> dict:
    settings = get_settings()
    return ContentStore(settings.project_root).read_examples()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest api/tests/test_content.py -q`

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add artifacts/metrics/summary.json artifacts/examples/gallery.json api/app/services/content_store.py api/app/api/routes.py api/tests/test_content.py
git commit -m "feat: add static metrics and examples endpoints"
```

### Task 3: Add Predict Endpoint With Injectable Inference Service

**Files:**
- Create: `api/app/schemas/predict.py`
- Create: `api/app/services/inference.py`
- Create: `api/app/services/visualization.py`
- Create: `api/tests/test_predict.py`
- Modify: `api/app/main.py`
- Modify: `api/app/api/routes.py`

- [ ] **Step 1: Write the failing prediction test**

```python
from io import BytesIO

from PIL import Image
from fastapi.testclient import TestClient

from api.app.main import create_app
from api.app.services.inference import InferenceResult


class FakeInferenceService:
    def predict(self, image_bytes: bytes, filename: str) -> InferenceResult:
        return InferenceResult(
            has_defect=True,
            confidence=0.91,
            latency_ms=12.3,
            overlay_filename="demo-overlay.png",
        )


def test_predict_returns_overlay_summary() -> None:
    app = create_app(inference_service=FakeInferenceService())
    client = TestClient(app)
    image = Image.new("RGB", (32, 32), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    response = client.post(
        "/predict",
        files={"file": ("sample.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "has_defect": True,
        "confidence": 0.91,
        "latency_ms": 12.3,
        "overlay_url": "/artifacts/generated/demo-overlay.png",
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest api/tests/test_predict.py -q`

Expected: FAIL because `create_app()` does not yet accept an injected inference service and `/predict` does not exist.

- [ ] **Step 3: Implement the minimal prediction flow**

`api/app/schemas/predict.py`
```python
from pydantic import BaseModel


class PredictResponse(BaseModel):
    has_defect: bool
    confidence: float
    latency_ms: float
    overlay_url: str
```

`api/app/services/inference.py`
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceResult:
    has_defect: bool
    confidence: float
    latency_ms: float
    overlay_filename: str


class InferenceService:
    def predict(self, image_bytes: bytes, filename: str) -> InferenceResult:
        raise NotImplementedError("Wire the real YOLO26 model after the API contract is stable.")
```

`api/app/main.py`
```python
from fastapi import FastAPI

from api.app.api.routes import build_router
from api.app.core.settings import get_settings
from api.app.services.inference import InferenceService


def create_app(inference_service: InferenceService | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.service_name)
    app.include_router(build_router(inference_service or InferenceService()))
    return app


app = create_app()
```

`api/app/api/routes.py`
```python
from fastapi import APIRouter, File, UploadFile

from api.app.core.settings import get_settings
from api.app.schemas.predict import PredictResponse
from api.app.services.content_store import ContentStore
from api.app.services.inference import InferenceService


def build_router(inference_service: InferenceService) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        settings = get_settings()
        return {"status": "ok", "service": settings.service_name}

    @router.get("/metrics")
    def metrics() -> dict:
        settings = get_settings()
        return ContentStore(settings.project_root).read_metrics()

    @router.get("/examples")
    def examples() -> dict:
        settings = get_settings()
        return ContentStore(settings.project_root).read_examples()

    @router.post("/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(...)) -> PredictResponse:
        result = inference_service.predict(await file.read(), file.filename or "upload.png")
        return PredictResponse(
            has_defect=result.has_defect,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            overlay_url=f"/artifacts/generated/{result.overlay_filename}",
        )

    return router
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest api/tests/test_predict.py -q`

Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add api/app/main.py api/app/api/routes.py api/app/schemas/predict.py api/app/services/inference.py api/tests/test_predict.py
git commit -m "feat: add injectable prediction endpoint"
```

### Task 4: Convert MVTec Masks Into YOLO Segmentation Labels

**Files:**
- Create: `scripts/convert_mvtec_to_yolo_seg.py`
- Create: `tests/scripts/test_convert_mvtec_to_yolo_seg.py`

- [ ] **Step 1: Write the failing conversion tests**

```python
import numpy as np

from scripts.convert_mvtec_to_yolo_seg import mask_to_yolo_rows


def test_mask_to_yolo_rows_converts_a_single_rectangle() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:8] = 255

    rows = mask_to_yolo_rows(mask)

    assert len(rows) == 1
    assert rows[0].startswith("0 ")


def test_mask_to_yolo_rows_returns_empty_for_blank_mask() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)

    rows = mask_to_yolo_rows(mask)

    assert rows == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/scripts/test_convert_mvtec_to_yolo_seg.py -q`

Expected: FAIL with `ModuleNotFoundError` because the script module does not exist yet.

- [ ] **Step 3: Implement the minimal conversion module**

`scripts/convert_mvtec_to_yolo_seg.py`
```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml


def _normalize_point(x: int, y: int, width: int, height: int) -> tuple[float, float]:
    return round(x / width, 6), round(y / height, 6)


def mask_to_yolo_rows(mask: np.ndarray) -> list[str]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    rows: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) <= 0:
            continue
        points: list[str] = []
        for point in contour.reshape(-1, 2):
            norm_x, norm_y = _normalize_point(int(point[0]), int(point[1]), width, height)
            points.extend([str(norm_x), str(norm_y)])
        rows.append("0 " + " ".join(points))
    return rows


def write_data_yaml(output_dir: Path) -> None:
    payload = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "defect"},
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/scripts/test_convert_mvtec_to_yolo_seg.py -q`

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/convert_mvtec_to_yolo_seg.py tests/scripts/test_convert_mvtec_to_yolo_seg.py
git commit -m "feat: add mask to yolo segmentation conversion helpers"
```

### Task 5: Validate Converted Data And Export Demo Gallery Metadata

**Files:**
- Create: `scripts/validate_dataset.py`
- Create: `scripts/export_eval_gallery.py`
- Create: `tests/scripts/test_validate_dataset.py`
- Create: `tests/scripts/test_export_eval_gallery.py`

- [ ] **Step 1: Write failing tests for validation and gallery export**

```python
from pathlib import Path

from scripts.validate_dataset import summarize_split
from scripts.export_eval_gallery import build_gallery_payload


def test_summarize_split_counts_images_and_labels(tmp_path: Path) -> None:
    images_dir = tmp_path / "images" / "train"
    labels_dir = tmp_path / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (images_dir / "sample.png").write_bytes(b"png")
    (labels_dir / "sample.txt").write_text("", encoding="utf-8")

    summary = summarize_split(images_dir, labels_dir)

    assert summary["images"] == 1
    assert summary["labels"] == 1


def test_build_gallery_payload_normalizes_paths() -> None:
    payload = build_gallery_payload(
        [
            {
                "id": "sample",
                "category": "bottle",
                "status": "success",
                "overlay_path": "artifacts/examples/sample_overlay.png",
            }
        ]
    )

    assert payload["items"][0]["image"] == "/artifacts/examples/sample_overlay.png"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/scripts/test_validate_dataset.py tests/scripts/test_export_eval_gallery.py -q`

Expected: FAIL with import errors because both modules are missing.

- [ ] **Step 3: Implement the minimal validation and gallery helpers**

`scripts/validate_dataset.py`
```python
from pathlib import Path


def summarize_split(images_dir: Path, labels_dir: Path) -> dict[str, int]:
    image_count = sum(1 for _ in images_dir.glob("*") if _.is_file())
    label_count = sum(1 for _ in labels_dir.glob("*.txt"))
    empty_label_count = sum(1 for path in labels_dir.glob("*.txt") if path.read_text(encoding="utf-8").strip() == "")
    return {
        "images": image_count,
        "labels": label_count,
        "empty_labels": empty_label_count,
    }
```

`scripts/export_eval_gallery.py`
```python
from pathlib import PurePosixPath
from typing import Any


def build_gallery_payload(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    items = []
    for row in rows:
        image_path = "/" + str(PurePosixPath(row["overlay_path"]))
        items.append(
            {
                "id": row["id"],
                "category": row["category"],
                "status": row["status"],
                "image": image_path,
            }
        )
    return {"items": items}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/scripts/test_validate_dataset.py tests/scripts/test_export_eval_gallery.py -q`

Expected: PASS with `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_dataset.py scripts/export_eval_gallery.py tests/scripts/test_validate_dataset.py tests/scripts/test_export_eval_gallery.py
git commit -m "feat: add dataset validation and gallery export helpers"
```

### Task 6: Build The React Showcase Shell

**Files:**
- Create: `web/package.json`
- Create: `web/tsconfig.json`
- Create: `web/tsconfig.node.json`
- Create: `web/vite.config.ts`
- Create: `web/index.html`
- Create: `web/src/main.tsx`
- Create: `web/src/App.tsx`
- Create: `web/src/api.ts`
- Create: `web/src/types.ts`
- Create: `web/src/styles.css`
- Create: `web/src/App.test.tsx`

- [ ] **Step 1: Write the failing frontend smoke test**

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "./App";

describe("App", () => {
  it("renders the core showcase sections", () => {
    render(<App />);

    expect(screen.getByText(/Industrial Defect Segmentation/i)).toBeInTheDocument();
    expect(screen.getByText(/Data Reframing/i)).toBeInTheDocument();
    expect(screen.getByText(/Live Demo/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm --prefix web test -- --run`

Expected: FAIL because the Vite project and app files do not exist yet.

- [ ] **Step 3: Implement the minimal single-page frontend**

`web/package.json`
```json
{
  "name": "yolo26-mvtec-seg-demo-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "test": "vitest"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.2",
    "@testing-library/react": "^16.0.1",
    "@types/react": "^18.3.5",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "typescript": "^5.6.2",
    "vite": "^5.4.8",
    "vitest": "^2.1.1"
  }
}
```

`web/src/App.tsx`
```tsx
export default function App() {
  return (
    <main className="page">
      <section className="hero">
        <p className="eyebrow">YOLO26 Resume Project</p>
        <h1>Industrial Defect Segmentation</h1>
        <p>
          A lightweight demo that reframes MVTec AD into a supervised YOLO26
          segmentation workflow.
        </p>
      </section>

      <section className="panel">
        <h2>Data Reframing</h2>
        <p>Mask annotations are converted into YOLO segmentation labels with a single `defect` class.</p>
      </section>

      <section className="panel">
        <h2>Model Results</h2>
        <p>Metrics and curated gallery items will render here after the API is connected.</p>
      </section>

      <section className="panel">
        <h2>Live Demo</h2>
        <p>Upload one image and receive an overlay result from the FastAPI backend.</p>
      </section>
    </main>
  );
}
```

`web/src/styles.css`
```css
body {
  margin: 0;
  font-family: "Segoe UI", sans-serif;
  background: linear-gradient(180deg, #f4efe6 0%, #dde8e1 100%);
  color: #102418;
}

.page {
  max-width: 1040px;
  margin: 0 auto;
  padding: 40px 24px 80px;
}

.hero,
.panel {
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(16, 36, 24, 0.08);
  border-radius: 24px;
  padding: 24px;
  margin-bottom: 20px;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 12px;
  color: #4c6b5a;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm --prefix web test -- --run`

Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add web
git commit -m "feat: add react showcase shell"
```

### Task 7: Wire The Frontend To API Data And Upload Flow

**Files:**
- Modify: `web/src/App.tsx`
- Create: `web/src/api.ts`
- Create: `web/src/types.ts`
- Create: `web/src/components/MetricGrid.tsx`
- Create: `web/src/components/ExampleGrid.tsx`
- Create: `web/src/components/LiveDemo.tsx`
- Modify: `web/src/App.test.tsx`

- [ ] **Step 1: Write the failing integration-style UI test**

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import App from "./App";

vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
  if (String(input).endsWith("/metrics")) {
    return new Response(JSON.stringify({ task: "defect-segmentation", latency_ms: 42.5 }));
  }
  if (String(input).endsWith("/examples")) {
    return new Response(JSON.stringify({ items: [{ id: "sample", category: "bottle", status: "success", image: "/artifacts/examples/sample.png" }] }));
  }
  return new Response(JSON.stringify({ has_defect: true, confidence: 0.91, latency_ms: 12.3, overlay_url: "/artifacts/generated/demo.png" }));
}) as typeof fetch);

describe("App API integration", () => {
  it("renders fetched metrics and examples", async () => {
    render(<App />);

    expect(await screen.findByText(/42.5 ms/i)).toBeInTheDocument();
    expect(await screen.findByText(/bottle/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm --prefix web test -- --run`

Expected: FAIL because the app does not fetch or render API data yet.

- [ ] **Step 3: Implement the minimal API wiring**

`web/src/types.ts`
```ts
export type Metrics = {
  task: string;
  latency_ms: number;
  dice?: number;
  iou?: number;
};

export type ExampleItem = {
  id: string;
  category: string;
  status: string;
  image: string;
};

export type PredictResponse = {
  has_defect: boolean;
  confidence: number;
  latency_ms: number;
  overlay_url: string;
};
```

`web/src/api.ts`
```ts
import type { ExampleItem, Metrics, PredictResponse } from "./types";

const API_BASE = "http://localhost:8000";

export async function fetchMetrics(): Promise<Metrics> {
  const response = await fetch(`${API_BASE}/metrics`);
  return response.json();
}

export async function fetchExamples(): Promise<{ items: ExampleItem[] }> {
  const response = await fetch(`${API_BASE}/examples`);
  return response.json();
}

export async function uploadImage(file: File): Promise<PredictResponse> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`${API_BASE}/predict`, { method: "POST", body });
  return response.json();
}
```

`web/src/App.tsx`
```tsx
import { useEffect, useState } from "react";

import { fetchExamples, fetchMetrics } from "./api";
import type { ExampleItem, Metrics } from "./types";

export default function App() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [examples, setExamples] = useState<ExampleItem[]>([]);

  useEffect(() => {
    fetchMetrics().then(setMetrics);
    fetchExamples().then((payload) => setExamples(payload.items));
  }, []);

  return (
    <main className="page">
      <section className="hero">
        <p className="eyebrow">YOLO26 Resume Project</p>
        <h1>Industrial Defect Segmentation</h1>
        <p>A lightweight demo that reframes MVTec AD into a supervised YOLO26 segmentation workflow.</p>
      </section>

      <section className="panel">
        <h2>Data Reframing</h2>
        <p>Mask annotations are converted into YOLO segmentation labels with a single `defect` class.</p>
      </section>

      <section className="panel">
        <h2>Model Results</h2>
        <p>{metrics ? `${metrics.latency_ms} ms latency` : "Loading metrics..."}</p>
        <ul>
          {examples.map((item) => (
            <li key={item.id}>
              <strong>{item.category}</strong> · {item.status}
            </li>
          ))}
        </ul>
      </section>

      <section className="panel">
        <h2>Live Demo</h2>
        <p>Upload one image and receive an overlay result from the FastAPI backend.</p>
      </section>
    </main>
  );
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm --prefix web test -- --run`

Expected: PASS with the API integration assertions green.

- [ ] **Step 5: Commit**

```bash
git add web/src
git commit -m "feat: connect showcase ui to api data"
```

### Task 8: Add Root Documentation And Local Run Commands

**Files:**
- Create: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing documentation check**

```python
from pathlib import Path


def test_readme_mentions_api_web_and_scripts() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "scripts/" in content
    assert "api/" in content
    assert "web/" in content
    assert "python -m uvicorn api.app.main:app --reload" in content
    assert "npm --prefix web run dev" in content
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_readme.py -q`

Expected: FAIL because `README.md` and `tests/test_readme.py` do not exist yet.

- [ ] **Step 3: Implement the minimal root docs**

`tests/test_readme.py`
```python
from pathlib import Path


def test_readme_mentions_api_web_and_scripts() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "scripts/" in content
    assert "api/" in content
    assert "web/" in content
    assert "python -m uvicorn api.app.main:app --reload" in content
    assert "npm --prefix web run dev" in content
```

`README.md`
```markdown
# YOLO26 MVTec Seg Demo

Resume-oriented industrial defect segmentation project built on YOLO26 and MVTec AD.

## Repository Layout

- `scripts/`: dataset conversion, validation, and gallery export helpers
- `api/`: FastAPI inference and static artifact endpoints
- `web/`: React/Vite showcase UI

## Local API

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r api/requirements.txt
python -m uvicorn api.app.main:app --reload
```

## Local Web

```bash
npm --prefix web install
npm --prefix web run dev
```
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_readme.py -q`

Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_readme.py .gitignore
git commit -m "docs: add root setup guide"
```

## Self-Review Notes

### Spec Coverage

- Data reframing: covered by Task 4 and Task 5.
- Lightweight FastAPI inference API: covered by Task 1, Task 2, and Task 3.
- Single-page resume demo: covered by Task 6 and Task 7.
- Lightweight scope and no heavy platform architecture: preserved by the file structure and the absence of queue/database tasks.
- Project documentation and runnable setup: covered by Task 8.

### Placeholder Scan

No `TODO`, `TBD`, or "implement later" placeholders remain in the tasks. The only deliberate non-final element is the `InferenceService` stub in Task 3, which is explicitly the minimal step needed before wiring a real YOLO26 model in a later red-green cycle.

### Type Consistency

- API contract uses `PredictResponse` consistently in Task 3 and frontend `PredictResponse` in Task 7.
- Metrics and examples payload shapes are defined in Task 2 and consumed consistently in Task 7.
- Paths used by the frontend and API both point at `/artifacts/...`, matching the static JSON artifacts.
