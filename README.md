# YOLO26 MVTec Seg Demo

Resume-oriented industrial defect segmentation project built around YOLO26-style segmentation workflows and the local MVTec AD dataset at `H:\YOLO-Train\MVTec AD`.

The project deliberately focuses on three engineering signals:

- reframing a non-YOLO industrial dataset into a supervised segmentation dataset
- packaging the result as a lightweight FastAPI + React demo
- keeping the system much smaller than the earlier YOLO11 detection framework

## Current Status

- The MVTec conversion pipeline has already been run for `bottle`, `cable`, `capsule`, `hazelnut`, `leather`, and `metal_nut`.
- Processed data now lives under `data/processed/`.
- The API exposes `/health`, `/metrics`, `/examples`, `/predict`, and `/artifacts/...`.
- The default prediction flow runs in `demo-mode`, but it now switches to real inference automatically when `YOLO26_MODEL_PATH` is configured.

## Repository Layout

- `scripts/`: dataset conversion, validation, and gallery export helpers
- `api/`: FastAPI endpoints and inference runtime
- `web/`: React/Vite showcase UI
- `artifacts/`: metrics JSON, gallery JSON, example images, generated overlays
- `data/processed/`: converted YOLO segmentation dataset output
- `docs/superpowers/specs/`: approved design spec
- `docs/superpowers/plans/`: implementation plan used for the current build

## Dataset Conversion

The conversion entry point is `scripts\convert_mvtec_to_yolo_seg.py`.

Example:

```powershell
H:\yolo26-mvtec-seg-demo\.venv\Scripts\python scripts\convert_mvtec_to_yolo_seg.py --source-root "H:\YOLO-Train\MVTec AD" --output-root "H:\yolo26-mvtec-seg-demo\data\processed" --categories bottle cable capsule hazelnut leather metal_nut
```

The script writes:

- `data/processed/images/{train,val,test}`
- `data/processed/labels/{train,val,test}`
- `data/processed/data.yaml`
- `data/processed/manifest.json`

Current processed summary:

- total samples: `2222`
- train: `1552`
- val: `335`
- test: `335`
- good samples: `1703`
- defect samples: `519`

## Python API

Set up the local environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r api/requirements.txt
```

Run the API:

```powershell
python -m uvicorn api.app.main:app --reload
```

Enable real inference later with a trained weight file:

```powershell
$env:YOLO26_MODEL_PATH = "H:\yolo26-mvtec-seg-demo\models\exports\best.pt"
pip install ultralytics
python -m uvicorn api.app.main:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /metrics`
- `GET /examples`
- `POST /predict`
- `GET /artifacts/examples/...`
- `GET /artifacts/generated/...`

## Web Showcase

Install the frontend packages:

```powershell
npm --prefix web install
```

Run the dev server:

```powershell
npm --prefix web run dev
```

Run the frontend tests:

```powershell
npm --prefix web test
```

The Vite dev server proxies `/metrics`, `/examples`, `/predict`, and `/artifacts` to `http://localhost:8000`.

## Tests

Run the Python tests:

```powershell
H:\yolo26-mvtec-seg-demo\.venv\Scripts\python -m pytest api/tests tests -q
```

Run the web tests:

```powershell
npm --prefix web test
```

## Attaching A Real Model Later

The current `InferenceService` saves the uploaded image into `artifacts/generated/` and returns that path as the overlay. This keeps the demo usable before training is complete.

To upgrade the project from demo mode to real inference:

1. Train or export a real YOLO26 segmentation model on `data/processed/`.
2. Point `YOLO26_MODEL_PATH` at the exported weight and install `ultralytics`.
3. Update `artifacts/metrics/summary.json` with real measured latency and model metrics.
