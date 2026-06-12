# YOLO26 MVTec Seg Demo

Resume-oriented industrial defect segmentation project built around YOLO26-style segmentation workflows and the local MVTec AD dataset at `H:\YOLO-Train\MVTec AD`.

The project deliberately focuses on three engineering signals:

- reframing a non-YOLO industrial dataset into a supervised segmentation dataset
- packaging the result as a lightweight FastAPI + React demo
- keeping the system much smaller than the earlier YOLO11 detection framework

## Current Status

- The MVTec conversion pipeline has already been run for `bottle`, `cable`, `capsule`, `hazelnut`, `leather`, and `metal_nut`.
- Processed data now lives under `data/processed/`.
- A YOLO26-style segmentation training run has been exported from `runs/segment/yolo26s_seg_mvtec6_defect_v1/results.csv`.
- `artifacts/metrics/summary.json` now reports real training metrics instead of only `demo-mode` placeholders.
- The API exposes `/health`, `/metrics`, `/examples`, `/predict`, and `/artifacts/...`.
- The API prediction flow still supports a no-model fallback for demos, and switches to real inference automatically when `YOLO26_MODEL_PATH` is configured.

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

## Training Metrics

The reported run is the unified single-class `defect` model trained on the rebalanced dataset (`data/processed_balanced`, train good:defect downsampled to ~1.5:1). Metrics are mask (M) results on the held-out val split.

- model: `yolo26s-seg` (unified cross-category `defect`)
- model status: `trained-evaluated`
- model path: `runs/segment/yolo26s_seg_mvtec6_balanced_v1/weights/best.pt`
- training epochs: `150`
- best epoch by mask mAP50: `116`
- precision(M): `0.47091`
- recall(M): `0.27027`
- mAP50(M): `0.26476`
- mAP50-95(M): `0.12006`
- inference latency: `24.44 ms` p50 / `27.90 ms` p95 (RTX 3070 Ti Laptop, CUDA, 50 samples)

### Unified vs per-category baseline

`docs/comparison_table.md` compares the single unified model against six per-category models on the same val split (mask mAP50):

| Category | Unified | Per-category |
| --- | ---: | ---: |
| bottle | 0.294 | 0.623 |
| cable | 0.302 | 0.465 |
| capsule | 0.717 | 0.590 |
| hazelnut | 0.201 | 0.335 |
| leather | 0.820 | 0.839 |
| metal_nut | 0.183 | 0.815 |
| **Overall** | **0.420** | **0.611** |

The per-category models reach higher overall accuracy (0.611 vs 0.420), but the unified model uses a single weight / one GPU footprint and even wins on `capsule` — a deliberate accuracy-vs-operational-cost trade-off that motivates the unified task reframing.

Regenerate these metrics after a new training run:

```powershell
H:\yolo26-mvtec-seg-demo\.venv\Scripts\python scripts\export_training_metrics.py
```

Model-quality and experiment utilities added for the rework:

```powershell
H:\yolo26-mvtec-seg-demo\.venv\Scripts\python scripts\rebalance_dataset.py --source data\processed --output data\processed_balanced --good-ratio 1.5 --seed 0
H:\yolo26-mvtec-seg-demo\.venv\Scripts\python train26.py --data data\processed_balanced\data.yaml --name yolo26s_seg_mvtec6_balanced_v1
H:\yolo26-mvtec-seg-demo\.venv\Scripts\python scripts\evaluate_per_category.py --weight runs\segment\yolo26s_seg_mvtec6_balanced_v1\weights\best.pt --data-root data\processed_balanced --split val
H:\yolo26-mvtec-seg-demo\.venv\Scripts\python scripts\make_per_category_datasets.py --source data\processed --output-root data\per_category --good-ratio 1.5 --seed 0
H:\yolo26-mvtec-seg-demo\.venv\Scripts\python scripts\compare_unified_vs_percategory.py --unified-weight runs\segment\yolo26s_seg_mvtec6_balanced_v1\weights\best.pt --percat-root runs\segment --data-root data\processed_balanced
```

`train26.py`, `evaluate_per_category.py`, and `compare_unified_vs_percategory.py` require trained YOLO weights and an environment with `ultralytics`; CI covers their importable/mockable logic rather than running GPU training.

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

## Real Inference

When no model path is configured, `InferenceService` saves the uploaded image into `artifacts/generated/` and returns that path as the overlay. This keeps the API and web demo usable in environments without the model file.

To run real inference with the trained weight:

1. Point `YOLO26_MODEL_PATH` at `runs/segment/yolo26s_seg_mvtec6_defect_v1/weights/best.pt` or another exported weight.
2. Install `ultralytics`, `opencv-python-headless`, and `numpy` if they are not already present.
3. Run the API and test `POST /predict` with held-out defect and good samples.
4. Measure and write latency into `artifacts/metrics/summary.json`:

```powershell
H:\yolo26-mvtec-seg-demo\.venv\Scripts\python scripts\benchmark_latency.py --weight runs\segment\yolo26s_seg_mvtec6_defect_v1\weights\best.pt --samples 50 --warmup 5
```
