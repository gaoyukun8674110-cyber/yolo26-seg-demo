# YOLO26 MVTec Seg Demo Design

Date: 2026-04-25
Status: Approved in chat, written for review

## Summary

This project is a resume-oriented industrial vision demo built around YOLO26 segmentation.
The core value is not "train another YOLO model", but "reframe a non-YOLO industrial anomaly dataset into a supervised defect-localization project and package the result as a clear, reviewable engineering artifact".

The dataset source is `H:\YOLO-Train\MVTec AD`.
The project root is `H:\yolo26-mvtec-seg-demo`.

## Why This Project Exists

The previous YOLO11 work on `H:\yolo11-detection-framework` already covers a heavier detection platform pattern:

- standard YOLO dataset structure
- multi-service frontend/backend architecture
- upload, queue, async worker, storage, and result viewing

This YOLO26 project must therefore differentiate on the parts that create stronger resume signal:

- non-standard dataset conversion
- industrial defect segmentation instead of generic detection
- leaner system design with clearer result presentation
- explicit focus on model evidence rather than platform complexity

## Goals

1. Convert MVTec AD subsets into a supervised YOLO26 segmentation dataset.
2. Train a first-pass industrial defect segmentation model using a single `defect` class.
3. Produce evaluation artifacts that are easy to discuss in interviews.
4. Build a lightweight web demo for single-image inference and result presentation.
5. Keep the implementation intentionally simpler than the prior YOLO11 platform.

## Non-Goals

1. No multi-user account system.
2. No task queue, async worker fleet, or database-first architecture.
3. No batch job orchestration in the first version.
4. No attempt to preserve the original MVTec anomaly-detection protocol for training.
5. No multi-class defect taxonomy in v1 unless the single-class pipeline is already stable.

## Dataset Constraints And Decision

Observed local dataset structure:

- category-level folders such as `bottle`, `cable`, `capsule`, `hazelnut`, `leather`, `metal_nut`
- `train\good`
- `test\<defect_type>` and `test\good`
- `ground_truth\<defect_type>\*_mask.png`

This means the local data is not directly usable as a standard YOLO segmentation dataset.
The project will therefore treat the dataset conversion step as a first-class engineering deliverable.

## Scope Of V1

V1 uses a subset-first strategy:

- initial categories: `bottle`, `capsule`, `metal_nut`
- task type: binary segmentation with one class named `defect`
- inference unit: single image
- deployment target: local demo on desktop, with future export path to ONNX or OpenVINO

The initial category set is chosen because the defect regions are visually easier to explain in a resume demo and the scope remains manageable.

## Data Pipeline Design

### Source Inputs

- source images from `H:\YOLO-Train\MVTec AD\<category>\train\good`
- source test images from `H:\YOLO-Train\MVTec AD\<category>\test\...`
- source masks from `H:\YOLO-Train\MVTec AD\<category>\ground_truth\...`

### Converted Dataset Output

The converted dataset will live under the project root, for example:

```text
data/
  raw_links/
  processed/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
    data.yaml
```

### Label Strategy

- all anomalous samples become class `0: defect`
- normal images receive empty segmentation labels
- mask images are converted to polygons in YOLO segmentation text format

### Split Strategy

The project will create a new supervised split rather than reuse the original anomaly-detection benchmark split.

Rules:

- stratify by category and defect type where possible
- keep a held-out test set for final screenshots and metrics
- keep a validation set for training control
- document exactly how many images from each category and defect type are used

### Data Quality Checks

Before training, the conversion pipeline must verify:

- image-label one-to-one mapping
- empty labels only on normal samples
- masks converted without zero-area polygons
- small defects are not silently dropped by contour filtering

## Model Design

### Task

- model family: YOLO26 segmentation
- first model target: `yolo26s-seg`
- class count: 1

### Training Principles

- prioritize a stable baseline over large model size
- preserve high enough image resolution for small defects
- compare full-image training against patching only if small-defect recall is poor

### Evaluation

V1 should report more than just mAP.
Primary metrics:

- segmentation mAP
- IoU or Dice on held-out examples
- defect recall
- per-image inference latency

The demo should also include curated qualitative examples:

- clear success cases
- borderline cases
- obvious failure cases

## Frontend Design

The frontend is a single-page resume-facing demo, not an operations console.

### Sections

1. Hero
   - one-sentence project positioning
   - key numbers such as categories used, task type, and latency

2. Data Reframing
   - explain why MVTec AD is not native YOLO data
   - show the conversion pipeline from masks to YOLO26 labels

3. Model Results
   - metrics cards
   - side-by-side examples of original image, ground truth, prediction, and overlay

4. Live Demo
   - upload one image
   - display returned overlay and summary text

5. Technical Highlights
   - data conversion
   - segmentation training
   - inference API
   - lightweight deployment path

### Frontend Stack

- React with Vite, or Next.js only if server-side routing becomes useful
- default recommendation: `React + Vite`

Reason:

- simpler local setup
- faster for a single-page showcase
- no need to mirror the heavier YOLO11 framework architecture

## Backend Design

The backend is intentionally small and centered around inference.

### API Endpoints

- `POST /predict`
  - input: image file
  - output: defect presence summary, overlay asset, confidence summary, latency

- `GET /metrics`
  - output: precomputed evaluation summary for frontend cards

- `GET /examples`
  - output: curated example metadata for the gallery

- `GET /health`
  - output: service health status

### Backend Stack

- FastAPI
- local file storage for generated overlays and curated example assets
- no database in v1

### Internal Modules

- `app/api`
- `app/core`
- `app/services/inference.py`
- `app/services/visualization.py`
- `app/schemas`
- `artifacts/examples`
- `artifacts/metrics`

## Suggested Repository Layout

```text
docs/
  superpowers/
    specs/
data/
  raw_links/
  processed/
scripts/
  convert_mvtec_to_yolo_seg.py
  validate_dataset.py
  export_eval_gallery.py
models/
  checkpoints/
  exports/
artifacts/
  examples/
  metrics/
api/
web/
README.md
```

## Delivery Phases

### Phase 1: Data Reframing

- choose categories for v1
- write conversion script
- validate converted labels
- produce dataset manifest

### Phase 2: Model Baseline

- train first YOLO26 segmentation baseline
- inspect failures
- adjust resolution or preprocessing if needed

### Phase 3: Result Packaging

- export overlay images
- prepare metrics JSON
- prepare example gallery data

### Phase 4: Demo Shell

- build single-page frontend
- expose simple inference API
- connect upload flow and result display

## Risks And Mitigations

### Risk: Small defects are lost after mask-to-polygon conversion

Mitigation:

- inspect converted polygons visually
- tune contour extraction and minimum area thresholds
- fall back to patch-based preprocessing if needed

### Risk: Model quality looks weak when mixing categories too early

Mitigation:

- start with three categories
- keep defect class binary in v1
- expand only after stable baseline evidence

### Risk: Frontend work dilutes the CV signal

Mitigation:

- keep the frontend single-page
- optimize for result clarity, not platform breadth

## Acceptance Criteria For V1 Design

The design is considered ready for implementation when:

1. the data conversion specification is fixed
2. the v1 category set is fixed
3. the API surface is limited to lightweight demo endpoints
4. the frontend is explicitly a showcase page, not a full platform
5. the implementation plan can be split cleanly into data, model, API, and web tasks
