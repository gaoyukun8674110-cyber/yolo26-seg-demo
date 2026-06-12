# Codex 返工单：YOLO26 MVTec 缺陷分割 — 模型质量 / 对比实验 / 延迟实测

> 给 Codex 的执行说明。三个任务相互独立，可分别交付。每个任务都给了：目标、改动文件、
> 接口约定、运行命令、验收标准。**不要改动已通过的现有测试的行为**；新增功能用新文件 + 新测试。

## 项目事实（执行前必读，避免猜测）

- 数据集是**单类分割**：`data/processed/data.yaml` 里 `names: {0: defect}`。
- 训练/验证数据布局：`data/processed/images/{train,val,test}` 与 `data/processed/labels/{train,val,test}`。
  good 样本的 label 文件为空（无缺陷），defect 样本 label 为多边形分割行。
- `data/processed/manifest.json` 结构：`{"summary": {...}, "samples": [ {record}, ... ]}`，
  每条 record 字段：`category`, `defect_type`, `split`, `source_image_path`,
  `source_mask_path`, `output_image_name`, `has_defect`(bool)。
- 文件名规则：`output_image_name` 形如 `bottle__train__good__000.png`，
  对应的 label 是同名 `.txt`。
- 当前指标（baseline，需超越）：mAP50(M)=0.120, mAP50-95(M)=0.048, P(M)=0.207, R(M)=0.162。
- 样本失衡：good 1703 / defect 519（约 3.3:1），是当前指标低的主因之一。
- Python 解释器：`H:\yolo26-mvtec-seg-demo\.venv\Scripts\python`。
- 训练入口：`train26.py`（GPU，ultralytics）。指标导出：`scripts/export_training_metrics.py`。

---

## 任务 1 — 提升模型质量

目标：把 mask mAP50 从 0.12 显著提升（先以 **mAP50(M) ≥ 0.30** 作为本轮验收门槛；达不到也要产出诊断）。

### 1.1 新增：训练集 good 样本下采样

新建 `scripts/rebalance_dataset.py`。

- 作用：在**不动 val/test** 的前提下，对 **train** split 的 good 样本做下采样，
  使 train 的 good:defect 比例 ≈ 由参数 `--good-ratio` 指定（默认 `1.5`，即 good 数 ≈ 1.5×defect 数）。
- 输入：现有 `data/processed`（images/labels/manifest.json）。
- 输出：新目录 `data/processed_balanced/`，结构与 `data/processed` 完全一致
  （images/{train,val,test}, labels/{train,val,test}, data.yaml, manifest.json）。
  **复制而非移动**原始文件；原始 `data/processed` 保持不变。
- 选择逻辑：用固定 `--seed`(默认 0) 随机抽取要保留的 train/good 样本，保证可复现。
- val 与 test 的全部样本原样复制。
- 同时写出 `data/processed_balanced/data.yaml`（path 指向新目录绝对路径，names 不变）
  和更新后的 `manifest.json`（summary 重新统计）。

CLI：
```
python scripts/rebalance_dataset.py \
  --source data/processed \
  --output data/processed_balanced \
  --good-ratio 1.5 \
  --seed 0
```

验收：
- 运行后 `data/processed_balanced/manifest.json` 的 train good:defect 比例落在 `[good-ratio±0.2]` 内。
- val/test 计数与原始一致。
- 新增 `tests/test_rebalance_dataset.py`：构造一个小型临时数据目录，断言比例与 val/test 不变。
- `H:\...\.venv\Scripts\python -m pytest tests -q` 全绿。

### 1.2 修改训练配方 `train26.py`

把 `train26.py` 改成参数化、可复现、干净起步：

- **不要再从自己的 best.pt 续训**。基础权重改为命令行可选，默认从 `yolo26s-seg.pt` 干净起步
  （续训会污染对比实验，见任务 2）。
- 暴露这些参数（argparse，给合理默认值）：
  `--data`(默认 `data/processed_balanced/data.yaml`)、`--model`(默认 `yolo26s-seg.pt`)、
  `--epochs`(默认 150)、`--imgsz`(默认 1024)、`--batch`(默认 -1)、`--name`、`--seed`(默认 0)。
- 训练参数加入：`cos_lr=True`、`seed=<--seed>`、`deterministic=True`、`patience=50`。
- 训练结束后自动调用 `scripts/export_training_metrics.py` 的导出函数（import 复用，不要复制逻辑），
  把新 run 的 `results.csv` 指标写进 `artifacts/metrics/summary.json`，并把 `model_path` 指向新权重。
- 保留 `model.export(format="onnx")` 与 `model.val()`。

验收：
- `python train26.py --help` 能打印全部新参数。
- 训练命令可跑（GPU 环境）：
  `python train26.py --data data/processed_balanced/data.yaml --name yolo26s_seg_mvtec6_balanced_v1`
- 跑完后 `artifacts/metrics/summary.json` 的 `metrics_source` 指向新 run，指标为真实值。
- **不在 CI 跑训练**；只保证脚本可导入、`--help` 可用、参数透传正确（可加一个 dry-run 单测 mock `YOLO`）。

### 1.3 新增：按类别诊断 `scripts/evaluate_per_category.py`

现有 `scripts/export_eval_gallery.py` 是空壳，本任务替代它的诊断职责（不要删旧文件，新增即可）。

- 作用：对给定权重，**分别**在 6 个品类的 val（或 test）子集上跑 `model.val()`，
  输出每类的 P/R/mAP50/mAP50-95(M)，以及总体。
- 子集构造：从 `manifest.json` 按 `category` 过滤 `output_image_name`，
  为每类临时生成一个 `data_{category}.yaml`（只含该类样本的 val 列表）。可用 ultralytics 支持的
  显式文件列表方式，或临时软/硬链接目录；实现方式自选，但**不得污染** `data/processed*`。
- 输出 `artifacts/metrics/per_category.json`：
  ```json
  {
    "weight": "<path>", "split": "val",
    "per_category": {"bottle": {"precision_mask":..,"recall_mask":..,"map50_mask":..,"map50_95_mask":..}, ...},
    "overall": {...}
  }
  ```
- 可选：把每类若干 FP/FN 的 overlay 导到 `artifacts/diagnostics/{category}/`，便于肉眼排查。

CLI：
```
python scripts/evaluate_per_category.py \
  --weight runs/segment/yolo26s_seg_mvtec6_balanced_v1/weights/best.pt \
  --data-root data/processed_balanced \
  --split val
```

验收：
- 生成 `artifacts/metrics/per_category.json`，6 类齐全。
- 新增 `tests/test_evaluate_per_category.py`：mock `YOLO.val` 返回假指标，断言 JSON 结构正确、按类切分正确。

---

## 任务 2 — 对比实验：单跨类模型 vs 按品类 baseline

目标：用同一套 val/test，对比"1 个统一 defect 模型" vs "6 个按品类单训模型"，产出一张可放简历的结论表。

### 2.1 生成按品类的训练配置

在 `scripts/rebalance_dataset.py` 或新建 `scripts/make_per_category_datasets.py` 中，
为每个品类输出独立数据集目录 `data/per_category/{category}/`，结构同标准布局，
只含该品类的 train/val/test 与 `data_{category}.yaml`。同样做 1.1 的 good 下采样。

### 2.2 训练与汇总

- 统一模型：复用任务 1 训练出的 `yolo26s_seg_mvtec6_balanced_v1`。
- 按品类 baseline：对每个 `data/per_category/{category}/data_{category}.yaml`，
  用 `train26.py`（同样配方、同 seed、同 imgsz、epochs 可按子集大小适当缩短）各训一个模型，
  权重落在 `runs/segment/percat_{category}_v1/`。
- 新建 `scripts/compare_unified_vs_percategory.py`：
  - 统一模型用任务 1.3 的 `per_category.json`（每类在统一模型上的表现）。
  - 每个 per-category 模型在**各自品类的同一 val 子集**上 `model.val()`。
  - 输出 `artifacts/metrics/comparison.json` 与一张 markdown 表 `docs/comparison_table.md`：
    列 = 品类；行 = {统一模型 mAP50, 单品类模型 mAP50, 参数量, 单图延迟ms}；末行总体。

CLI：
```
python scripts/compare_unified_vs_percategory.py \
  --unified-weight runs/segment/yolo26s_seg_mvtec6_balanced_v1/weights/best.pt \
  --percat-root runs/segment \
  --data-root data/processed_balanced
```

验收：
- 生成 `artifacts/metrics/comparison.json` 和 `docs/comparison_table.md`，6 类 + 总体齐全。
- 表中每个单元格都有真实数值（延迟可复用任务 3 的测量函数）。
- 新增对应单测（mock 推理），断言表结构与字段。

> 注意：必须先完成任务 1 把统一模型练到像样水平，否则对比无说服力。CI 不跑训练。

---

## 任务 3 — 实测推理延迟并回填 latency_ms（优先级最高、最快交付）

目标：用真实权重在目标机器上测延迟，把 `artifacts/metrics/summary.json` 里占位的 `latency_ms: 0` 换成实测值。

事实：`api/app/services/inference.py` 的 `UltralyticsSegmentationPredictor.predict()`
已经在测真实 `latency_ms`（`perf_counter`，见该文件 ~92 行），只是没人跑过、summary 没回填。

### 实现：新建 `scripts/benchmark_latency.py`

- 复用 `api/app/services/inference.py` 的 `UltralyticsSegmentationPredictor`（import，不要复制推理逻辑）。
- 流程：加载权重 → 用 `--warmup`(默认 5) 张图预热（不计时）→ 对 `--samples`(默认 50) 张
  held-out 图推理并记录每次 `latency_ms` → 输出 p50/p95/mean。
- 取图来源：默认从 `data/processed/images/test` 随机抽（固定 `--seed`）。
- 把 **p50** 写回 `artifacts/metrics/summary.json` 的 `latency_ms`（保留两位小数），
  并新增字段 `latency_p95_ms`、`latency_samples`、`latency_device`(cuda/cpu)。
  写回时**保留 summary 其余字段不变**（读取-合并-写回，复用 export 脚本的写法）。

CLI：
```
set YOLO26_MODEL_PATH=H:\yolo26-mvtec-seg-demo\runs\segment\yolo26s_seg_mvtec6_defect_v1\weights\best.pt
H:\yolo26-mvtec-seg-demo\.venv\Scripts\python scripts\benchmark_latency.py ^
  --weight %YOLO26_MODEL_PATH% --samples 50 --warmup 5
```

验收：
- 运行后 `artifacts/metrics/summary.json` 的 `latency_ms` > 0，且新增 p95/samples/device 字段。
- 新增 `tests/test_benchmark_latency.py`：mock predictor 返回固定 latency 列表，断言
  p50/p95 计算正确、summary 合并写回不丢字段。
- 现有 `api/tests`、`tests` 全绿（`H:\...\.venv\Scripts\python -m pytest api/tests tests -q`）。

---

## 全局验收（交付前自检）

1. `H:\yolo26-mvtec-seg-demo\.venv\Scripts\python -m pytest api/tests tests -q` 全绿。
2. 不破坏现有 API 契约（`/health /metrics /examples /predict /artifacts`）。
3. 新增脚本都有 `--help`，默认参数能在仓库现有数据上跑通（训练类脚本除外，标注"需 GPU"）。
4. 不提交大文件权重/ONNX 到 git（遵守现有 `.gitignore`）。
5. README 增补：新增脚本的用途与命令各一行。

## 建议执行顺序

任务 3（最快有产出）→ 任务 1.1+1.2（重训一版）→ 任务 1.3（按类诊断）→ 任务 2（对比收尾）。
