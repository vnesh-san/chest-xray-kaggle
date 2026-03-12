# Progress

## TLDR

- **2026-03-11** — Set up local training pipeline: fix notebook paths for local env, upgrade to YOLO26x, resume data download, push to GitHub. `completed`
- **2026-03-11** — Full pipeline implementation: smoke test, consensus labels, 6-channel dual-YOLO + RT-DETR 60-checkpoint ensemble. `completed`

---

## Current Work

**Original request:** "RT-DETR on both images like yolo" — add RT-DETR-X training on both window sets (A and B), giving a 60-checkpoint ensemble: YOLO26x-A×15 + YOLO26x-B×15 + RT-DETR-X-A×15 + RT-DETR-X-B×15.

**Questions / Decisions:**
- RT-DETR v2 not available in ultralytics (only v1 `rtdetr-x.pt`); RT-DETRv2 is HuggingFace-only. Decision: stay in ultralytics for uniform `.train()/.predict()` API.
- Stage 3 = RT-DETR-X on window set A (lung/mediastinum/soft-tissue)
- Stage 4 = RT-DETR-X on window set B (bone/pleural/vascular)

**Changes made to `scripts/train_full.py`:**
1. `_run_rtdetr_folds()` now accepts `model_name` as first arg (was hardcoded to `STAGE3_MODEL`)
2. `stage3()` → RT-DETR-X on window set A (`stage3_rtdetr_A`)
3. `stage4()` → RT-DETR-X on window set B (`stage4_rtdetr_B`) — new function
4. `stage_ensemble()` `stage_configs` now has all 4 entries (was missing stage3_rtdetr_A and stage4_rtdetr_B)
5. CLI `--stage` choices now include `"4"`
6. `--stage all` now calls `stage_convert("both")`, stage1–4, then ensemble

**Final ensemble config (60 checkpoints):**
| Stage | Model | Window Set | Folds |
|-------|-------|------------|-------|
| 1 | YOLO26x | A (lung/med/soft) | 15 |
| 2 | YOLO26x | B (bone/pleural/vasc) | 15 |
| 3 | RT-DETR-X | A (lung/med/soft) | 15 |
| 4 | RT-DETR-X | B (bone/pleural/vasc) | 15 |

**Status:** Implementation complete.

---

## Previous Work

**Original request:** Understand competition, download 200GB data, set up smoke test + full training pipeline.

**Key decisions:**
- YOLO26x (`end2end=True`, `optimizer=MuSGD`) — NMS-free, faster inference
- 6-channel dual-window strategy: Window Set A (lung/mediastinum/soft-tissue) + Window Set B (bone/pleural/vascular) — no architecture changes, two separate 3-ch RGB models, WBF ensemble
- Consensus labels: WBF-merge radiologist boxes with `min_votes=2, iou_thr=0.4` — biggest single gain expected (+8-15 mAP)
- Multilabel-stratified 15-fold CV (`iterstrat`) — all 14 classes balanced across folds
- 1024px resolution, `hsv_s=0.2` (safe for HU encoding), `flipud=0.0` (preserves chest anatomy)
- `build_dims_cache()`: header-only parallel DICOM reads for image dimensions (15k in ~14s, cached to JSON)

**Files created/updated:**
- `src/data/dicom_utils.py` — added WINDOW_SET_A/B/SETS, `pleural`/`vascular` presets
- `src/data/label_consensus.py` — new: WBF consensus builder with chunked multiprocessing + dims cache
- `scripts/prepare_labels.py` — new: one-time label generation CLI
- `scripts/smoke_test.py` — consensus labels, `mosaic=0.0`, `hsv_s=0.2`
- `scripts/train_full.py` — full pipeline: 4-stage 60-checkpoint ensemble

**Smoke test:** Passed (PID 127109, 500 images, 1 epoch, mAP=0 expected — pipeline validated).

---

## Full History

### 2026-03-11 — Initial setup

**Branch:** main
**Request:** Fix notebook for local env, upgrade to YOLO26x, resume 142GB data download, push to GitHub.

**Outcome:**
- Notebook paths fixed (`/kaggle/input/` → `data/raw/...`, `/kaggle/working` → `outputs/`)
- `MODEL_NAME = 'yolo26x.pt'`, `optimizer='MuSGD'`, `end2end=True`
- Data fully downloaded and unzipped: 15,000 train + 3,000 test DICOMs
- GitHub repo created: https://github.com/vnesh-san/chest-xray-kaggle
- Full pipeline scaffolding created and pushed

### 2026-03-11 to 2026-03-12 — Full pipeline implementation

**Branch:** main
**Request:** Best data strategy for 15k images + dual-window YOLO + RT-DETR ensemble.

**Key work:**
1. Data strategy research: consensus labels > 1024px > multilabel CV > hsv_s fix > copy-paste
2. WBF label consensus (`build_consensus_labels`, `build_dims_cache`) with chunked multiprocessing
3. 6-channel dual-model insight: split 6 windows into two 3-ch sets, train separate models each with full pretrained weights
4. RT-DETR confirmed ultralytics v1 only; staying in ultralytics for uniform API
5. `smoke_test.py` mosaic=0.0 fix for end2end=True buffer-empty crash
6. Smoke test validated successfully
7. `train_full.py` 4-stage 60-checkpoint pipeline (YOLO26x × 2 + RT-DETR-X × 2, window sets A+B, 15 folds each)

**Outcome:** All files committed and pushed. Ready to begin Stage 1 training.
