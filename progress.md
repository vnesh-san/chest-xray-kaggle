# Progress

## TLDR

- **2026-03-11** — Set up local training pipeline: fix notebook paths for local env, upgrade to YOLO26x, resume data download, push to GitHub. `completed`
- **2026-03-11** — Full pipeline implementation: smoke test, consensus labels, 6-channel dual-YOLO + RT-DETR 60-checkpoint ensemble. `completed`
- **2026-03-13** — 24h parallel training pipeline: no-folds full-data mode, 2-GPU parallel launch, rare-class oversampling, cache=ram. `completed`

---

## Current Work

**Original request:** 24-hour full-data parallel training pipeline. Competition deadline March 30. Need trained models by tomorrow. Key insight: instead of DDP (42-day estimate), run two independent single-GPU processes in parallel — GPU 0 trains window set A, GPU 1 trains window set B simultaneously. No folds, train on full 15k images.

**Key decisions:**
- `--no-folds` flag: full-data 95/5 stratified split (14,250 train / 750 val) — no 15-fold CV
- Single-GPU per process (no DDP, no /dev/shm issues, workers=8)
- Rare-class oversampling: repeat-factor sampling (sqrt(median/freq), clamp [1,3]) for Atelectasis (1), Calcification (2), Consolidation (4), ILD (5), Pneumothorax (12)
- `cache="ram"` — 349GB RAM available, dataset ~42GB, big dataloader speedup
- `warmup_epochs=1.0` (from 3.0/2.0), `close_mosaic=5` (from 10), add `mixup=0.1`, `patience=0`
- Ultralytics has no built-in class balancing → added oversampling via symlink duplication
- Oversampling is deterministic (seeded by hash(img_id) & 0xFFFF) for idempotency

**Changes made to `scripts/train_full.py`:**
1. Added `NO_FOLD_DIR` and `RARE_CLASS_IDS` constants
2. Updated `YOLO_TRAIN_KWARGS`: `cache="ram"`, `warmup_epochs=1.0`, `close_mosaic=5`, `mixup=0.1`, `patience=0`
3. Updated `RTDETR_TRAIN_KWARGS`: same hyperparams
4. Added `compute_oversample_ids(img_ids, anns_map)` — repeat-factor oversampling, returns augmented ID list
5. Added `write_no_fold_dataset(all_ids, anns_map, window_set, oversample=True)` — 95/5 split with oversampling symlinks, returns YAML path
6. Added `_run_yolo_no_folds()` and `_run_rtdetr_no_folds()` — single training run, saves to `full/weights/`
7. Updated `stage1()–stage4()` to accept `no_folds=False` param, dispatch to no-fold runners when set
8. Updated `stage_ensemble()` checkpoint gathering to also glob `full/weights/best.pt` + `full/weights/epoch*.pt`
9. Added `--no-folds` CLI flag and `nf` dispatch variable

**New file `scripts/run_24h.sh`:**
- Phase 1: YOLO26x-A (GPU 0) || YOLO26x-B (GPU 1) in parallel via `wait`
- Phase 2: RT-DETR-X-A (GPU 0) || RT-DETR-X-B (GPU 1) in parallel
- Phase 3: WBF ensemble
- Logs to `outputs/train_stage{1..4}.log` and `outputs/train_ensemble.log`
- Usage: `bash scripts/run_24h.sh [--batch 8] [--workers 8]`

**Timeline:**
| Phase | GPU 0 | GPU 1 | Wall Time |
|-------|-------|-------|-----------|
| YOLO 50ep, cache=ram | Win A (15k imgs) | Win B (15k imgs) | ~10-11h |
| RT-DETR 50ep, cache=ram | Win A | Win B | ~10-11h |
| WBF Ensemble | Both | — | ~1h |
| **Total** | | | **~22h** |

**Status:** Implementation complete. Ready to dry-run: `PYTHONPATH=. python scripts/train_full.py --stage 1 --no-folds --device 0 --batch 8`

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

### 2026-03-13 — 24h parallel training pipeline

**Branch:** vignesh-santhalingam/aimo2026/grpo-more-traces
**Request:** 24-hour full-data parallel training pipeline. Deadline March 30 (17 days). Need models by tomorrow.

**Key insight:** DDP (both GPUs on one model) takes ~42 days for 4 stages × 15 folds × 50 epochs. Instead, run two independent single-GPU processes in parallel on the 2 window sets simultaneously. No folds — train on full 15k images.

**Key decisions:**
- `--no-folds` flag for full-data 95/5 stratified split (no 15-fold CV)
- Rare-class oversampling via symlink duplication (Atelectasis, Calcification, Consolidation, ILD, Pneumothorax)
- `cache="ram"` (349GB available, dataset ~42GB)
- `warmup_epochs=1.0`, `close_mosaic=5`, `mixup=0.1`, `patience=0`

**Files changed:**
- `scripts/train_full.py` — all changes described in Current Work above
- `scripts/run_24h.sh` — new orchestration script

**Outcome:** Implementation complete.
