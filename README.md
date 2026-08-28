# EDG reproduction on TVR

This repository packages the refactored historical implementation used to reproduce the
TVR results of **Event-driven Hybrid and Cross-stage Guide for Video Corpus
Moment Retrieval (EDG)**. Its public structure and primary model classes follow
the paper terminology: `EventDrivenHybrid`, `CrossStageGuide`, Video Retrieval
(VR), Single Video Moment Retrieval (SVMR), and Video Corpus Moment Retrieval
(VCMR). Separate task paths preserve the exact historical checkpoint behavior.

## Reproduced results

All numbers are recall in percent on the TVR validation split.

| Task | IoU | R@1 | R@5 | R@10 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VR | - | 30.30 | 55.72 | 66.26 | 91.24 |
| SVMR | 0.5 | 44.25 | 64.35 | 72.19 | 89.40 |
| SVMR | 0.7 | 23.43 | 44.96 | 54.69 | 76.34 |
| VCMR | 0.5 | 15.74 | 23.71 | 26.31 | 35.06 |
| VCMR | 0.7 | 8.61 | 16.62 | 19.79 | 29.87 |

The machine-readable reference is in `expected_metrics/tvr.json`.

## Repository layout

```text
edg/event_driven_hybrid/                              Event-Driven Hybrid VR stage
edg/cross_stage_guide/single_video_moment_retrieval/ Cross-Stage Guide SVMR path
edg/cross_stage_guide/video_corpus_moment_retrieval/ Cross-Stage Guide VCMR path
edg/data/                                            TVR proposal helpers
edg/evaluation/                                      TVR metric implementation
edg/utils/                                           shared EDG utilities
third_party/                                         attributed compatibility code
scripts/                                             train/evaluate/verify commands
configs/                                             local path template
expected_metrics/                                    paper-result reference
```

The ActionFormer component under `third_party/` is not an additional EDG stage;
it is the minimal temporal backbone code required for historical checkpoint
compatibility. TVR proposal, NMS, and metric logic lives under `edg/data` and
`edg/evaluation`. Visualizers, abandoned model variants, duplicate SQNet trees,
caches, and experiment logs were removed.

## Required artifacts

Large datasets, features, and checkpoints are not committed. Prepare:

- the TVR validation JSONL, query BERT features, subtitle BERT features,
  subtitle metadata, and video-duration index;
- TVR SlowFast features (`slowfast_data.h5`);
- face and portrait features for standalone VR;
- the archived VR checkpoint directory;
- the archived moment checkpoint directory (used for SVMR and VCMR);
- the NetVLAD first-stage prediction reranked with `sigma=40` for VCMR.

Each model directory must contain its matching `model.ckpt` and `opt.json`.
Use a writable copy for `VCMR_MODEL_DIR`, because the historical inference
program writes predictions next to the checkpoint. Artifact provenance and the
checksum of the exact VCMR first-stage input are recorded in
`ARTIFACTS.md`.

## Environment

The successful reconstruction used Python 3.10, PyTorch 2.2.1 and CUDA. To
create a comparable environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/paths.env.example configs/paths.env
```

Edit `configs/paths.env`, then check all paths before inference:

```bash
python scripts/check_environment.py
```

You may keep the configuration elsewhere by setting
`EDG_PATHS_FILE=/absolute/path/to/paths.env`.

## Evaluation

Run each task from any working directory:

```bash
./scripts/run_vr.sh
./scripts/run_svmr.sh
./scripts/run_vcmr.sh
```

The historical programs place generated JSON files in the corresponding model
directory. Compare a generated metrics file with the exact target:

```bash
python scripts/verify_metrics.py /path/to/VR_metrics.json --task VR
python scripts/verify_metrics.py /path/to/SVMR_metrics.json --task SVMR
python scripts/verify_metrics.py /path/to/VCMR_metrics.json --task VCMR
```

## Training

The release includes the two checkpoint-producing training stages used by
EDG. Configure the training paths in `configs/paths.env`, then run:

```bash
./scripts/train_video_retrieval.sh
./scripts/train_single_video_moment_retrieval.sh
```

The first command trains the Event-Driven Hybrid video-retrieval model. The
second trains the Cross-Stage Guide single-video moment model from first-stage
VR candidates on the TVR training split. Its required `TRAIN_VR_INPUT` is an
explicit artifact because candidate mining is part of the two-stage training
protocol. The historical candidate file is about 4.3 GB and its in-memory JSON
loader requires approximately 23 GB of host RAM. Both scripts reproduce the
checkpoint options: 100 epochs, seed
2018, learning rate `1e-4`, weight decay `0.01`, hard negatives from epoch 20,
VR batch size 128, and moment batch size 16.

Additional CLI arguments are appended to the command, so a startup smoke test
can be run without an optimization epoch:

```bash
./scripts/train_video_retrieval.sh --n_epoch 0
./scripts/train_single_video_moment_retrieval.sh --n_epoch 0
```

Use `--debug --n_epoch 1 --data_ratio 0.0001` to execute a small real
forward/backward/optimizer smoke test without running validation.

For a detached long-running experiment, redirect the command to a log and use
the status helper to inspect the process, GPU usage, and newest progress lines:

```bash
./scripts/training_status.sh
```

VCMR uses the trained moment model with the sigma-40 first-stage rankings; it
does not have a third independently trained checkpoint.

## Historical compatibility details

- Temporal decoding must use `min_pred_l=1` and `max_pred_l=24`.
- The Event-Driven Hybrid path includes the guard needed for VR-only inference.
- VCMR repeats each query over its actual number of candidate videos (top 10),
  rather than the obsolete fixed single-video SVMR behavior.
- Final VCMR consumes the sigma-40 NetVLAD result directly and performs only
  the moment stage; applying the cross-stage rerank a second time changes the
  metric.

The VCMR first stage is therefore an explicit input, not silently regenerated
by a different revision of the research code. This is important: the raw
NetVLAD top-100 JSON alone does not contain enough scores to reconstruct the
historical sigma-40 reranking.

## Publication note

This is a reconstructed reproducibility package assembled from historical
experiments. Before public redistribution, verify the licenses and
publication permissions for the original EDG code, TVR annotations/features,
pretrained checkpoints, and third-party components. See `NOTICE.md`.
