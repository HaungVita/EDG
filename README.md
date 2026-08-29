# EDG: Event-Driven Hybrid and Cross-Stage Guide

Official implementation of **Event-Driven Hybrid and Cross-Stage Guide for
Video Corpus Moment Retrieval**.

EDG uses two learned stages for three TVR tasks:

1. **Event-Driven Hybrid** retrieves videos for Video Retrieval (VR).
2. **Cross-Stage Guide** localizes moments for Single Video Moment Retrieval
   (SVMR) and Video Corpus Moment Retrieval (VCMR).

This repository contains both training and inference code. Dataset features,
trained checkpoints, and large intermediate predictions are distributed
separately.

## Results on TVR

Recall is reported on the TVR validation split.

| Task | IoU | R@1 | R@5 | R@10 | R@100 |
|:--|:--:|--:|--:|--:|--:|
| VR | — | 29.66 | 55.43 | 64.42 | 91.28 |
| SVMR | 0.5 | 44.25 | 64.35 | 72.19 | 89.40 |
| SVMR | 0.7 | 23.43 | 44.96 | 54.69 | 76.34 |
| VCMR | 0.5 | 15.74 | 23.71 | 26.31 | 35.06 |
| VCMR | 0.7 | 8.61 | 16.62 | 19.79 | 29.87 |

The machine-readable reference is
[`expected_metrics/tvr.json`](expected_metrics/tvr.json). These values were
reproduced by full inference from the paper checkpoints after refactoring.

The table reports the paper's recommended efficiency/accuracy configuration.
The higher-compute VR variant reaches `30.22 / 55.67 / 65.14 / 91.23`.
A later historical experiment reached R@1 30.30, but it is not used as the
paper's primary EDG result.

## Installation

The reconstructed environment uses Python 3.10, PyTorch 2.2.1, and CUDA.

```bash
git clone https://github.com/HaungVita/EDG.git
cd EDG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/paths.env.example configs/paths.env
```

Edit `configs/paths.env`, then validate the required artifacts:

```bash
python scripts/check_environment.py
```

The local configuration is ignored by Git. Set `EDG_PATHS_FILE` to use a path
configuration stored elsewhere.

## Data and checkpoints

The following artifacts are not stored in this repository:

- TVR train/validation annotations;
- query and subtitle BERT features;
- SlowFast video features (`slowfast_data.h5`);
- face and portrait features used by VR;
- Event-Driven Hybrid VR checkpoint;
- Cross-Stage Guide moment checkpoint;
- first-stage VR predictions used by VCMR.

Each checkpoint directory must contain `model.ckpt` and its matching
`opt.json`. Provenance and known SHA-256 values are documented in
[`ARTIFACTS.md`](ARTIFACTS.md).

Download the paper checkpoints from the
[`v1.0-paper-checkpoints` release](https://github.com/HaungVita/EDG/releases/tag/v1.0-paper-checkpoints):

```bash
wget https://github.com/HaungVita/EDG/releases/download/v1.0-paper-checkpoints/edg_tvr_event_driven_hybrid_vr.tar.gz
wget https://github.com/HaungVita/EDG/releases/download/v1.0-paper-checkpoints/edg_tvr_cross_stage_guide_moment.tar.gz
wget https://github.com/HaungVita/EDG/releases/download/v1.0-paper-checkpoints/SHA256SUMS

sha256sum -c SHA256SUMS
tar -xzf edg_tvr_event_driven_hybrid_vr.tar.gz
tar -xzf edg_tvr_cross_stage_guide_moment.tar.gz
```

Point `VR_MODEL_DIR` at `event_driven_hybrid_vr/`. Point both
`SVMR_MODEL_DIR` and a writable `VCMR_MODEL_DIR` copy at
`cross_stage_guide_moment/`.

## Inference

```bash
./scripts/run_vr.sh       # Video Retrieval
./scripts/run_svmr.sh     # Single Video Moment Retrieval
./scripts/run_vcmr.sh     # Video Corpus Moment Retrieval
```

Verify generated metrics against the reference:

```bash
python scripts/verify_metrics.py /path/to/vr_metrics.json --task VR
python scripts/verify_metrics.py /path/to/svmr_metrics.json --task SVMR
python scripts/verify_metrics.py /path/to/vcmr_metrics.json --task VCMR
```

VCMR uses the Cross-Stage Guide moment checkpoint and first-stage VR
predictions. It does not require a third independently trained model.

## Training

EDG has two checkpoint-producing training stages:

```bash
# 1. Event-Driven Hybrid video retrieval
./scripts/train_video_retrieval.sh

# 2. Cross-Stage Guide moment retrieval
./scripts/train_single_video_moment_retrieval.sh
```

The second stage consumes first-stage training predictions through
`TRAIN_VR_INPUT`. The historical JSON is approximately 4.3 GB and requires
about 23 GB of host memory when loaded; at least 32 GB RAM is recommended.

The paper configuration uses 100 epochs, seed 2018, learning rate `1e-4`,
weight decay `0.01`, and hard negatives from epoch 20. VR and moment batch
sizes are 128 and 16.

Startup and minimal optimization checks:

```bash
./scripts/train_video_retrieval.sh --n_epoch 0
./scripts/train_single_video_moment_retrieval.sh --n_epoch 0

./scripts/train_video_retrieval.sh --debug --n_epoch 1 --data_ratio 0.0001
./scripts/train_single_video_moment_retrieval.sh --debug --n_epoch 1 --data_ratio 0.0001
```

Both training paths have passed forward, loss, backward, and optimizer-update
checks. A complete 100-epoch from-scratch reproduction is still in progress;
the table above comes from full inference using the paper checkpoints. See
[`REPRODUCTION.md`](REPRODUCTION.md) for the exact validation status.

## Code structure

```text
edg/
├── event_driven_hybrid/                 # VR model, training, inference
├── cross_stage_guide/
│   ├── single_video_moment_retrieval/   # SVMR model, training, inference
│   └── video_corpus_moment_retrieval/   # VCMR inference
├── data/                                # temporal proposal helpers
├── evaluation/                          # TVR metrics and post-processing
└── utils/                               # shared utilities

scripts/                                 # public train/evaluate commands
configs/                                 # local path template
expected_metrics/                        # metric reference
third_party/actionformer/                # temporal backbone dependency
```

SVMR and VCMR intentionally retain separate Cross-Stage Guide implementations
because their paper checkpoints depend on different historical inference
behavior. `third_party/actionformer` is an internal backbone dependency, not
an additional EDG stage.

## Exact reproduction notes

- Temporal decoding uses `min_pred_l=1` and `max_pred_l=24`.
- VCMR repeats each query over its actual top-10 candidate videos.
- Final VCMR consumes the sigma-40 first-stage ranking directly.
- Cross-stage reranking must not be applied twice.

The raw NetVLAD top-100 output does not retain enough scores to reconstruct the
historical sigma-40 ranking exactly. The matching first-stage prediction is
therefore treated as a versioned artifact.

## License and attribution

Before redistributing TVR annotations, features, checkpoints, or third-party
components, verify their licenses. See [`NOTICE.md`](NOTICE.md) and
[`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES/).
