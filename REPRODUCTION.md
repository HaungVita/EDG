# Refactor verification

The refactored code was evaluated from the original checkpoints on the full
TVR validation set on 2026-08-28. No cached prediction file was used as model
input, except for the documented sigma-40 first-stage VR input required by the
paper's VCMR cross-stage pipeline.

| Task | Result | Exact target match |
| --- | --- | --- |
| VR R@1/5/10/100 | 29.66 / 55.43 / 64.42 / 91.28 | yes, from the sigma-40 first-stage prediction |
| SVMR IoU 0.5 R@1/5/10/100 | 44.25 / 64.35 / 72.19 / 89.40 | yes |
| SVMR IoU 0.7 R@1/5/10/100 | 23.43 / 44.96 / 54.69 / 76.34 | yes |
| VCMR IoU 0.5 R@1/5/10/100 | 15.74 / 23.71 / 26.31 / 35.06 | yes |
| VCMR IoU 0.7 R@1/5/10/100 | 8.61 / 16.62 / 19.79 / 29.87 | yes |

The recommended VR result above was re-evaluated as part of the complete VCMR
pipeline using the archived sigma-40 first-stage prediction. The refactored
standalone VR path was additionally run from a later DK checkpoint and produced
`30.30 / 55.72 / 66.26 / 91.24`; that auxiliary result is not the paper's
recommended efficiency/accuracy configuration. The archived high-compute
variant produced `30.22 / 55.67 / 65.14 / 91.23`.

Structural cleanup reduced the publication package from 523 files and about
12 MB to the focused EDG implementation. Removed material consisted of
obsolete training variants, visualizers, ad-hoc tests, logs, caches, obsolete model
variants, and duplicated dependency trees.

## Training-path verification

On 2026-08-28 both restored training stages passed two levels of validation:

1. `--n_epoch 0` loaded all training artifacts and built the Dataset, model,
   CUDA state, and BertAdam optimizer.
2. `--debug --n_epoch 1 --data_ratio 0.0001` completed a real batch through
   forward inference, loss computation, backward propagation, and an optimizer
   update.

The Event-Driven Hybrid video-retrieval model contains 50,272,101 trainable
parameters. The Cross-Stage Guide moment model contains 56,116,996 trainable
parameters. The historical moment-training candidate JSON is approximately
4.3 GB and the loader consumed roughly 23 GB of host RAM while parsing it;
users should provision at least 32 GB of host memory for this path.
