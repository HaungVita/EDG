# Artifact manifest

The repository deliberately excludes private data, features, predictions, and
model weights. The following historical artifacts were used for the verified
TVR run. Local paths are included only as provenance for the reconstruction;
configure portable paths in `configs/paths.env`.

| Purpose | Historical artifact |
| --- | --- |
| SlowFast feature | `/opt/data/private/tvr_feature_release/video_feature/slowfast_data.h5` |
| Recommended VR model | `tvr-video_face_sub-NetVLAD-2024_10_22_14_34_53/{model.ckpt,opt.json}` |
| Auxiliary higher-result VR model | `tvr-video_face_sub-DK-2024_11_04_13_46_11/{model.ckpt,opt.json}` |
| SVMR/moment model | `tvr-video-sub-test_run_face-2024_10_13_19_36_58/{model.ckpt,opt.json}` |
| VCMR first stage | `inference_tvr_val_sigma_40_predictions_VCMR_SVMR_VR.json` |
| VR training annotations | `tvr_train_select100_release.jsonl` |
| Moment training annotations | `tvr_neg_train_select100_release.jsonl` |
| Moment training VR candidates | `inference_tvr_train_2_predictions_VCMR_SVMR_VR.json` |
| TVR training index mapping | `idx2video.json` |

Exact VCMR first-stage SHA-256:

```text
4aa5ab73f5fb6d39fce975f223f018b7209813e3553933d74ef732290291a852
```

This input was produced by the archived NetVLAD top-100 retrieval stage and
the historical Gaussian cross-stage reranking with `sigma=40`; the final
moment stage consumes its first 10 videos per query.

For provenance, the verified final VCMR metrics JSON had SHA-256
`4e56610aa1a4fca02f6a0a44f157bb7cd92b338cc2481c3302bf4d4c5347df13`.
It is not required to run the code because the expected values are stored in
`expected_metrics/tvr.json`.

Training artifact SHA-256 values:

```text
af6457ac5051c34824563f47a4cb89e3e31c36f8a357375cef3e901e19815f98  tvr_train_select100_release.jsonl
e8452d7a05c51b60a5532a78311f6c685fc3ac091c28bfe1001d0a3ab66bb625  tvr_neg_train_select100_release.jsonl
ce1a54b4b85827c8f9d2270fc18521c36cbdef35a3bf075e1d62357f330f5fa9  inference_tvr_train_2_predictions_VCMR_SVMR_VR.json
b84ab3e7394911d40b1a9ee13d47e16d42bc6a63ff9620a0892dee75367f3c35  idx2video.json
```
