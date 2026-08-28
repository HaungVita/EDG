# Third-party components

This directory contains the prior open-source model component that the
historical EDG implementation imports:

- `actionformer/`: temporal convolution/transformer backbone layers adapted
  from ActionFormer and used inside the EDG encoders.

This is not an additional stage of EDG. It is an implementation dependency
retained for checkpoint compatibility. See
`../THIRD_PARTY_LICENSES/` and `../NOTICE.md` for attribution.
