# IVIF-Reproduction-and-Improvement

## P²Fusion weather adaptation on AWMM

The `P2Fusion-AWMM/` directory contains the uploaded P²Fusion source and paired clean-reference weather adaptation implementation.

- [Server setup and training guide (中文)](P2Fusion-AWMM/README_AWMM.md)
- [Default configuration](P2Fusion-AWMM/configs/awmm.json)
- [CPU validation report](P2Fusion-AWMM/VALIDATION_REPORT.json)

The new entry points are `train_awmm.py` and `infer_awmm.py`. Original source files are preserved. Dataset and pretrained weights are not included.

Validation: 33 CPU checks passed, including training, checkpoint resume and adverse-only inference with synthetic data/test teachers. Real AWMM, pretrained teachers and A40 CUDA execution still require server validation.
