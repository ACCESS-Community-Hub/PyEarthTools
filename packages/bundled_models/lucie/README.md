# LUCIE: Lightweight Uncoupled ClImate Emulator

Please note - this is a fork of https://github.com/ISCLPennState/LUCIE which has been adapted included in PyEarthTools for the purposes of maintenance, compatbility and to supply an integrated approach to using the LUCIE model within the PyEarthTools framework.

---

## Paper & Data

- [arXiv Preprint: arxiv.org/abs/2405.16297](https://arxiv.org/abs/2405.16297)
- [Zenodo Archive: zenodo.org/records/15164648](https://zenodo.org/records/15164648)

---

## Overview

LUCIE is a lightweight climate emulator with a backbone of Spherical Fourier Neural Operator (SFNO). This model can be trained with 1 A100 GPU with around 4 hours at most.
This repository prvides the following:
1. A local torch-harmonics (https://github.com/NVIDIA/torch-harmonics) utility file to avoid packaging issue.
2. A pretrained LUCIE checkpoint that is used for the paper.
3. A inference file to replicate the autoregressive inference used for the results in the paper.
4. A training file that trains the model from scratch.
5. The data generator file that precprocesses the regridded ERA5 data.

## Note
Please refer to the zenodo link for the regridded ERA5 data. The link also includes the preprocessed data from the data generator file.
