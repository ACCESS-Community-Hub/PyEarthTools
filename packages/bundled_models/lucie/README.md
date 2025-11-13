# LUCIE: Lightweight Uncoupled ClImate Emulator

Please note - this is a adaptation of https://github.com/ISCLPennState/LUCIE which has been modified for inclusion in PyEarthTools for the purposes of maintenance, compatbility and to supply an integrated approach to using the LUCIE model within the PyEarthTools framework.

This code was copied from the LUCIE repository from commit hash 19a1d6ebe844f49893f92e8b377ebdca8f6aa0e6 (Jul 9th, 2025).

---

## Paper & Data

These are the links for the original paper, code and data published by the LUCIE authors. The code was published to Zenodo under a Creative Commons license but the license in their github repository was MIT to allow improved code re-use.

- [arXiv Preprint: https://doi.org/10.48550/arXiv.2405.16297](https://doi.org/10.48550/arXiv.2405.16297)
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
Please refer to the LUCIE zenodo link for the regridded ERA5 data. The link also includes the preprocessed data from the data generator file.
