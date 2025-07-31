This file contains a listing of all code which is included within the repository which is adapted from, extends, or includes code from another open source project. The filenames, sources, copyright and licenses are recorded here.

The file packages/pipeline/operations/xarray/remapping/healpix.py contains code from https://github.com/nathanielcresswellclay/zephyr, released under the MIT license, with copyright attributed to Jonothan Weyn (2018). The file demarkates which sections have been copied with minimal modificatiions.

The file config.py in various modules contains code taken from https://github.com/dask/dask/tree/main/dask, released under the BSD 3-Clause license, with copyright attributed to Anaconda Inc (2014).

The file packages/utils/src/pyearthtools/utils/initialisation/init_parsing.py contains code from https://github.com/Lightning-AI/pytorch-lightning, released under the Apache 2.0 license, with copyright attributed to the Lightning AI team.

The file packages/utils/src/pyearthtools/utils/parsing/init_parsing.py contains code from https://github.com/Lightning-AI/pytorch-lightning, released under the Apache 2.0 license, with copyright attributed to the Lightning AI team.

The file packages/data/src/pyearthtools/data/indexes/utilities/folder_size.py contains code from https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python, released under Creative Commons BY-SA 4.0 (International).

The file packages/data/src/pyearthtools/data/indexes/extensions.py extends and is largely sourced from https://github.com/pydata/xarray/blob/main/xarray/core/extensions.py, released under the Apache 2.0 license.

The package packages/bundled_models/fourcastnext extends and is significantly based on the code from https://github.com/nci/FourCastNeXt which is made available under the Apache 2.0 license. That repository in turn extends the code from https://github.com/NVlabs/FourCastNet/, released under the BSD 3-Clause license. The FourCastNet model is described in detail at https://arxiv.org/abs/2202.11214. The FourCastNeXt model is described in detail at https://arxiv.org/abs/2401.05584, and a version of the FourCastNeXt code is bundled, adapted for compatibility and maintained within the PyEarthTools repository so it can continue to be a useful reference implementation and learning aid.

The package packages/bundled_models/lucie extends and is based on the code from https://github.com/ISCLPennState/LUCIE, which is made available under the MIT license. The LUCIE model is described in detail at https://doi.org/10.48550/arXiv.2405.16297. The version of the model bundled in PyEarthTools may undergo changes associated with package maintenance and compatibility so it can continue to be a useful reference implementation and learning aid. Within that repository, those authors bundle the file "torch_harmonics_local.py", which is based on https://github.com/NVIDIA/torch-harmonics . The bundled file has an Apache 2.0 copyright statement included in it but at the time of writing the NVIDIA repository carries the BSD 3-clause license. Both of these licenses allow bundling to occur and all relevant files preserve the copyright statement within the files. Copyright for the original works go to the LUCIE and torch-harmonics developers respectively.
