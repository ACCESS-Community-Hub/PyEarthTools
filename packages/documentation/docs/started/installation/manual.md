# Manual

Steps for a manual install of `EDIT`

## Required Dependencies

### Data

- Python (3.10 or later)
- [xarray](https://docs.xarray.dev/)
- [netCDF4]()

#### Optional Dependencies

- [geopandas]()

### Training

- [edit.data](https://git.nci.org.au/bom/dset/edit-package/data)
- [edit.utils](https://git.nci.org.au/bom/dset/edit-package/utils)
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://lightning.ai/)
- [scipy]()
- [scikit-learn]()

## Instructions


edit is currently published to a package registry on NCI git, so using the below pip install, it can be installed.

```bash
pip install edit-SUBMODULE --index-url https://git.nci.org.au/api/v4/projects/1664/packages/pypi/simple

```
