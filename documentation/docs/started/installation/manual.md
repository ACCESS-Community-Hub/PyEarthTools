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

As `EDIT` is not published to any Python Package repositories, it has to be manually cloned and installed.

```bash
git clone https://git.nci.org.au/bom/dset/edit-package/[PACKAGE]
cd [PACKAGE]
pip install ./
```
