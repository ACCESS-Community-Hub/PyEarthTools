# PyEarthTools tutorials

First clone this repository:

```
git clone https://github.com/ACCESS-Community-Hub/PyEarthTools.git
cd PyEarthTools/packages/tutorial
```

Then create a Conda environment to install all dependencies:

```
conda env create -f environment.yml -p ./venv
```

To run the example [notebooks](nbook/), you can either

- start a JupyterLab instance

```
conda activate ./venv
jupyter lab
```

- or install a Jupyter kernel to use in a pre-existing JupyterLab installation

```
conda activate ./venv
python -m ipykernel install --user --name PET-tutorial
```
