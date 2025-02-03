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

**Note for maintainers:** The file `environment.lock.yaml` has been generated to keep a record of a working environment on a Linux system. It has been exported using the following command to make it more portable:

```
conda env export --no-builds -p ./venv | sed \
    -e '/^name/,+1 d' \
    -e '/^prefix/,+1 d' \
    -e 's|pyearthtools-data.*|-e ../data|' \
    -e 's|pyearthtools-pipeline.*|-e ../pipeline|' \
    -e 's|pyearthtools-training.*|-e ../training[lightning]|' \
    -e 's|pyearthtools-tutorial.*|-e .|' \
    -e 's|pyearthtools-utils.*|-e ../utils|' \
    -e 's/python-graphviz/graphviz/' \
    > environment.lock.yml
```
