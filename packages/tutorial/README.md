# EDIT tutorials

First clone this repository:

```
git clone git@github.com:informatics-lab/EDIT.git
cd EDIT/packages/tutorial
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
python -m ipykernel install --user --name EDIT-tutorial
```

**Note for maintainers:** The file `environment.lock.yaml` has been generated to keep a record of a working environment on a Linux system. It has been exported using the following command to make it more portable:

```
conda env export --no-builds -p ./venv | sed \
    -e '/^name/,+1 d' \
    -e '/^prefix/,+1 d' \
    -e 's|edit-data.*|-e ../data|' \
    -e 's|edit-pipeline.*|-e ../pipeline|' \
    -e 's|edit-training.*|-e ../training[lightning]|' \
    -e 's|edit-tutorial.*|-e .|' \
    -e 's|edit-utils.*|-e ../utils|' \
    -e 's/python-graphviz/graphviz/' \
    > environment.lock.yml
```
