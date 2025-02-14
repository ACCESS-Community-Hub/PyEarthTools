# PyEarthTools Documentation

This documentation has been prepared with [MkDocs](https://www.mkdocs.org/).

To generate it locally, first install all dependencies in a Conda environment:

```
git clone https://github.com/ACCESS-Community-Hub/PyEarthTools.git
cd PyEarthTools/packages/documentation
conda env create -f environment.yml -p ./venv
```

then use the `mkdocs` tool to build the documentation

```
conda run -p ./venv mkdocs build
```
