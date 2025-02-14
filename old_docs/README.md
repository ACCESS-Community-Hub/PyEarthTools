# PyEarthTools Documentation

> [!WARNING]
> This documentation is outdated and will be replaced.
> It does not represent the current state of the project.
>
> Current target audience are PyEarthTools developers doing the documentation refactoring.

This documentation has been prepared with [MkDocs](https://www.mkdocs.org/).

To generate it locally, first install all dependencies in a Conda environment:

```
git clone https://github.com/ACCESS-Community-Hub/PyEarthTools.git
cd PyEarthTools/old_docs
conda env create -f environment.yml -p ./venv
```

then use the `mkdocs` tool to build the documentation

```
conda run -p ./venv mkdocs build
```
