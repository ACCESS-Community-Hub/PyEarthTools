# PyEarthTools Documentation

> [!WARNING]
> This documentation is outdated and will be replaced.
> It does not represent the current state of the project.
>
> Current target audience are PyEarthTools developers doing the documentation refactoring.

This documentation has been prepared with [MkDocs](https://www.mkdocs.org/).

To generate it locally, first install all dependencies, for example in a Python virtual environment:

```
git clone https://github.com/ACCESS-Community-Hub/PyEarthTools.git
cd PyEarthTools/old_docs

python3 -m venv ./venv
. venv/bin/activate

pip install -r requirements.txt
```

Then use the `mkdocs` tool to build the documentation

```
mkdocs build
```

or serve it locally to see all changes live

```
mkdocs serve
```
