# PyEarthTools Documentation

This documentation has been prepared with [MkDocs](https://www.mkdocs.org/).

To generate it locally, first create a virtual environment and install dependencies in it:

```
git clone https://github.com/ACCESS-Community-Hub/PyEarthTools.git
cd PyEarthTools/packages/documentation
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

then use the `mkdocs` tool to build the documentation

```
. venv/bin/activate
mkdocs build
```
