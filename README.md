# EDIT

*This is a placeholder README.*

First clone the repository using the `monorepo` branch:

```
git clone -b monorepo git@github.com:informatics-lab/EDIT.git
cd EDIT
```

Then create a Python virtual environment:

```
python3 -m venv venv
venv/bin/activate
```

or a Conda environment to install all dependencies:

```
conda create -p ./venv -y python=3.11
conda activate ./venv
```

And finally install all dependencies via pip:

```
pip install -r requirements-dev.txt
```
