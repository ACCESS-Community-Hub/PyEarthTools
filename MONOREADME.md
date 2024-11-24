# EDIT

*This is a placeholder README.*

First clone the repository using the `monorepo` branch:

```
git clone git@github.com:informatics-lab/EDIT.git
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

And install all dependencies via pip:

```
pip install -r requirements-dev.txt
```

You can run some of the tests to check that your installation worked:

```
pytest packages/data/tests/
```

To run example [notebooks from the tutorial package](packages/tutorial/nbook), you may need to also install a Jupyter kernel for your environment:

```
# from the activate virtual or conda environment
pip install ipykernel
python -m ipykernel install --user --name EDIT
```
