# EDIT tutorials

First clone this repository:

```
git clone git@github.com:informatics-lab/EDIT.git
cd EDIT/packages/tutorial
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
pip install -r requirements.txt
```

To run example [notebooks](nbook/), you may need to also install a Jupyter kernel for your environment:

```
# from the activate virtual or conda environment
pip install ipykernel
python -m ipykernel install --user --name EDIT-tutorial
```
