from distutils.core import setup

setup(
    name="dset-training",
    version="0.1",
    description="DSET Machine Learning Training",
    author="Harrison Cook",
    author_email="harrison.cook@bom.gov.au",
    url="www.bom.gov.au",
    packages=["dset"],
    entry_points={
        "console_scripts": ["training=dset.training.trainer.from_yaml:entry_point"]
    },
)
