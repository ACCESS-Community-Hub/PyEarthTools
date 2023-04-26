from distutils.core import setup

setup(
    name="edit-training",
    version="0.1",
    description="EDIT Machine Learning Training",
    author="Harrison Cook",
    author_email="harrison.cook@bom.gov.au",
    url="www.bom.gov.au",
    # packages=["edit"],
    package_dir={"": "src"},
    install_requires=["torch", "einops", "pytorch-lightning"],
    entry_points={
        "console_scripts": ["edit_training=edit.training.trainer.commands:entry_point"]
    },
)
