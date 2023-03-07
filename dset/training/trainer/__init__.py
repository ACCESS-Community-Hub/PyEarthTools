
from dset.training.trainer.trainer import DSETTrainerWrapper
from dset.training.trainer import commands
from dset.training.trainer.yaml import load_from_yaml
from dset.training.trainer.index import MLDataIndex

if __name__ == "__main__":
    commands.entry_point()
