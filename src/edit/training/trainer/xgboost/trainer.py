from __future__ import annotations

import numpy as np
import xarray as xr
from pathlib import Path


from edit.training.trainer.template import EDITTrainer
from edit.training.data.templates import DataStep

class EDITXGBoostTrainer(EDITTrainer):
    def __init__(self, model, train_data: DataStep, valid_data: DataStep = None, path: str | Path = None, **kwargs) -> None:
        super().__init__(model, train_data, valid_data, path)

        # Initialise Model

    def fit(self, num_iterations: int):
        for i, data in self.train_data:
            if i >= num_iterations:
                break
            # Fit Model
            pass

    def eval(self):
        #Evaluate Model
        pass

    def _predict_from_data(self, data, **kwargs):
        #Handle Predictions
        pass

    def load(self, path : str | Path):
        ##Load Model
        raise NotImplementedError

    def save(self, path: str | Path):
        raise NotImplementedError
        