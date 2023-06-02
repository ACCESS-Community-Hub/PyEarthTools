from __future__ import annotations

import numpy as np
import tqdm
import xarray as xr
from pathlib import Path
import os
import json
import yaml
import sklearn
import xgboost
import matplotlib.pyplot as plt
import logging

import sys
sys.path.append('/g/data/kd24/mr1465/EDIT')
import edit.data
import edit.training

from edit.training.trainer.template import EDITTrainer
from edit.training.data.templates import DataStep

class EDITXGBoostTrainer(EDITTrainer):
    def __init__(self, 
                 model, 
                 train_data: DataStep, 
                 valid_data: DataStep = None, 
                 path: str | Path = None, 
                 **kwargs
                 ) -> None:
        super().__init__(model, train_data, valid_data, path)

        # Initialise Model
        self.model = model

        self.path = Path(path)


    def fit(self, num_batches: int=2, load: bool = False, verbose: bool = True):

        if load:
            # Use existing model
            xgb_model = self.load(path = self.path)

        else:
            xgb_model = None

        print(f'Getting batch 0 / {num_batches}...')
        for i, data in tqdm.tqdm(enumerate(self.train_data), disable=not verbose):
            print(f'Getting batch {i} / {num_batches}...')
            if i >= num_batches - 1:
                break

            # Fit Model
            print('Fitting model...')
            self.model.fit(*data, xgb_model)
            xgb_model = self.model.get_booster()

    
    def _predict_from_data(self, data: tuple, **kwargs):
        # Handle Predictions

        (X, y) = data
        y_pred = self.model.predict(X)
        if y_pred.shape == 1:
            y_pred = np.expand_dims(y_pred, -1)
        
        return X, y_pred 

    def load(self, path : str | Path = None):
        # Load Model
        if isinstance(path, bool):
            if path:
                path = None
            else:
                return

        if path is None:
            path = Path(self.path)

        self.model = xgboost.XGBRegressor()
        self.model.load_model(path / 'model.json')


    def save(self, path: str | Path = None):
        # Save model
        if path is None:
            path = Path(self.path)

        self.model.save_model(path / "model.json")

    
    def eval(self, max_samples: int=None):
        # Evaluate Model

        # Feature importance
        print('Getting feature importances...')
        feature_names = self.get_feature_names()
        self._feature_importance(feature_names)
        
        # Tree
        print('Getting tree...')
        self._view_tree()

        # Case study time plots
        try:
            print('Plotting case study time...')
            self._plot_case()
        except:
            print("Couldn't do case study")

        # Stats

        data_sets = dict(valid_data = self.valid_data,
                         train_data = self.train_data,
                         )

        for key, data_pipe in data_sets.items():
            print(f'Getting {key} statistics...')
            for data in data_pipe:
                # Only first batch
                break

            if max_samples:
                data = tuple(map(lambda x: x[:max_samples], data))

            X, y = data
            print(f'{key} y_min:', y.min())
            print(f'{key} y_max:', y.max())

            y_pred = self.model.predict(X)

            print(f'{key} pred y_min:', y.min())
            print(f'{key} pred y_max:', y.max())

            self.eval_statistics(y, y_pred, key)
        

    def eval_statistics(self, y, y_pred, data_set):
        # Evaluate on given data

        metrics = {}
        metrics['mae'] = sklearn.metrics.mean_absolute_error(y, y_pred).astype(float)
        metrics['rmse'] = np.sqrt(sklearn.metrics.mean_squared_error(y, y_pred)).astype(float)
        metrics['bias'] = np.mean(y_pred-y).astype(float)
        metrics['corr'] = sklearn.metrics.r2_score(y, y_pred).astype(float)
        # metrics['mgd'] = sklearn.metrics.mean_gamma_deviance(y, y_pred)

        print(metrics)

        # Save
        with open(self.path / f"{data_set}_evaluation.json", 'w') as f:
            f.write(json.dumps(metrics))

        # TODO view metrics instead of print.

    
    def _plot_case(self, test_time: str = '20220303T0000', vmax: int=255):
        # Get preds for time

        with edit.training.data.context.PatchingUpdate(self, stride_size=[1,1]):
            truth, predictions = self.predict(test_time)

        # Plot
        fig, axs = plt.subplots(1,3,figsize=(15,4), layout='tight')

        truth.cloud_optical_depth.plot(ax=axs[0], vmin=0, vmax=vmax)
        axs[0].set_title('Truth')
        predictions.cloud_optical_depth.plot(ax=axs[1], vmin=0, vmax=vmax)
        axs[1].set_title('Prediction')
        (predictions - truth).cloud_optical_depth.rename('error').plot(ax=axs[2], vmin=-255, vmax=255)
        axs[2].set_title('Error')
        fig.suptitle(test_time)
        fig.savefig(self.path / f"example_view-{test_time}.jpg", dpi=300)

        # Scatter
        fig, ax = plt.subplots(1,1)
        ax.scatter(truth.cloud_optical_depth.values.flatten(), predictions.cloud_optical_depth.values.flatten(), s=1)
        fig.savefig(self.path / f"example_scatter-{test_time}.jpg", dpi=300)            


    
    def _view_tree(self):

        fig, ax = plt.subplots(1,1, figsize=(15,50))
        
        xgboost.plot_tree(self.model.get_booster(), rankdir='LR', ax=ax)
        fig.savefig(self.path / "tree.jpg", dpi=300)


    def _feature_importance(self, feature_names: np.array | None, max_num_features: int=40):
        # Importance

        if feature_names:
            self.model.get_booster().feature_names = feature_names

        f_score = self.model.get_booster().get_score(importance_type='gain')

        with open(self.path / "f_scores.json", 'w') as f:
            f.write(json.dumps(f_score))

        fig, ax = plt.subplots(1,1,figsize=(8,15))
        xgboost.plot_importance(self.model.get_booster(), ax=ax, max_num_features=max_num_features, color='k')
        fig.savefig(self.path / "f_scores.jpg", dpi=300, bbox_inches='tight')


    def get_feature_names(self):

        self.train_data('2021-03-03 00:00')
        variable_order = self.train_data.patching_config.Variables[0]

        config_path = Path(str(self.path) + '.yaml')

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        di_source = config['data']['Source']
        tsteps = di_source['iterators.TemporalInterface']['samples'][0]
        n_patches = di_source['operations.PatchingDataIndex']['kernel_size'][0]

        tsteps_str = ['t_minus' + str(x*10) for x in range(1,tsteps+1)][::-1]
        lats_str = ['lat' + str(x) for x in range(1,n_patches+1)]
        lons_str = ['lon' + str(x) for x in range(1,n_patches+1)]

        feature_names = []
        for var in variable_order:
            for tstep in tsteps_str:
                for lat in lats_str:
                    for lon in lons_str:
                        feature_names.append('_'.join([var, tstep, lat, lon]))

        return feature_names

        