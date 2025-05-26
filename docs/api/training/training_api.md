# Training API Docs

## `training.dataindex`

```{eval-rst}

.. autoclass:: pyearthtools.training.dataindex.MLDataIndex
    :members:

```

## `training.manage`

```{eval-rst}

.. autoclass:: pyearthtools.training.manage.Variables
    :members:
```

## `training.data`

```{eval-rst}

.. autoclass:: pyearthtools.training.data.PipelineDataModule
    :members:

.. autofunction:: pyearthtools.training.data.default
.. autofunction:: pyearthtools.training.data.save
.. autofunction:: pyearthtools.training.data.load
```

## `training.wrapper`

```{eval-rst}

.. autoclass:: pyearthtools.training.wrapper.ModelWrapper
    :members:
.. autoclass:: pyearthtools.training.wrapper.TrainingWrapper
    :members:    
.. autoclass:: pyearthtools.training.wrapper.Predictor

.. autoclass:: pyearthtools.training.wrapper.lightning.Predict
    :members:    
.. autoclass:: pyearthtools.training.wrapper.lightning.predict.LoggingContext
    :members:        
.. autoclass:: pyearthtools.training.wrapper.lightning.Train
    :members:        
.. autofunction:: pyearthtools.training.wrapper.lightning.train.get_logger
.. autofunction:: pyearthtools.training.wrapper.lightning.train.make_callback
.. autoclass:: pyearthtools.training.wrapper.lightning.wrapper.LightningWrapper

.. autoclass:: pyearthtools.training.wrapper.predict.Predictor
    :members:        
.. autoclass:: pyearthtools.training.wrapper.predict.TimeSeriesPredictor
    :members:            
.. autoclass:: pyearthtools.training.wrapper.predict.TimeSeriesAutoRecurrentPredictor
    :members:         
.. autoclass:: pyearthtools.training.wrapper.predict.TimeSeriesManagedPredictor
    :members:
.. autoclass:: pyearthtools.training.wrapper.predict.ManualTimeSeriesPredictor
    :members:

.. autoclass:: pyearthtools.training.wrapper.onnx.ONNXWrapper
    :members:    

```