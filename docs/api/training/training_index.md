# Training API Index

This is the training package which forms a part of the [PyEarthTools package](https://github.com/ACCESS-Community-Hub/PyEarthTools).

The rest of this page contains reference information for the components of the Training package. The Training API docs can be viewed at [Training API Docs](training_api.md).

|  Module               |       Purpose                        |   API Docs     |
|-----------------------|--------------------------------------|----------------|
|  `training.dataindex` |                                      | - [MLDataIndex](training_api.md#pyearthtools.training.dataindex.MLDataIndex)  |
|  `training.manage`    |                                      | - [Variables](training_api.md#pyearthtools.training.manage.Variables)  |
|  `training.data`      |                                      | - [PipelineDataModule](training_api.md#pyearthtools.training.data.PipelineDataModule)  |
|                       |                                      | - [default](training_api.md#pyearthtools.training.data.default)  |
|                       |                                      | - [save](training_api.md#pyearthtools.training.data.save)  |
|                       |                                      | - [load](training_api.md#pyearthtools.training.data.load)  |
|  `training.wrapper`   |                                      | - [ModelWrapper](training_api.md#pyearthtools.training.wrapper.ModelWrapper)  |
|                       |                                      | - [TrainingWrapper](training_api.md#pyearthtools.training.wrapper.TrainingWrapper)  |
|                       |                                      | - [Predictor](training_api.md#pyearthtools.training.wrapper.Predictor)  |
|  `training.wrapper.lightning`   |                            | - [predict.LightningPrediction](training_api.md#pyearthtools.training.wrapper.lightning.Predict)  |
|                       |                                      | - [train.LightningTraining](training_api.md#pyearthtools.training.wrapper.lightning.Train)  |
|                       |                                      | - [predict.LoggingContext](training_api.md#pyearthtools.training.wrapper.lightning.predict.LoggingContext)  |
|                       |                                      | - [train.get_logger](training_api.md#pyearthtools.training.wrapper.lightning.train.get_logger)  |
|                       |                                      | - [train.make_callback](training_api.md#pyearthtools.training.wrapper.lightning.train.make_callback)  |
|                       |                                      | - [wrapper.LightningWrapper](training_api.md#pyearthtools.training.wrapper.lwrapper.LightningWrapper)  |
|  `training.wrapper.predict`   |                              | - [Predictor](training_api.md#pyearthtools.training.wrapper.predict.Predictor)  |
|                       |                                      | - [TimeSeriesPredictor](training_api.md#pyearthtools.training.wrapper.predict.TimeSeriesPredictor)  |
|                       |                                      | - [TimeSeriesAutoRecurrentPredictor](training_api.md#pyearthtools.training.wrapper.predict.TimeSeriesAutoRecurrentPredictor)  |
|                       |                                      | - [TimeSeriesManagedPredictor](training_api.md#pyearthtools.training.wrapper.predict.TimeSeriesManagedPredictor)  |
|                       |                                      | - [ManualTimeSeriesPredictor](training_api.md#pyearthtools.training.wrapper.predict.ManualTimeSeriesPredictor)  |
|  `training.wrapper.onnx` |                                   | - [ONNXWrapper](training_api.md#pyearthtools.training.wrapper.onnx.ONNXWrapper)  |


