
# API Documentation

PyEarthTools comprises multiple sub-packages which can be used individually or together.

|    Sub-Package                            |  Purpose  |
|-------------------------------------------|---------------------- |
|  [Data](data/data_index.md)               | Loading and indexing into well-known Earth system data sets to produce ML-ready data structures |
|  [Utils](utils/utils_index.md)            | Code for common functionality across the sub-packages |
|  [Pipeline](pipeline/pipeline_index.md)   | Definining reproducible sequences of operations with the ability to cache results |
|  [Training](training/training_index.md)   | Code defining the training processes and schedules of a machine learning model |
|  [Tutorial](tutorial/tutorial_index.md)   | Contains helper code for data sets used in tutorials |
|  [Bundled Models](bundled_models/bundled_index.md) | Maintained versions of specific, bundled models which can be easily trained and run |
|  [Zoo](zoo/zoo_index.md)                  | Contains code for managing registered models (such as the bundled models) |
|  Evaluation                               | (Coming soon) Contains code for producing standard evaluations (such as benchmarks and scorecards) |


```{toctree}
:hidden:

bundled_models/bundled_index
bundled_models/bundled_api
data/data_index
data/data_api
pipeline/pipeline_index
pipeline/pipeline_api
training/training_index
training/training_api
tutorial/tutorial_index
tutorial/tutorial_api
utils/utils_index
utils/utils_api
zoo/zoo_index
zoo/zoo_api
```