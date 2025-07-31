
# API Documentation Index

PyEarthTools comprises multiple sub-packages which can be used individually or together.

|    API                                         | How-To Guide                        |     Purpose   |
|---------------------------------------------|-------------------------------------|---------------|
|   [Data API](data/data_index.md)              |  [Data How-To](data/data_how_to.md) | Loading Earth system data sets to xarray for processing  |
|    [Utils API](utils/utils_index.md)           |   n/a (internal)                    | Code for common functionality across the sub-packages |
|    [Pipeline API](pipeline/pipeline_index.md)  | [Pipeline How-To](pipeline/pipeline_how_to.md)  | Process and normalise Earth system data ready for machine learning |
|     [Bundled Models](bundled_models/bundled_index.md) | n/a (to be done)              | Maintained versions of specific, bundled models which can be easily trained and run |
|    [Training API](training/training_index.md)  |  n/a (to be done)                   | Training processes for machine learning models |
|    [Tutorial API](tutorial/tutorial_index.md)  | n/a (internal)                      | Contains helper code for data sets used in tutorials |
|     [Zoo API](zoo/zoo_index.md)                 | na/ (internal)                      | Contains code for managing registered models (such as the bundled models) |
|   Evaluation                               | (to be done)                        | (Coming soon) Contains code for producing standard evaluations (such as benchmarks and scorecards) |


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
