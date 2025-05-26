# Utils API Index

This is the Utils package which forms a part of the [PyEarthTools package](https://github.com/ACCESS-Community-Hub/PyEarthTools).

The rest of this page contains reference information for the components of the Utils package. The Utils API docs can be viewed at [Utils API Docs](utils_api.md).

|  Module             |       Purpose                        |   API Docs     |
|---------------------|--------------------------------------|----------------|
|  `utils`            |                                      | - [dynamic_import](utils_api.md#pyearthtools.utils.dynamic_import)  |
|                     |                                      | - [load](utils_api.md#pyearthtools.utils.load)  |
|                     |                                      | - [save](utils_api.md#pyearthtools.utils.save)  |
| `utils.config`      |                                      | - [canonical_name](utils_api.md#pyearthtools.utils.config.canonical_name)
|                     |                                      | - [merge](utils_api.md#pyearthtools.utils.config.merge)
|                     |                                      | - [collect_yaml](utils_api.md#pyearthtools.utils.config.collect_yaml)
|                     |                                      | - [collect_env](utils_api.md#pyearthtools.utils.config.collect_env)
|                     |                                      | - [ensure_file](utils_api.md#pyearthtools.utils.config.ensure_file)
|                     |                                      | - [collect](utils_api.md#pyearthtools.utils.config.collect)
|                     |                                      | - [refresh](utils_api.md#pyearthtools.utils.config.refresh)
|                     |                                      | - [get](utils_api.md#pyearthtools.utils.config.get)
|                     |                                      | - [pop](utils_api.md#pyearthtools.utils.config.pop)
|                     |                                      | - [update_defaults](utils_api.md#pyearthtools.utils.config.update_defaults)
|                     |                                      | - [expand_environment_variables](utils_api.md#pyearthtools.utils.config.expand_environment_variables)
|                     |                                      | - [rename](utils_api.md#pyearthtools.utils.config.rename)
|                     |                                      | - [check_deprecations](utils_api.md#pyearthtools.utils.config.check_deprecations)
|                     |                                      | - [set](utils_api.md#pyearthtools.utils.config.set)
| `utils.context`     |                                      | - [ChangeValue](utils_api.md#pyearthtools.utils.context.ChangeValue)
|                     |                                      | - [Catch](utils_api.md#pyearthtools.utils.context.Catch)
|                     |                                      | - [PrintOnError](utils_api.md#pyearthtools.utils.context.PrintOnError)
| `utils.data`        |                                      | - [Tesselator](utils_api.md#pyearthtools.utils.data.Tesselator)
|                     |                                      | - [converter.xarrayConverter](utils_api.md#pyearthtools.utils.data.converter.xarrayConverter)
|                     |                                      | - [converter.NumpyConverter](utils_api.md#pyearthtools.utils.data.converter.NumpyConverter)
|                     |                                      | - [converter.DaskConverter](utils_api.md#pyearthtools.utils.data.converter.DaskConverter)
| `utils.decorators`  | Define parameters which are aliases (e.g 't2m' and '2t')  | - [alias_arguments](utils_api.md#pyearthtools.utils.decorators.alias_arguments)  |
|                     | Reverse a 1:1 dictionary mapping     | - [invert_dictionary_list](utils_api.md#pyearthtools.utils.decorators.invert_dictionary_list) |
|                     |                                      | - [classproperty](utils_api.md#pyearthtools.utils.decorators.classproperty) |
| `utils.initialisation`   |                                 | - [dynamic_import](utils_api.md#pyearthtools.utils.dynamic_import) |
|                     |                                      | - [load](utils_api.md#pyearthtools.utils.load) |
|                     |                                      | - [save](utils_api.md#pyearthtools.utils.save) |
|                     |                                      | - [update_contents](utils_api.md#pyearthtools.utils.update_contents) |
|                     |                                      | - [InitialisationRecordingMixin](utils_api.md#pyearthtools.utils.InitialisationRecordingMixin) |
|                     |                                      | - [Dumper](utils_api.md#pyearthtools.utils.Dumper) |
|                     |                                      | - [Loader](utils_api.md#pyearthtools.utils.Loader) |
| `utils.logger`      |                                      | - [initiate_logging](utils_api.md#pyearthtools.utils.logger.initiate_logging)
|                     |                                      | - [reconfigure](utils_api.md#pyearthtools.utils.logger.reconfigure)
| `utils.parameter`   |                                      | - [SingleParameter](utils_api.md#pyearthtools.utils.parameter.SingleParameter) |
|                     |                                      | - [ListParameter](utils_api.md#pyearthtools.utils.parameter.ListParameter) |
|                     |                                      | - [RangeParameter](utils_api.md#pyearthtools.utils.parameter.RangeParameter) |
|                     |                                      | - [search](utils_api.md#pyearthtools.utils.parameter.search) |
|                     |                                      | - [search_threaded](utils_api.md#pyearthtools.utils.repr_utils.search_threaded) |
| `utils.repr_utils`  |                                      | - [provide_html](utils_api.md#pyearthtools.utils.repr_utils.provide_html) |
|                     |                                      | - [html](utils_api.md#pyearthtools.utils.repr_utils.html) |
|                     |                                      | - [default](utils_api.md#pyearthtools.utils.repr_utils.default  ) |

