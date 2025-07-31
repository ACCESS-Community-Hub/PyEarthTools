# Utils API Docs


## `utils`

```{eval-rst}

.. autofunction:: pyearthtools.utils.dynamic_import
.. autofunction:: pyearthtools.utils.load
.. autofunction:: pyearthtools.utils.save
```


## `utils.config`

```{eval-rst}

.. autofunction:: pyearthtools.utils.config.canonical_name
.. autofunction:: pyearthtools.utils.config.merge
.. autofunction:: pyearthtools.utils.config.collect_yaml
.. autofunction:: pyearthtools.utils.config.collect_env
.. autofunction:: pyearthtools.utils.config.ensure_file
.. autofunction:: pyearthtools.utils.config.collect
.. autofunction:: pyearthtools.utils.config.refresh
.. autofunction:: pyearthtools.utils.config.get
.. autofunction:: pyearthtools.utils.config.pop
.. autofunction:: pyearthtools.utils.config.update_defaults
.. autofunction:: pyearthtools.utils.config.expand_environment_variables
.. autofunction:: pyearthtools.utils.config.rename
.. autofunction:: pyearthtools.utils.config.check_deprecations

.. autoclass:: pyearthtools.utils.config.set
    :members:


```


## `utils.context`

```{eval-rst}

.. autoclass:: pyearthtools.utils.context.ChangeValue
    :members:
.. autoclass:: pyearthtools.utils.context.Catch
    :members:
.. autoclass:: pyearthtools.utils.context.PrintOnError
    :members:
```

## `utils.data`

```{eval-rst}

.. autoclass:: pyearthtools.utils.data.Tesselator
    :members:
.. autoclass:: pyearthtools.utils.data.converter.xarrayConverter
    :members:
.. autoclass:: pyearthtools.utils.data.converter.NumpyConverter
    :members:
.. autoclass:: pyearthtools.utils.data.converter.DaskConverter
    :members:
```

## `utils.decorators`

```{eval-rst}

.. autofunction:: pyearthtools.utils.decorators.alias_arguments
.. autofunction:: pyearthtools.utils.decorators.invert_dictionary_list

.. autoclass:: pyearthtools.utils.decorators.classproperty
    :members:

```

## `utils.initialisation`

```{eval-rst}

.. autofunction:: pyearthtools.utils.initialisation.dynamic_import
.. autofunction:: pyearthtools.utils.initialisation.load
.. autofunction:: pyearthtools.utils.initialisation.save
.. autofunction:: pyearthtools.utils.initialisation.update_contents

.. autoclass:: pyearthtools.utils.initialisation.InitialisationRecordingMixin
    :members:

.. autoclass:: pyearthtools.utils.initialisation.Dumper
    :members:

.. autoclass:: pyearthtools.utils.initialisation.Loader
    :members:

```

## `utils.logger`

```{eval-rst}

.. autofunction:: pyearthtools.utils.logger.initiate_logging
.. autofunction:: pyearthtools.utils.logger.reconfigure
```

## `utils.parameter`

```{eval-rst}

.. autofunction:: pyearthtools.utils.parameter.search
.. autofunction:: pyearthtools.utils.parameter.search_threaded
.. autoclass:: pyearthtools.utils.parameter.SingleParameter
    :members:
.. autoclass:: pyearthtools.utils.parameter.ListParameter
    :members:
.. autoclass:: pyearthtools.utils.parameter.RangeParameter
    :members:
```

## `utils.repr_utils`

```{eval-rst}

.. autofunction:: pyearthtools.utils.repr_utils.provide_html
.. autofunction:: pyearthtools.utils.repr_utils.html
.. autofunction:: pyearthtools.utils.repr_utils.default
```
