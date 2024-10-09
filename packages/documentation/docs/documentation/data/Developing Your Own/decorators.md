# Initialisation Decorators

[edit.data.indexes][edit.data.indexes.decorators] contains a few decorators useful for controlling and ensuring the sanity of inputs into `data indexes`.

As datasets are fairly well defined, it can be useful to limit the input range of various fields, or provide aliases for parameters to suit the various conventions used by various teams and agencies. These decorators are thus useful to wrap the `__init__` function of `indexes` with.

See [decorators][edit.data.indexes.decorators] for reference.

## How to use

### Aliases

This section discusses the purpose of the `@alias_argument` decorator, and shows how to use it.

This decorator allows a function to accept a range of parameters names, to make the interface easier to use for the end-user.

```python
from edit.data.indexes.decorators import alias_argument

@alias_argument(param="parameter")
def func(param):
    return param

func(param = "test")
# -> "test"

func(parameter = "test")
# -> "test"
```

The decorator also accepts a list of alias values.

```python
from edit.data.indexes.decorators import alias_argument

@alias_argument(param=["parameter", "variables"])
def func(param):
    return param

func(param = "test")
# -> "test"

func(parameter = "test")
# -> "test"

func(variables = "test")
# -> "test"
```

### Checking

This section discusses the purpose of the `@check_arguments` decorator, and shows how to use it.

This decorator limits the range of valid parameters to a given list, and if `str` provides a list of similar words, in the case of a spelling error.

```python
from edit.data.indexes.decorators import check_arguments

@check_arguments(variable = ['temp', 'sst'])
def func(variable):
    return variable

func(variable = "temp")
# -> "temp"

func(variable = "u")
# An Error is Raised
```

#### Path to a .valid file

The `@check_arguments` can also take a path to a `.valid` file. This path must be part of the package, and can contain `{}` with keys from other initlisation arguments.

Say `module.submodule.default.valid` contains:

```plain
temp
sst
```

And `module.submodule.test.valid` contains:

```plain
u
v
```

```py
from edit.data.indexes.decorators import check_arguments

@check_arguments(variable = 'module.submodule.{argument}.valid')
def func(variable, argument = 'default'):
    return variable

func(variable = "temp")
# -> "temp"

func(variable = "u")
# An Error is Raised

func(variable = "u", argument = 'test')
# -> "u"
```

#### Path to a .struc file

The `@check_arguments` can also take a path to a `.struc` file, which is a form of the directory structure
of the data, which can be defined by the class arguments.

This file must be structured like a `yaml` file, and must contain at least an `order` field, which 
specifies the order of the variables. The rest of the file must contain a tree of valid arguments in the order defined in `order`.

A `struc` file can be automatically created with `edit.data.commands.structure`, and then the `order` added.

For example, a `.struc` file for an fake dataset:

```yaml

order:
    product
    variable
    sub_variable
product_1:
    variable_1:
        sub_var_1_1
    variable_2:
        sub_var_2_1
        sub_var_2_2
        sub_var_2_3
product_2:
    variable_3:
        sub_var_3_1
        sub_var_3_2
```

This clearly outlines the directory structure, and will force the user to use valid arguments depending on the branch being followed.

##### Variable Defaults

It may be obvious already, but with this structure it is impossible to set an appropriate default value, as underlying arguments change depending on the branch.
When using the `struc` file, it is possible to set a `VariableDefault` which will collapse the variable if only one option is available, and raise an error if
more than one is available.

```py
from edit.data.indexes.decorators import check_arguments
from edit.data.indexes import VariableDefault

@check_arguments(struc = 'module.submodule.data.struc')
def func(product = 'product_1', variable = VariableDefault, sub_variable = VariableDefault):
    print(product, variable, sub_variable)

func(product = "product_1", variable_1 = "variable_1")
# -> product_1, variable_1, sub_var_1_1

func(product = "product_1", variable_1 = "variable_2")
# An Error is Raised, as more than one sub_variable is available

func(product = "product_2", sub_variable = "sub_var_3_1")
# -> product_2, variable_3, sub_var_3_1
```

### Modifications

`edit.data` allows for data modifications to be made. These directly interface with the indexes, and allow advanced usage of the variables each dataset has. 

When specifying the variables, the modification syntax can be used

```!MODIFICATION[KWARGS]:VARIABLE>NEWNAME```

Where, `MODIFICATION` references the modification name, `KWARGS` contains the parameters needed for the modification in json form, and `VARIABLE` is the variable name as would be used normally and with anything after `>` being the new name, which can be ommited.

Or dictionary with following keys

```text
    - source_var (REQUIRED)     Variable to modify
    - modification (REQUIRED)   Modification to apply
    - target_var                Rename of variable
    - **                        Any other keys for `modification`
```

```python
Archive('!accumulate[period = "6 hourly"]:tcwv)
```

However, they must be enabled by each index, specifiying how to find the variable keyword.


```python
class DataIndex(ArchiveIndex):
    @decorators.variable_modifications(variable_keyword="variables")
    def __init__(self, variables, ...):
        ...
```