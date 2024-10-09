# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

#
# Taken from https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/utilities/parsing.py
#

from collections import OrderedDict
import types
from typing import (
    Any,
    Dict,
    List,
    Literal,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    Mapping,
)
import inspect


class InitialisationRecord(Mapping):
    def __init__(self, args, kwargs) -> None:
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    def __getitem__(self, key):
        if key == "args":
            return self._args
        return self._kwargs[key]

    def __iter__(self):
        yield "args"
        for key in self._kwargs:
            yield key

    def __len__(self):
        return len(self._kwargs) + 1

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs


def parse_class_init_keys(cls: Type) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse key words for standard ``self``, ``*args`` and ``**kwargs``.

    Examples:

        >>> class Model:
        ...     def __init__(self, hparams, *my_args, anykw=42, **my_kwargs):
        ...         pass
        >>> parse_class_init_keys(Model)
        ('self', 'my_args', 'my_kwargs')
    """
    init_parameters = inspect.signature(cls.__init__).parameters
    # docs claims the params are always ordered
    # https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
    init_params = list(init_parameters.values())
    # self is always first
    n_self = init_params[0].name

    def _get_first_if_any(
        params: List[inspect.Parameter],
        param_type: Literal[inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD],
    ) -> Optional[str]:
        for p in params:
            if p.kind == param_type:
                return p.name
        return None

    n_args = _get_first_if_any(init_params, inspect.Parameter.VAR_POSITIONAL)
    n_kwargs = _get_first_if_any(init_params, inspect.Parameter.VAR_KEYWORD)

    return n_self, n_args, n_kwargs


def _get_init_args(frame: types.FrameType) -> Tuple[Optional[Any], Dict[str, Any]]:
    _, _, _, local_vars = inspect.getargvalues(frame)
    if "__class__" not in local_vars:
        return None, {}
    cls = local_vars["__class__"]
    init_parameters = inspect.signature(cls.__init__).parameters
    self_var, args_var, kwargs_var = parse_class_init_keys(cls)
    filtered_vars = [n for n in (self_var, args_var, kwargs_var) if n]
    exclude_argnames = (*filtered_vars, "__class__", "frame", "frame_args")
    # only collect variables that appear in the signature

    keys = list(set(init_parameters.keys() & local_vars.keys()))
    local_args = {k: local_vars[k] for k in keys}
    # kwargs_var might be None => raised an error by mypy
    if kwargs_var:
        local_args.update(local_args.get(kwargs_var, {}))
    if args_var:
        local_args["__args"] = local_args[args_var]
    local_args = {k: v for k, v in local_args.items() if k not in exclude_argnames}
    self_arg = local_vars.get(self_var, None)
    return self_arg, local_args


def collect_init_args(
    frame: types.FrameType,
    path_args: List[Dict[str, Any]],
    inside: bool = False,
    classes: Tuple[Type, ...] = (),
) -> List[Dict[str, Any]]:
    """Recursively collects the arguments passed to the child constructors in the inheritance tree.

    Args:
        frame: the current stack frame
        path_args: a list of dictionaries containing the constructor args in all parent classes
        inside: track if we are inside inheritance path, avoid terminating too soon
        classes: the classes in which to inspect the frames

    Return:
          A list of dictionaries where each dictionary contains the arguments passed to the
          constructor at that level. The last entry corresponds to the constructor call of the
          most specific class in the hierarchy.
    """
    _, _, _, local_vars = inspect.getargvalues(frame)

    # frame.f_back must be of a type types.FrameType for get_init_args/collect_init_args due to mypy
    if not isinstance(frame.f_back, types.FrameType):
        return path_args

    local_self, local_args = _get_init_args(frame)

    if "__class__" in local_vars and (not classes or isinstance(local_self, classes)):
        # recursive update
        path_args.append(local_args)
        return collect_init_args(frame.f_back, path_args, inside=True, classes=classes)

    if not inside:
        return collect_init_args(frame.f_back, path_args, inside=False, classes=classes)
    return path_args


def get_initialise_args(
    obj,
    ignore: Optional[Union[Sequence[str], str]] = None,
    frame: Optional[types.FrameType] = None,
) -> dict:
    if not frame:
        current_frame = inspect.currentframe()
        # inspect.currentframe() return type is Optional[types.FrameType]: current_frame.f_back called only if available
        if current_frame:
            frame = current_frame.f_back
    if not isinstance(frame, types.FrameType):
        raise AttributeError("There is no `frame` available while being required.")

    init_args = {}
    for local_args in collect_init_args(frame, [], classes=(type(obj),))[-1:]:
        init_args.update(local_args)

    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]
    elif isinstance(ignore, (list, tuple)):
        ignore = [arg for arg in ignore if isinstance(arg, str)]

    ignore = list(set(ignore))
    init_args = {k: v for k, v in init_args.items() if k not in ignore}

    init_args_keys = list(init_args.keys())
    init_args_keys.sort()
    init_args = {k: init_args[k] for k in init_args_keys}

    return init_args
