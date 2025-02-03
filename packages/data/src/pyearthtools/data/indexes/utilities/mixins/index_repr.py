# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


UTILS_REPR = False
try:
    import pyearthtools.utils

    UTILS_REPR = True
except ImportError:
    UTILS_REPR = False

import pyearthtools.data


class reprMixin:
    def __get_steps_for_repr(self):
        class Information:
            def __init__(self, name, info, doc=""):
                self._doc_ = doc
                self._name_ = name
                self._info_ = info

        info = []
        if hasattr(self, "catalog"):
            if isinstance(self.catalog, (pyearthtools.data.Catalog, pyearthtools.data.CatalogEntry)):
                cat_dict = self.catalog.to_dict()
                # info.append(Information('args', {'args': cat_dict['args']}))
                info.append(Information("Initialisation", cat_dict["kwargs"]))
            # elif isinstance(self.catalog, pyearthtools.data.Collection):
            #     info.append(Information('Catalog', {cat.name: cat.to_dict()['kwargs'] for cat in self.catalog}))
            else:
                raise NotImplementedError()

        if hasattr(self, "base_transforms"):
            transforms = {}
            for trans in self.base_transforms:
                name = str(trans.__class__.__name__)
                while name in transforms:
                    if "[" in name:
                        existing_number = int(name[name.index("[") + 1 : name.index("]")])
                        name = name.replace(f"[{existing_number}]", f"[{existing_number+1}]")
                    else:
                        name = name + "[1]"
                transforms[name] = getattr(trans, "_info_", None)
            info.append(Information("Transforms", transforms))

        if hasattr(self, "preprocess_transforms"):
            preprocess = self.preprocess_transforms
            if not preprocess is None:
                preprocess = pyearthtools.data.transforms.TransformCollection(preprocess)

                transforms = {}
                for trans in preprocess:
                    name = str(trans.__class__.__name__)
                    while name in transforms:
                        if "[" in name:
                            existing_number = int(name[name.index("[") + 1 : name.index("]")])
                            name = name.replace(f"[{existing_number}]", f"[{existing_number+1}]")
                        else:
                            name = name + "[1]"
                    transforms[name] = getattr(trans, "_info_", None)
                info.append(Information("Preprocess", transforms))

        return info

    def __repr__(self) -> str:
        if not UTILS_REPR:
            return super().__repr__()

        try:
            steps = self.__get_steps_for_repr()
        except NotImplementedError:
            return super().__repr__()

        return pyearthtools.utils.repr_utils.standard(
            *steps,
            name=f"{self.__class__.__name__}",
            description=getattr(self, "_desc_", None),
            documentation_attr="_doc_",
            info_attr="_info_",
            name_attr="_name_",
            expanded=True,
        )

    def _repr_html_(self):
        if not UTILS_REPR:
            return repr(self)

        try:
            steps = self.__get_steps_for_repr()
        except NotImplementedError:
            return repr(self)

        backup_repr = self.__repr__() or "HTML Repr Failed"

        return pyearthtools.utils.repr_utils.html(
            *steps,
            name=f"{self.__class__.__name__}",
            description=getattr(self, "_desc_", None),
            documentation_attr="_doc_",
            name_attr="_name_",
            info_attr="_info_",
            backup_repr=backup_repr,
            expanded=True,
        )

    @property
    def _info_(self):
        if hasattr(self, "_recorded_info"):
            return self._recorded_info

        if hasattr(self, "catalog"):
            if isinstance(self.catalog, (pyearthtools.data.Catalog, pyearthtools.data.CatalogEntry)):
                cat_dict = self.catalog.to_dict()
                return cat_dict["kwargs"]
        return {}

    @_info_.setter
    def _info_(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"'_info_' must be a dictionary, not {type(value)!r}")
        self._recorded_info = value
