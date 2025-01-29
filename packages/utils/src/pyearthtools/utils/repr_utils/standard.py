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



from typing import Optional

import copy


def dynamic_padding(name, length_):
    return name + "".join([" "] * (length_ - len(name)))


def clean_repr(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for key, val in obj.items():
            new_obj[key] = clean_repr(copy.copy(val))
        return new_obj
    if isinstance(obj, (tuple, list)) and len(obj) > 10:
        return repr(obj[:4])[:-1] + " ... " + repr(obj[-4:])[1:]
    return repr(obj)


def summarise_kwargs(variables: dict, padding: int = 1):
    li_items = []
    if not isinstance(variables, dict):
        return li_items

    padding = "".join(["\t"] * padding)  # type: ignore

    for k, v in variables.items():
        v_repr = clean_repr(v)
        li_items.append(f"{padding} {dynamic_padding(k, 30)} {v_repr}")

    return li_items


def information_section(name, inline_details="", details=[], collapsed=False, padding: int = 1):
    padding = "".join(["\t"] * padding)  # type: ignore

    sections = [f"{padding}{dynamic_padding(name, 30)} {inline_details}"]
    if not collapsed and details:
        for item in details:
            sections.append(item)
    return f"\n{padding}".join(sections)


def provide_repr(
    *objects,
    name: Optional[str] = None,
    description: Optional[dict] = None,
    documentation_attr: Optional[str] = None,
    info_attr: Optional[str] = None,
    name_attr: Optional[str] = None,
    backup_repr: str = "Failed to create HTML repr",
    expanded: Optional[bool] = None,
) -> str:
    expanded = expanded or len(objects) < 4
    sections = [str(name)]

    if description:
        sections.append(
            information_section(
                "Description",
                inline_details=description.pop("singleline", ""),
                details=summarise_kwargs(description),
                collapsed=False,
            )
        )
        sections.append("\n")

    for item in objects:
        inline_details = getattr(item, documentation_attr or "__doc__", getattr(item, "__doc__", None))
        inline_details = "" if inline_details is None else str(inline_details)
        inline_details = inline_details.split("\n")[0]

        details = getattr(item, info_attr, None) if info_attr else None

        name = getattr(item, name_attr, None) if name_attr else item.__class__.__name__

        sections.append(
            information_section(
                name,
                inline_details=inline_details,
                details=summarise_kwargs(details) if details else None,
                collapsed=not expanded,
            )
        )

    return f"\n".join(sections)
