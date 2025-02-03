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


from __future__ import annotations

import uuid
from functools import lru_cache, partial
from html import escape
from importlib.resources import read_binary

STATIC_FILES = (
    ("pyearthtools.utils.repr_utils.static.html", "icons-svg-inline.html"),
    ("pyearthtools.utils.repr_utils.static.css", "style.css"),
)


@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed"""
    return [read_binary(package, resource).decode("utf-8") for package, resource in STATIC_FILES]


def _collapsible_section(
    name,
    sub_name="",
    inline_details="",
    details="",
    n_items=None,
    enabled=True,
    collapsed=False,
):
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())

    has_items = n_items is not None and n_items
    n_items_span = "" if n_items is None else f" <span>({n_items})</span>"
    enabled = "" if enabled and (details or has_items) else "disabled"
    collapsed = "" if collapsed else "checked"
    tip = " title='Expand/collapse section'" if enabled else ""

    return_str = (
        f"<input id='{data_id}' class='object-section-summary-in' "
        f"type='checkbox' {enabled} {collapsed}>"
        f"<label for='{data_id}' class='object-section-summary' {tip}>"
        f"{name}:{n_items_span}</label>"
        f"<div class='object-section-inline-details'>{escape(sub_name)}</div>"
        f"<div class='object-section-inline-details'>{inline_details}</div>"
    )
    return_str += f"<div class='object-section-details'>{details}</div>" if details else ""
    return return_str


def _icon(icon_name):
    # icon_name should be defined in /static/html/icon-svg-inline.html
    return "<svg class='icon object-{0}'>" "<use xlink:href='#{0}'>" "</use>" "</svg>".format(icon_name)


def _obj_repr(header_components, sections, backup_repr):
    header = f"<div class='object-header'>{''.join(h for h in header_components)}</div>"
    sections = "".join(f"<li class='object-section-item'>{s}</li>" for s in sections)

    icons_svg, css_style = _load_static_files()

    return (
        "<div>"
        f"{icons_svg}<style>{css_style}</style>"
        f"<pre class='object-text-repr-fallback'>{escape(backup_repr)}</pre>"
        "<div class='object-wrap' style='display:none'>"
        f"{header}"
        f"<ul class='object-sections'>{sections}</ul>"
        "</div>"
        "</div>"
    )


def summarise_argument(name, var, expand: bool = True):
    # cssclass_idx = " class='object-has-index'" if is_index else ""
    # dims_str = f"({', '.join(escape(dim) for dim in var.dims)})"
    name = escape(str(name))
    dtype = escape(f"({str(type(var).__name__)})")

    # "unique" ids required to expand/collapse subsections
    # attrs_id = "attrs-" + str(uuid.uuid4())
    data_id = "data-" + str(uuid.uuid4())
    # disabled = "" if len(var.attrs) else "disabled"

    preview = escape(f"{var!r}")
    # attrs_ul = summarize_attrs(var.attrs)
    data_repr = var  # short_data_repr_html(variable)
    if isinstance(data_repr, dict):
        data_repr = "<br>".join(f"{key} : {val!s}" for key, val in data_repr.items())
    elif isinstance(data_repr, (list, tuple)):
        data_repr = "<br>".join(f" - {val!s}" for val in data_repr)
    else:
        data_repr = repr(data_repr).replace("\n", "<br>")

    # attrs_icon = _icon("icon-file-text2")
    data_icon = _icon("icon-storm")

    if isinstance(var, str) and "http" in var:
        return_str = (
            f"<div class='object-var-name'><span>{name}</span></div>"
            # f"<div class='object-var-dtype'>(link)</div>"
            f"<div class='object-var-preview'><a href='{var}'>{var}</a></div>"
        )
    else:
        return_str = (
            f"<div class='object-var-name'><span>{name}</span></div>"
            # f"<div class='object-var-dtype'>{dtype}</div>"
            f"<div class='object-var-preview object-preview'>{preview}</div>"
        )
    if expand:
        return_str += (
            f"<input id='{data_id}' class='object-var-data-in' type='checkbox'>"
            f"<label for='{data_id}' title='Show/Hide data repr'>"
            f"{data_icon}</label>"
            f"<div class='object-var-data'>{data_repr}</div>"
        )

    return return_str


def summarise_kwargs(variables, expand: bool = True):
    li_items = []
    if not isinstance(variables, dict):
        return li_items

    for k, v in variables.items():
        li_content = summarise_argument(k, v, expand=expand)
        li_items.append(f"<li class='object-var-item'>{li_content}</li>")

    vars_li = "".join(li_items)

    return f"<ul class='object-var-list'>{vars_li}</ul>"


def provide_html(
    *objects,
    name: str = None,
    description: dict = None,
    documentation_attr: str = None,
    info_attr: str = None,
    name_attr: str = None,
    backup_repr="Failed to create HTML repr",
    expanded: bool = None,
) -> str:
    """Create a html_repr from a list of objects.

    Formatted like the xarray repr, with docstrings or documentation_attr being retrieved

    Args:
        name (str, optional):
            Name of overall repr. Defaults to None.
        documentation_attr (str, optional):
            Attribute to retrieve as documentation. Defaults to None.
        backup_repr (str, optional):
            If HTML repr fails, fail over repr. Defaults to "Failed to create HTML repr".

    Returns:
        (str):
            HTML repr of objects
    """
    header_components = [
        f"<div class='object-name'><h2>{name}</h2></div>",
    ]
    if description:
        component = _collapsible_section(
            "Description",
            sub_name=description.pop("singleline", ""),
            details=summarise_kwargs(description, expand=False) if description else None,
            enabled=True,
            collapsed=True,
        )
        header_components.append(f"<li class='object-section-item'>{component}</li>")

    expanded = expanded or len(objects) < 4
    sections = []

    for item in objects:
        inline_details = getattr(item, documentation_attr, getattr(item, "__doc__", None))

        inline_details = "" if inline_details is None else str(inline_details)
        inline_details = inline_details.split("\n")[0]

        details = getattr(item, info_attr, None) if info_attr else None

        name = getattr(item, name_attr) if name_attr else item.__class__.__name__

        sections.append(
            _collapsible_section(
                name,
                sub_name=inline_details,
                details=summarise_kwargs(details) if details else None,
                n_items=None,
                enabled=True,
                collapsed=not expanded,
            )
        )

    if len(sections) == 0:
        sections = [_collapsible_section("Empty", sub_name="No Items supplied", enabled=False, collapsed=True)]

    return _obj_repr(header_components, sections, backup_repr=backup_repr)
