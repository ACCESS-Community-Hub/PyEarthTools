"""Utilities for NCI indexes"""
from __future__ import annotations

from pathlib import Path

JOIN_LINK = "https://my.nci.org.au/mancini/project/{code}/join"

def check_project(project_code: str = None):
    """
    Check project code data existance.
    """

    default_root_path = Path('/g/data/')

    project_code = str(project_code)
    if '/scratch' in project_code:
        default_root_path = Path('/scratch/')

    project_code = project_code.replace('/g/data/', '').replace('/scratch/', '').split('/')[0]
    
    if not(default_root_path / project_code).exists():
        raise FileNotFoundError(
            f"Could not find data path for {project_code}."
            "\nTherefore no data can be loaded from this index."
            f"\nJoin this project at {JOIN_LINK.format(code = project_code)}"
            )
