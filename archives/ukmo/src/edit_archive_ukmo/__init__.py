"""
UKMO Archive

Implemented:

| Name | Description |
| ---- | ----------- |
| UnifiedModel | Specific UM index for MASS |
| MASS | Generic MASS superclass |

"""
import edit.data
from edit.data.archive import register_archive

import edit_archive_ukmo

ROOT_DIRECTORIES = {
    "UM": "moose:/opfc/atm/global/prod{spec}",
    "UMProcessed": "moose:/opfc/atm/global/lev1/",
}

register_archive('ROOT_DIRECTORIES')(ROOT_DIRECTORIES)

from edit_archive_ukmo.mass import MASS
from edit_archive_ukmo.um import UnifiedModel

register_archive('UKMO')(edit_archive_ukmo)

__all__ = ['UnifiedModel']