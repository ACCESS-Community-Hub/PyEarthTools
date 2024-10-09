# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

import warnings

from pathlib import Path

from edit.utils import initialisation
from edit.data.utils import parse_path


SUFFIX = "edi"


class CatalogMixin(initialisation.InitialisationRecordingMixin):
    def make_catalog(self, *args, **kwargs):
        warnings.warn("`make_catalog` is deprecated, please use `record_initialisation`.")
        return self.record_initialisation(*args, **kwargs)

    # @functools.wraps(initialisation.save)
    def save_index(self, path: Path | str | None = None, **kwargs):
        if path is not None:
            path = parse_path(path)
            if not path.suffix:
                path = path.with_suffix(SUFFIX)
            path = str(path)

        return initialisation.save(self, path, **kwargs)
