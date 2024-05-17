# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import warnings

try:
    from edit.training.modules import loss
except ImportError as e:
    warnings.warn(f"Unable to import loss functions because of {e}", ImportWarning)


from edit.training.modules.loss import get_loss
