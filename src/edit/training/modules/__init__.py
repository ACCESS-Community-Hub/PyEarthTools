
import warnings

try:
    from edit.training.modules import loss
except ImportError as e:
    warnings.warn(f"Unable to import loss functions because of {e}", ImportWarning)


from edit.training.modules.loss import get_loss
