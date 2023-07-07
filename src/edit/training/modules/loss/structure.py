import torch
from piqa import SSIM

import einops


class SSIMLoss(torch.nn.Module):
    """
    Uses piqa.SSIM to create a structural similarity score, then convert to loss

    Refer to [piqa.SSIM][https://piqa.readthedocs.io/en/stable/api/piqa.ssim.html#piqa.ssim.SSIM] for kwargs
    ```
    ssim = SSIM(output, target)
    loss = 1 - ssim
    return loss
    ```

    !!! Warning
        If used on 4D + batch data, each 3D slice as determined by the second dimension will be calculated
        then averaged.
    """

    def __init__(
        self, normalise: bool = False, format: str = None, **ssim_kwargs: dict
    ) -> None:
        super().__init__()
        self.ssim = SSIM(**ssim_kwargs)
        self.normalise = normalise

        if isinstance(format, str):
            format = format.replace("N ", "")
        self.format = format

    def rearrange(self, output, target):
        if self.format is None:
            return output, target

        if len(output.shape) > 4:
            pattern = f"N {self.format} -> N T C H W"
        else:
            pattern = f"N {self.format} -> N C H W"

        rearr_func = lambda x: einops.rearrange(x, pattern)
        return tuple(map(rearr_func, (output, target)))

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.normalise:
            min_value = torch.Tensor([0]).to(output)
            max_value = torch.Tensor([1]).to(output)
            output = torch.minimum(torch.maximum(output, min_value), max_value)
            target = torch.minimum(torch.maximum(target, min_value), max_value)

        output, target = self.rearrange(output, target)

        if len(output.shape) > 4:
            loss = torch.stack(
                [self.ssim(output[:, i], target[:, i]) for i in range(output.shape[1])],
                dim=-1,
            )
            loss = loss.mean(dim=-1)
        else:
            loss = self.ssim(output, target)
        return 1 - loss
