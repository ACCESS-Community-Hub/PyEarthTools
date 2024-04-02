# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

import torch
from piqa import SSIM

import einops


class SSIMLoss(torch.nn.Module):
    """
    Uses `piqa.SSIM` to create a structural similarity score, then convert to loss

    See [piqa.SSIM][https://piqa.readthedocs.io/en/stable/api/piqa.ssim.html#piqa.ssim.SSIM]

    ```
    ssim = SSIM(output, target)
    loss = 1 - ssim
    return loss
    ```

    !!! Warning
        If used on 4D + batch data, each 3D slice as determined by the second dimension will be calculated
        then averaged.

    """

    def __init__(self, normalise: bool = False, format: str | None = None, **ssim_kwargs: dict) -> None:
        """
        Create SSIM Loss

        Args:
            normalise (bool, optional):
                Whether to force the data to be between 0 and 1. Defaults to False.
            format (str, optional):
                Format of data if not B T C H W. Defaults to None.
            **ssim_kwargs (Any, optional):
                All kwargs passed to [piqa.SSIM][https://piqa.readthedocs.io/en/stable/api/piqa.ssim.html#piqa.ssim.SSIM]

        !!! Tip
            Useful kwargs for piqa.SSIM
            | kwarg | Description |
            | ----- | ----------- |
            | window_size (int) | The size of the window. |
            | sigma (float) | The standard deviation of the window.|
            | n_channels (int) | The number of channels |
            | reduction (str) | Specifies the reduction to apply to the output: 'none', 'mean' or 'sum'.|
        """
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
