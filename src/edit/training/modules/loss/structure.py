import torch


class SSIMLoss(torch.nn.MSELoss):
    def __init__(self, **ssim_kwargs) -> None:
        from skimage.metrics import structural_similarity as ssim
        self.ssim = ssim
        self.ssim_kwargs = ssim_kwargs

    def forward(self, output, target):
        loss = self.ssim(output, target, **self.ssim_kwargs)
        return 1 - loss
