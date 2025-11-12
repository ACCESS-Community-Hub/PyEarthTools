# MIT License

# Copyright (c) 2025 ISCLPennState

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from math import ceil
import torch

# import torch_harmonics as th
# import torch_harmonics.distributed as thd

# from torch_harmonics import *
import torch.fft
from tqdm import tqdm

import torch

from torch.utils.data import TensorDataset, DataLoader

from lucie.torch_harmonics_local import *

from torch.optim.lr_scheduler import CosineAnnealingLR

from lucie import inference


def integrate_grid(
    ugrid,
    nlon,
    quad_weights,
    dimensionless=False, polar_opt=0):

    dlon = 2 * torch.pi / nlon
    radius = 1 if dimensionless else radius
    if polar_opt > 0:
        out = torch.sum(
            ugrid[..., polar_opt:-polar_opt, :] * quad_weights[polar_opt:-polar_opt] * dlon * radius**2, dim=(-2, -1)
        )
    else:
        out = torch.sum(ugrid * quad_weights * dlon * radius**2, dim=(-2, -1))
    return out


def l2loss_sphere(prd, tar, nlon, quad_weights, relative=False, squared=True):
    loss = integrate_grid((prd - tar) ** 2, nlon, quad_weights, dimensionless=True).sum(dim=-1)
    if relative:
        loss = loss / integrate_grid(tar**2, nlon, quad_weights, dimensionless=True).sum(dim=-1)

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss


def train_model(
    device,
    model,
    train_loader,
    val_loader,
    optimizer,
    data_inp=None,
    prog_means=None,
    prog_stds=None,
    diag_means=None,
    diag_stds=None,
    diff_stds=None,
    nlon=96,
    scheduler=None,
    nepochs=20,
    quad_weights=None,
    true_clim=None,
    nfuture=0,
    num_examples=256,
    num_valid=8,
    reg_rate=0,
    ):
    '''
    Train your own weights for the LUCIE model
    '''

    infer_bias = 1e80
    recall_count = 0

    debug_sample_limit = 5

    print("Starting Training")
    for epoch in tqdm(range(nepochs)):


        if epoch < 149:
            if scheduler is not None:
                scheduler.step()
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-6

        optimizer.zero_grad()

        model.train()

        batch_num = 0

        zz = 0

        for inp, tar in train_loader:
            batch_num += 1
            loss = 0

            zz += 1
            if zz > debug_sample_limit:
                break

            inp = inp.to(device)
            tar = tar.to(device)
            prd = model(inp)

            loss_delta = l2loss_sphere(prd[:, :5, :, :], tar[:, :5, :, :], nlon, quad_weights, relative=True)
            loss_tp = torch.mean((prd[:, 5:, :, :] - tar[:, 5:, :, :]) ** 2)
            loss = loss_delta + loss_tp / tar.shape[1]

            lat_index = np.r_[7:15, 32:40]
            # lat_index = np.r_[0:48]
            # quad_weight_reg = quad_weights.reshape(1,1,48,1)[:,:,lat_index,:]
            out_fft = torch.mean(torch.abs(torch.fft.rfft(prd[:, :, lat_index, :], dim=3)), dim=2)
            target_fft = torch.mean(torch.abs(torch.fft.rfft(tar[:, :, lat_index, :], dim=3)), dim=2)
            loss_reg = 0.05 * torch.mean(torch.abs(out_fft - target_fft))

            if epoch > 150:
                loss = loss + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            # rollout_steps = 2920
            rollout_steps = 50
            rollout = torch.tensor(
                inference.infer(
                    model,
                    rollout_steps,
                    data_inp[0:1].to(device),
                    data_inp[:1460, -2:].to(device),
                    1,
                    prog_means,
                    prog_stds,
                    diag_means,
                    diag_stds,
                    diff_stds,
                )
            ).to(device)
            rollout_clim = torch.mean(rollout[1460:], dim=0)
            clim_bias = torch.mean(torch.abs(rollout_clim - true_clim))
            print("2 year rollout bias", clim_bias)
            if epoch > 60:
                if clim_bias <= infer_bias:
                    infer_bias = clim_bias
                    torch.save(model.state_dict(), "regular_training_checkpoint.pth")
                    recall_count = 0
                else:
                    state_pth = torch.load("regular_training_checkpoint.pth")
                    model.load_state_dict(state_pth)
                    recall_count += 1
                    if recall_count > 3:
                        break



def load_data_and_train(
    device,
    regridded_data,
    preprocessed_data,
    *,
    ntrain: int | None = 16000,
    nval: int | None = 100):
    '''

    args:
        unprocessed_data
        reprocessed_data: dictionary or numpy collection containing 'diagn_means', 'diag_stds', 'diff_means' and 'diff_stds'

    '''

    regridded_data = regridded_data[..., :6]
    true_clim = torch.tensor(np.mean(regridded_data, axis=0)).to(device).permute(2, 0, 1)

    data = preprocessed_data  # dictionary-like numpy array
    data_inp = torch.tensor(data["data_inp"], dtype=torch.float32)  # input data
    data_tar = torch.tensor(data["data_tar"], dtype=torch.float32)
    raw_means = torch.tensor(data["raw_means"], dtype=torch.float32).reshape(1, -1, 1, 1).to(device)
    raw_stds = torch.tensor(data["raw_stds"], dtype=torch.float32).reshape(1, -1, 1, 1).to(device)
    prog_means = raw_means[:, :5]
    prog_stds = raw_stds[:, :5]
    diag_means = torch.tensor(data["diag_means"], dtype=torch.float32).reshape(1, -1, 1, 1).to(device)
    diag_stds = torch.tensor(data["diag_stds"], dtype=torch.float32).reshape(1, -1, 1, 1).to(device)
    diff_means = torch.tensor(data["diff_means"], dtype=torch.float32).reshape(1, -1, 1, 1).to(device)
    diff_stds = torch.tensor(data["diff_stds"], dtype=torch.float32).reshape(1, -1, 1, 1).to(device)

    train_set = TensorDataset(data_inp[:ntrain], data_tar[:ntrain])
    val_set = TensorDataset(data_inp[ntrain : ntrain + nval], data_tar[ntrain : ntrain + nval])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)


    grid = "legendre-gauss"
    nlat = 48
    nlon = 96
    hard_thresholding_fraction = 0.9
    lmax = ceil(nlat / 1)
    mmax = lmax
    modes_lat = int(nlat * hard_thresholding_fraction)
    modes_lon = int(nlon // 2 * hard_thresholding_fraction)
    modes_lat = modes_lon = min(modes_lat, modes_lon)
    # sht = RealSHT(nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid=grid, csphase=False)
    radius = 6.37122e6
    _cost, quad_weights = legendre_gauss_weights(nlat, -1, 1)
    quad_weights = (torch.as_tensor(quad_weights).reshape(-1, 1)).to(torch.float32).to(device)  # mps only supports float32, todo only do this if mps

    model = SphericalFourierNeuralOperatorNet(
        params={},
        spectral_transform="sht",
        filter_type="linear",
        operator_type="dhconv",
        img_shape=(48, 96),
        num_layers=8,
        in_chans=7,
        out_chans=6,
        scale_factor=1,
        embed_dim=72,
        activation_function="silu",
        big_skip=True,
        pos_embed="latlon",
        use_mlp=True,
        normalization_layer="instance_norm",
        hard_thresholding_fraction=hard_thresholding_fraction,
        mlp_ratio=2.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)
    train_model(device, model, train_loader, val_loader, optimizer,
        prog_means=prog_means,
        prog_stds=prog_stds,
        diag_means=diag_means,
        diag_stds=diag_stds,
        diff_stds=diff_stds,
        true_clim=true_clim,
     # data_inp=data_inp, nlon=nlon, quad_weights=quad_weights, scheduler=scheduler, nepochs=500)
     data_inp=data_inp, nlon=nlon, quad_weights=quad_weights, scheduler=scheduler, nepochs=5)
    torch.save(model.state_dict(), "model.pth")
