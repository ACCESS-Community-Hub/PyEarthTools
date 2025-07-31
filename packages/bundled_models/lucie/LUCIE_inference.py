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


from torch_harmonics_local import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)


def inference(
    model, steps, initial_frame, forcing, initial_forcing_idx, prog_means, prog_stds, diag_means, diag_stds, diff_stds
):
    inf_data = []
    model.eval()
    with torch.no_grad():
        inp_val = initial_frame
        for i in tqdm(range(steps)):
            forcing_idx = (initial_forcing_idx + i) % 1460  # tisr is repeating and orography is
            previous = inp_val[:, :5, :, :]

            pred = model(inp_val)
            pred[:, :5, :, :] = pred[:, :5, :, :] * diff_stds  # denormalize the predicted tendency

            # demornalzie the previous time step and add to the tendecy to reconstruct the current field
            pred[:, :5, :, :] += previous[:, :5, :, :] * prog_stds + prog_means
            tp_frame = pred[:, 5:, :, :] * diag_stds + diag_means
            # pred_frame += (previous_frame + 1) / 2 * (input_maxs - input_mins) + input_mins
            raw = torch.cat((pred[:, :5, :, :], tp_frame), 1)

            inp_val = (
                raw[:, :5, :, :] - prog_means
            ) / prog_stds  # normalize the current time step for autoregressive prediction
            inp_val = torch.cat((inp_val, forcing[forcing_idx, :, :, :].reshape(1, 2, 48, 96)), dim=1)
            raw = raw.cpu().clone().detach().numpy()
            inf_data.append(raw[0])

    inf_data = np.array(inf_data)
    inf_data[:, 5, :, :] = (
        np.exp(inf_data[:, 5, :, :]) - 1
    ) * 1e-2  # denormalzie precipitation that was normalized in log space
    return inf_data


if __name__ == "__main__":

    data = load_data("era5_T30_regridded.npz")[..., :6]
    true_clim = torch.tensor(np.mean(data, axis=0)).to(device).permute(2, 0, 1)

    data = np.load(
        "era5_T30_preprocessed.npz"
    )  # standardized data with mean and stds generated from dataset_generator.py
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

    grid = "legendre-gauss"
    nlat = 48
    nlon = 96
    hard_thresholding_fraction = 0.9
    lmax = ceil(nlat / 1)
    mmax = lmax
    modes_lat = int(nlat * hard_thresholding_fraction)
    modes_lon = int(nlon // 2 * hard_thresholding_fraction)
    modes_lat = modes_lon = min(modes_lat, modes_lon)
    sht = RealSHT(nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid=grid, csphase=False)
    radius = 6.37122e6
    cost, quad_weights = legendre_gauss_weights(nlat, -1, 1)
    quad_weights = (torch.as_tensor(quad_weights).reshape(-1, 1)).to(device)

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

    path = torch.load("regular_8x72_fftreg_baseline.pth")
    model.load_state_dict(path)

    forcing = data_inp[:1460, -2:]  # repeating tisr and constant oro
    print(forcing.shape)
    rollout_step = 14600
    initial_frame_idx = 16000 + 100
    forcing_initial_idx = (16000 + 100) % 1460 + 1
    rollout = inference(
        model,
        rollout_step,
        data_inp[initial_frame_idx].unsqueeze(0).to(device),
        forcing.to(device),
        forcing_initial_idx,
        prog_means,
        prog_stds,
        diag_means,
        diag_stds,
        diff_stds,
    )
