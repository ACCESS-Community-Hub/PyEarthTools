import pyearthtools.data as petdata
import pyearthtools.pipeline as petpipe
import site_archive_nci
from pyearthtools.data.time import Petdt
from pyearthtools.pipeline.operations.xarray.join import GeospatialTimeSeriesMerge
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pyearthtools.training
import pyearthtools.training as training
import warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", message="In a future version of xarray")  
warnings.filterwarnings("ignore", message="Engine 'kerchunk' loading failed")
import site_archive_nci
import pyearthtools.pipeline as petpipe
from pathlib import Path
import numpy as np
import os
import csv
import torch.optim as optim

from PyEarthTools import PassthroughModel



class SimpleAutoEncoder(PassthroughModel, nn.Module):
    def __init__(self, input_channels, layers, height, width):

        def computeOutput(dim, kernel_size, stride, padding):
            out = (dim + 2 * padding - (kernel_size - 1) - 1) / stride + 1
            return int(out) #This computeOutput function is used to account for the reduced spatial dimension produced after a convolution. 

        nn.Module.__init__(self)
        PassthroughModel.__init__(self)

        self.input_channels = input_channels
        self.channels = [input_channels]
        self.width = width
        self.height = height
        spatial_dims = [(height, width)]

        kernel_size = 3
        stride = 2
        padding = 1

        for i in range(1, layers + 1):
            self.channels.append(16 * i) 
            
            '''
            Each layer will increase the output channels by a factor of 16, starting at the number of channels being plugged into the model. In a model with 1 input_channels
            and 4 layers the list of channels will be [1, 16, 32, 64, 128]. Each comment from this point forward will use these input parameters as an example. 
            '''
             

        encoder_layers = []
        num_pairs = len(self.channels) - 1
        #There will be 4 pairs(input channel --> output channel): (1 --> 16), (16 --> 32), (32 --> 64), (64 --> 128)


        for _ in range(num_pairs):
            prev_h, prev_w = spatial_dims[-1]
            new_h = computeOutput(prev_h, kernel_size, stride, padding)
            new_w = computeOutput(prev_w, kernel_size, stride, padding)
            #Based on the input a new height and width of the data is computed to account for the reduced spatial dimensions outputted by the convolution. 
            if new_h <= 0 or new_w <= 0:
                raise ValueError(
                    f"Too many layers ({layers}) for input size ({height}x{width}); "
                    f"spatial dims collapsed to {new_h}x{new_w}. Reduce `layers`."
                )
            spatial_dims.append((new_h, new_w))
            #As a safety measure the script will raise an error to inform you that the data is too small to be reduced by the number of layers inputted.

        for i in range(num_pairs):
            #The code in this for loop is simply adding the steps the inputted data will go through in the encoding phase. These steps are stored in the encoder_layers variable
            chan_in = self.channels[i]
            chan_out = self.channels[i + 1]
            #Chan in and out are assigned to the pairs aforementioned: (1 --> 16), (16 --> 32), (32 --> 64), (64 --> 128) 
            encoder_layers.append(
                nn.Conv2d(chan_in, chan_out, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            #This is the convolution that reduces spatial dimensions but increases channel size. 
            if i < num_pairs - 1:
                encoder_layers.append(nn.ReLU())
                #ReLU is used to introduce nonlinearity because, without an activation function, stacking convolutional layers would still amount to essentially one big linear transformation which is not good
                #for learning
        self.encoder = nn.Sequential(*encoder_layers)
        # Here the encoder is assigned to the unpackaged list of steps. 
        
        reversed_channels = self.channels[::-1]
        reversed_sd = spatial_dims[::-1]
        '''
        Since the decoder reverses the steps of the encoder the list of channels and spatial dimensions are reversed. 
        Channels: (128 → 64), (64 → 32), (32 → 16), (16 → 1) and their respective spatial dimensions to recontruct the data in it's original form. 
        The list of spatial dimensions are used here to account for the needed output_padding of each convolutional transpose computation 
        to restorethe data's orginal spatial dimensions. The decoder layers act similarly to the encoder layers list storing the steps 
        needed to complete each process before being assigned to the decoder and econder variables.
        '''


        decoder_layers = []
        for i in range(num_pairs):
            chan_in = reversed_channels[i]
            chan_out = reversed_channels[i + 1]

            in_h, in_w = reversed_sd[i]
            out_h, out_w = reversed_sd[i + 1]

            base_h = (in_h - 1) * 2 - 2 * 1 + kernel_size
            base_w = (in_w - 1) * 2 - 2 * 1 + kernel_size  # output hxw after ConvTranspose2d
            output_padding = (out_h - base_h, out_w - base_w)  # correct for rounding mismatch
            decoder_layers.append(
                nn.ConvTranspose2d(
                    chan_in, chan_out, kernel_size=3, stride=2, padding=1,
                    output_padding=output_padding,
                )
            )
            if i < num_pairs - 1:
                decoder_layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers) 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3) # Updates the model's weights to reduce the error
        self.criterion = nn.MSELoss()  # Measures the difference between the original and reconstructed data

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


    def _run_epoch(self, pipeline, batch_size, max_samples, device, train):
        #This is a helper method that collects a number of samples according to the inputed batch_size and handles traning and validation phases
        
        self.train() if train else self.eval()

        iterator = iter(pipeline)
        buffer = []
        total_loss = 0.0
        loss_count = 0
        epoch_samples = 0
        print_per = 500
        stop_epoch = False

        while not stop_epoch:
            while len(buffer) < batch_size:
                try:
                    raw = next(iterator)
                except StopIteration:
                    print('stop - end of pipeline')
                    stop_epoch = True
                    break
                except Exception as e:
                    print(f'error - skipping sample: {e}')
                    continue

                buffer.append(raw)

            if not buffer:
                break  # nothing left to process

            sample = np.concatenate(buffer, axis=0) if len(buffer) > 1 else buffer[0]
            buffer = []
            x = torch.from_numpy(sample).float().to(device)

            if torch.any(torch.isnan(x)):
                continue

            actual_batch_size = x.shape[0]
            epoch_samples += actual_batch_size

            if epoch_samples % print_per < actual_batch_size:
                print(epoch_samples)

            if train:
                self.optimizer.zero_grad()
                y = self.forward(x)
                loss = self.criterion(y, x)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    y = self.forward(x)
                    loss = self.criterion(y, x)

            total_loss += loss.item()
            loss_count += actual_batch_size

            if max_samples is not None and epoch_samples >= max_samples:
                print(f'reached max_samples={max_samples}, ending epoch early')
                stop_epoch = True

        avg_loss = (total_loss / loss_count) if loss_count != 0 else None
        return avg_loss, loss_count

    def fit(self, train_pipeline, epochs, val_pipeline=None, batch_size=8, max_samples= 2000, device='cuda'):
        '''
        Trains the AutoEncoder using the provided training data.

        Parameters:
        - train_pipeline: Provides the training data used to update the model.
        - epochs: Number of times the model will train over the dataset.
        - val_pipeline: Optional validation data used to evaluate the model
          without updating its weights.
        - batch_size: Number of samples processed before updating the model's weights.
        - max_samples: Optional limit on the number of samples processed per epoch.
        - device: Specifies whether the model runs on the CPU or GPU.
    
        During training, the model reconstructs the input, calculates the MSE loss,
        uses backpropagation to calculate gradients, and updates its weights using Adam.
        Validation is performed every 2 epochs when a validation pipeline is provided.
    
        Please note that the following files will be stored in your current working directory:
    
        1. A loss.csv file that records the training and validation loss for each epoch.
    
        2. A checkpoints folder containing three checkpoint files:
           - last_checkpoint.pth: Stores the most recent model and optimizer state,
             allowing training to resume if interrupted.
           - val_checkpoint.pth: Stores the model state from the most recent validation run.
           - best_checkpoint.pth: Stores the model with the lowest validation loss observed.
    
        The checkpoints store both the model and optimizer states so that training can
        be resumed from the most recent checkpoint.
        '''


        
        start_epoch = 0
        self.to(device)  
        csv_file_path = 'loss.csv'

        main_dir = "checkpoints"
        os.makedirs(main_dir, exist_ok=True)
        last_checkpoint_path = os.path.join(main_dir, "last_checkpoint.pth")
        val_checkpoint_path = os.path.join(main_dir, "val_checkpoint.pth")
        best_checkpoint_path = os.path.join(main_dir, "best_checkpoint.pth")
        best_loss = float("inf")

        def save_checkpoint(path, epoch, loss, best_loss=None):
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
                "best_loss": best_loss if best_loss is not None else loss,
            }, path)
        
        if os.path.exists(last_checkpoint_path):
            checkpoint = torch.load(last_checkpoint_path, map_location=device)
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            print(f"Resumed from epoch {start_epoch}, best loss : {best_loss:.4f}")

        if not os.path.exists(csv_file_path) or start_epoch == 0:
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(start_epoch, epochs):
            print(f'current epoch {epoch + 1}')

            avg_loss, loss_count = self._run_epoch(
                train_pipeline, batch_size, max_samples, device, train=True
            )
            if loss_count != 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}')
                save_checkpoint(last_checkpoint_path, epoch, avg_loss, best_loss)
                print(f"Saved checkpoint: {last_checkpoint_path}")
            else:
                print('no samples processed this epoch')

            val_avg_loss = None
            if (epoch + 1) % 2 == 0 and val_pipeline is not None:
                val_avg_loss, val_loss_count = self._run_epoch(
                    val_pipeline, batch_size, max_samples, device, train=False
                )
                if val_loss_count != 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Validation Average Loss: {val_avg_loss:.4f}')
                    save_checkpoint(val_checkpoint_path, epoch, val_avg_loss, best_loss)
                    print(f"Saved checkpoint: {val_checkpoint_path}")
                    if val_avg_loss < best_loss:
                        best_loss = val_avg_loss
                        save_checkpoint(best_checkpoint_path, epoch, best_loss, best_loss)
                        print(f"New best model for this run (val_avg_loss={val_avg_loss:.4f}) saved: {best_checkpoint_path}")
                else:
                    print('no validation samples processed this epoch')

            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch + 1,
                    avg_loss if avg_loss is not None else "",
                    val_avg_loss if val_avg_loss is not None else "",
                ])

    def predict(self, pipeline, device='cuda'):
        checkpoint_path = os.path.join("checkpoints", "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.to(device)
        self.eval()
        pet_iterator = iter(pipeline)
        sample_numpy = next(pet_iterator)
        sample_gpu = torch.from_numpy(sample_numpy).float().to(device)
        with torch.no_grad():
            prediction = self.forward(sample_gpu)
        return prediction.cpu()
