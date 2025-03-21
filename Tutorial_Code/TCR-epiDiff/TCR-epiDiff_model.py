import os
import random
import pickle
import inspect
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.layers import Layer, LayerNormalization
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split


#### TCR-epiDiff ####
# Define noise scheduling variables
def get_alpha_schedule(beta_start, beta_end, time_steps):
    beta = np.linspace(beta_start, beta_end, time_steps)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)  # Cumulative product
    sqrt_alpha_bar = np.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar)
    return beta, alpha, alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bnorm = nn.BatchNorm1d(out_channels)

        # Residual layer
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        # Time embedding layers
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        # epitope
        self.epitope_mlp = nn.Linear(1024, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t_emb, epitope=None):
        h = self.bnorm(self.relu(self.conv(x)))

        # Time embedding
        time_emb = self.relu(self.time_mlp(t_emb))
        time_emb = time_emb.unsqueeze(-1)

        # Processing Epitope
        if epitope is not None:
            epitope_emb = self.relu(self.epitope_mlp(epitope))
            epitope_emb = epitope_emb.unsqueeze(-1)
            h = h + epitope_emb

        # Add time embedding to feature map
        h = h + time_emb

        return self.relu(h + self.residual(x))
    
class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(UNet1D, self).__init__()

        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder
        self.encoder = nn.ModuleList([
            ResidualBlock(in_channels, 64, time_emb_dim),
            ResidualBlock(64, 128, time_emb_dim),
            ResidualBlock(128, 256, time_emb_dim),
            ResidualBlock(256, 512, time_emb_dim),
            ResidualBlock(512, 1024, time_emb_dim)
        ])

        # Decoder
        self.decoder = nn.ModuleList([
            ResidualBlock(1024, 512, time_emb_dim),
            ResidualBlock(512, 256, time_emb_dim),
            ResidualBlock(256, 128, time_emb_dim),
            ResidualBlock(128, 64, time_emb_dim)
        ])

        # Final output
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x, t, epitope):
        t = t.float()

        # Time embedding
        t_emb = self.time_emb(t.unsqueeze(-1))

        # Encoder (skip connection)
        encoder_outputs = []
        latent_outputs = []  # List to store intermediate layer outputs

        for i, block in enumerate(self.encoder):
            x= block(x, t_emb, epitope)
            encoder_outputs.append(x)
            if i == 3:  # Extract intermediate layers from ResidualBlock(512, 1024, time_emb_dim)
                latent_outputs.append(x)

        # Decoder
        for i, block in enumerate(self.decoder):
            if i < len(encoder_outputs):
                x = block(x + encoder_outputs[-(i + 1)], t_emb, epitope)  # Pass epitope to the decoder with skip connection
            else:
                x = block(x, t_emb, epitope)

        # Final output
        return self.final_conv(x), latent_outputs

    
# Noise addition function
def add_noise(x_0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar):
    """
    Noise addition formula: x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    """
    noise = torch.randn_like(x_0)
    sqrt_alpha_bar_t = sqrt_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)

    return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise


def get_denoised_data(noisy_data, predicted_noise, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar):
    """
    Noise removal: x_0 = (x_t - sqrt(1 - alpha_t) * predicted_noise) / sqrt(alpha_t)
    """
    sqrt_alpha_bar_t = sqrt_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
    denoised_data = (noisy_data - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
    return denoised_data

    
def train_ddpm_with_schedule(model, dataloader, num_epochs, lr=0.001, time_steps=10, beta_start=0.0001, beta_end=0.1, 
                             device=None, save_path="best_model.pth", patience=5, min_delta=0.001):
    beta, alpha, alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar = get_alpha_schedule(beta_start, beta_end, time_steps)
    sqrt_alpha_bar = torch.tensor(sqrt_alpha_bar, dtype=torch.float32).to(device)
    sqrt_one_minus_alpha_bar = torch.tensor(sqrt_one_minus_alpha_bar, dtype=torch.float32).to(device)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        latent_output_result = []

        for batch_data in dataloader:
            TCR = batch_data[0].to(device)
            epitope = batch_data[1].to(device)
            real_epitope = batch_data[3]

            for t in range(time_steps):
                batch_latent = []
                t_tensor = torch.tensor([t] * TCR.size(0), dtype=torch.long).to(device)
                noisy_data, noise = add_noise(TCR, t_tensor, sqrt_alpha_bar, sqrt_one_minus_alpha_bar)

                predicted_noise, latent_output = model(noisy_data, t_tensor, epitope)
                # Denoised data 계산
                denoised_data = get_denoised_data(noisy_data, predicted_noise, t_tensor, sqrt_alpha_bar, sqrt_one_minus_alpha_bar)
                loss = criterion(predicted_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            if epoch == num_epochs - 1:
                latent_output_result.append(latent_output)

            original_TCR.append(TCR)
            denoised_visual_result.append(denoised_data)
            real_epitope_result.append(real_epitope)
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_epoch_loss)
        denoised_data_result.append(denoised_data[-1])
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        # Early Stopping check
        if avg_epoch_loss < best_loss - min_delta:
            best_loss = avg_epoch_loss
            early_stop_counter = 0
            torch.save(model, save_path)
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"Training stopped at epoch {epoch+1} due to early stopping.")
            break
    # Model save
    torch.save(model, save_path)
    print(f"Model saved at {save_path}")
    
    # Loss Graph
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

    return latent_output_result

def denoise_with_trained_model(model, time_steps, denoised_time, 
                               sqrt_alpha_bar, sqrt_one_minus_alpha_bar, device):
    
    # validation
    model.eval() 
    denoised_data = []
    real_epitope_list = []
    original_TCR = []
    

    # Disable gradients
    with torch.no_grad():
        
        for batch_data_each in valid_data:
            TCR = batch_data_each[0].to(device)
            epitope = batch_data_each[1].to(device)
            real_epitope = batch_data_each[3]
            
            ## save list
            original_TCR.append(TCR)
            real_epitope_list.append(real_epitope)
    
            for t in range(time_steps):
                t_tensor = torch.tensor([t] * TCR.size(0), dtype=torch.long).to(device)
                noisy_data, noise = add_noise(TCR, t_tensor, sqrt_alpha_bar, sqrt_one_minus_alpha_bar)

            iteration = 0
            while iteration < denoised_time:
                predicted_noise, latent_output = model(noisy_data, t_tensor, epitope)
                x_t = get_denoised_data(noisy_data, predicted_noise, t_tensor, 
                                        sqrt_alpha_bar, sqrt_one_minus_alpha_bar)
                denoised_data.append(x_t)
                noisy_data = x_t
                t_tensor = t_tensor - 1

                if torch.any(t_tensor < 0):
                    break
                iteration += 1

    return denoised_data, original_TCR, real_epitope_list 