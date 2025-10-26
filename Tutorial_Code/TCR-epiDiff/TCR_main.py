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
from TCR-epiDiff_model import *

#### Load Data ####
with open("/tcr_epitope_peptide.pkl", "rb") as file:
    Train_data = pickle.load(file)

BATCH_SIZE = 64
trainloader = DataLoader(Train_data, batch_size=BATCH_SIZE, shuffle=False)

#### Setting Seed ####
def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
set_seed(42)
BATCH_SIZE = 64

# load
with open("/CDR3_epi_onehot_emb_covid.pkl", "rb") as file:
    valid = pickle.load(file)
validloader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
## data loading ##
batch_data = next(iter(validloader)) 

#### training Model ####
original_TCR = []
denoised_visual_result = []
real_epitope_result = []
denoised_data_result = []
loss_save = []
model = UNet1D(in_channels=4, out_channels=4)

#### Covid-19 Data External Valid ####
with open("/tcr_epitope_peptide.pkl", "rb") as file:
    Valid_data = pickle.load(file)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_denoised_data(noisy_data, predicted_noise, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar):
    sqrt_alpha_bar_t = sqrt_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]

    # Denoised data
    denoised_data = (noisy_data - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
    return denoised_data

def get_alpha_schedule(beta_start, beta_end, time_steps):
    beta = np.linspace(beta_start, beta_end, time_steps)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    sqrt_alpha_bar = np.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar)
    return beta, alpha, alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar

def add_noise(x_0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar):
    noise = torch.randn_like(x_0)
    sqrt_alpha_bar_t = sqrt_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]

    return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise

def denoise_with_trained_model(model, time_steps, denoised_time, 
                               sqrt_alpha_bar, sqrt_one_minus_alpha_bar, device):
    model.eval()

    denoised_data = []
    noised_data_list = []
    epitope_list = []
    original_tcr_list = []

    with torch.no_grad():
        TCR = original_TCR[0].to(device)
        epitope = batch_data[1].to(device)
        original_tcr_list.append(batch_data[0])
        epitope_list.append(batch_data[3])
        
        for t in range(time_steps):
            t_tensor = torch.tensor([t] * TCR.size(0), dtype=torch.long).to(device)
            noisy_data, noise = add_noise(TCR, t_tensor, sqrt_alpha_bar, sqrt_one_minus_alpha_bar)
            noised_data_list
        
        t_tensor = torch.tensor([time_steps - 1] * TCR.size(0), dtype=torch.long).to(device)
        noisy_data, _ = add_noise(TCR, t_tensor, sqrt_alpha_bar, sqrt_one_minus_alpha_bar)

        for d in range(denoised_time):
            predicted_noise, latent_output = model(noisy_data, t_tensor, epitope)

            x_t = get_denoised_data(noisy_data, predicted_noise, t_tensor, 
                                    sqrt_alpha_bar, sqrt_one_minus_alpha_bar)

            denoised_data.append(x_t)

            noisy_data = x_t

            t_tensor = t_tensor - 1

            if torch.any(t_tensor < 0):
                break

    return denoised_data, epitope_list, original_tcr_list

# Schedule variables
beta, alpha, alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar = get_alpha_schedule(0.0001, 0.02, 10)
sqrt_alpha_bar = torch.tensor(sqrt_alpha_bar, dtype=torch.float32).to(device)
sqrt_one_minus_alpha_bar = torch.tensor(sqrt_one_minus_alpha_bar, dtype=torch.float32).to(device)

denoised_data_batch, noisy_data_batch = denoise_with_trained_model(model=model,
                                                                    time_steps=10,
                                                                    denoised_time=10,
                                                                    sqrt_alpha_bar=sqrt_alpha_bar,
                                                                    sqrt_one_minus_alpha_bar=sqrt_one_minus_alpha_bar,
                                                                    device=device
                                                                    )