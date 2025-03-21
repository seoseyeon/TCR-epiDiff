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
with open("/home/seri9148/seyeon_project/TCR_generation/data/final_data/tcr_epitope_peptide.pkl", "rb") as file:
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

#### training Model ####
original_TCR = []
denoised_visual_result = []
real_epitope_result = []
denoised_data_result = []
loss_save = []
model = UNet1D(in_channels=4, out_channels=4)
result = train_ddpm_with_schedule("/Best_model.pth")

#### Covid-19 Data External Valid ####
with open("/tcr_epitope_peptide.pkl", "rb") as file:
    Valid_data = pickle.load(file)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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