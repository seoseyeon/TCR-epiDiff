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
from sklearn.metrics import f1_score, accuracy_score
from TCR-epiDiff_model import *
from TCR-epi*BP import *


#### TCR-epi*BP ####
class UNet1DClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2, dropout_prob=0.6):
        super(UNet1DClassifier, self).__init__()
        self.encoder = encoder  # encoder
        self.flatten = nn.Flatten()  # Flatten layer

        # Add HLA information to each Linear layer
        self.fc1 = nn.Linear(512 * 72 + 3, 1024)
        self.fc2 = nn.Linear(1024 + 3, 512)
        self.fc3 = nn.Linear(512 + 3, 128)
        self.fc4 = nn.Linear(128 + 3, 64)
        self.fc5 = nn.Linear(64 + 3, num_classes)

        self.dropout = nn.Dropout(dropout_prob)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()

    def forward(self, x, t, epitope, hla):
        # Pass Encoder
        for layer in self.encoder:
            x = layer(x, t, epitope)

        # Flatten layer
        x = self.flatten(x)

        # 1st Fully Connected Layer
        x = torch.cat([x, hla], dim=1)
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 2nd Fully Connected Layer
        x = torch.cat([x, hla], dim=1)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 3rd Fully Connected Layer
        x = torch.cat([x, hla], dim=1)
        x = self.fc3(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 4st Fully Connected Layer
        x = torch.cat([x, hla], dim=1)
        x = self.fc4(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 5st Fully Connected Layer
        x = torch.cat([x, hla], dim=1)
        x = self.fc5(x)

        return x
    
#### Load TCR-epiDiff Model ####
model = UNet1D(in_channels=4, out_channels=4)
model = torch.load('/TCR-epiDiff_Best_model.pth')
# Load encoder
encoder = model.encoder