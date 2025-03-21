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
from TCR-epiBP import *

#### Load Dataset ####
# Training
with open("/TCR-epiBP_train.pkl", "rb") as file:
    Train = pickle.load(file)
# Test
with open("/TCR-epiBP_test.pkl", "rb") as file:
    Test = pickle.load(file)

BATCH_SIZE = 64
trainloader = DataLoader(Train, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(Test, batch_size=BATCH_SIZE, shuffle=True)

#### Load TCR-epiDiff Model ####
model = UNet1D(in_channels=4, out_channels=4)
model = torch.load('/TCR-epiDiff_Best_model.pth')
# Load encoder
encoder = model.encoder

#### Load TCR-epiBP model ####
classifier = UNet1DClassifier(encoder, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

#### TCR-epiBP Training ####
best_loss = float('inf')
early_stop_count = 0
patience = 5  
train_losses = []
test_losses = []  
f1_scores = []
accuracies = []
num_epoch = 50

for epoch in range(num_epoch):
    classifier.train()
    epoch_loss = 0
    all_labels = []
    all_preds = []

    for batch_data in trainloader:
        TCR = batch_data[0].to(device)
        epitope = batch_data[1].to(device)
        hla = batch_data[2].to(device)
        label = batch_data[3].to(device)

        t_tensor = torch.tensor([1.0] * TCR.size(0), dtype=torch.float)
        t_tensor = t_tensor.unsqueeze(1).expand(-1, 128).to(device)
        output = classifier(TCR, t_tensor, epitope, hla)

        # loss
        loss = criterion(output, label.long())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # prediction save
        _, preds = torch.max(output, 1)  
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    # loss mean
    avg_epoch_loss = epoch_loss / len(trainloader)
    train_losses.append(avg_epoch_loss)

    # F1 score, accuracy
    epoch_f1 = f1_score(all_labels, all_preds)  
    epoch_acc = accuracy_score(all_labels, all_preds)
    f1_scores.append(epoch_f1)
    accuracies.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {avg_epoch_loss:.4f}, F1 Score: {epoch_f1:.4f}, Accuracy: {epoch_acc:.4f}")

    # model save
    torch.save(classifier, f"/TCR-epi*BP_{epoch+1}.pth")
    
    classifier.eval()

    all_test_labels = []
    all_test_preds = []
    test_loss = 0 

    with torch.no_grad():  
        for batch_data in testloader:
            TCR = batch_data[0].to(device)
            epitope = batch_data[1].to(device)
            hla = batch_data[2].to(device)
            label = batch_data[3].to(device)

            t_tensor = torch.tensor([1.0] * TCR.size(0), dtype=torch.float)
            t_tensor = t_tensor.unsqueeze(1).expand(-1, 128).to(device)
            
            output = classifier(TCR, t_tensor, epitope, hla)
            loss_val = criterion(output, label.long())
            test_loss += loss_val.item() 

            # Save predicted values
            _, preds = torch.max(output, 1) 
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(label.cpu().numpy())
            
    # Test average loss
    avg_test_loss = test_loss / len(testloader)
    test_losses.append(avg_test_loss)

    # Calculate F1 score and accuracy
    test_f1 = f1_score(all_test_labels, all_test_preds)
    test_acc = accuracy_score(all_test_labels, all_test_preds)

    print(f"<Test Loss>: {avg_test_loss:.4f}, Test F1 Score: {test_f1:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Check early stopping conditions
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        early_stop_count = 0  # Initialization
    else:
        early_stop_count += 1
        print(f"Early stopping counter {early_stop_count}/{patience}")
        if early_stop_count >= patience:
            print("Early Stop")

    # F1 score early stop
    if epoch_f1 > 0.90:
        print("F1 Score > 0.9")
        break
    
#### Neo-TCR Data External Valid ####
with open("/Neo_TCR_valid.pkl", "rb") as file:
    valid = pickle.load(file)
    
BATCH_SIZE = 64
validaloader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=True)

# Load TCR-epiDiff Encoder
model = UNet1D(in_channels=4, out_channels=4)
model = torch.load('/TCR-epiDiff_best_model.pth')
encoder = model.encoder
classifier = UNet1DClassifier(encoder, num_classes=2)
classifier = torch.load("/TCR-epiBP_best_model.pth")

# Validation
classifier.eval()

all_test_labels = []
all_test_preds = []

with torch.no_grad():
    for batch_data in validaloader:
        TCR = batch_data[0].to(device)
        epitope = batch_data[1].to(device)
        hla = batch_data[2].to(device)
        label = batch_data[3].to(device)

        t_tensor = torch.tensor([1.0] * TCR.size(0), dtype=torch.float)
        t_tensor = t_tensor.unsqueeze(1).expand(-1, 128).to(device)

        # Prediction
        output = classifier(TCR, t_tensor, epitope, hla)
        _, preds = torch.max(output, 1)
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(label.cpu().numpy())

# Accuracy & F1 Score
test_f1 = f1_score(all_test_labels, all_test_preds)
test_acc = accuracy_score(all_test_labels, all_test_preds)

print(f"Test F1 Score: {test_f1:.4f}, Test Accuracy: {test_acc:.4f}")