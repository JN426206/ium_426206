import torch
import pandas as pd
import numpy as np

import torch.nn as nn
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim

df = pd.read_csv("mms_norm.csv", header=0, sep=',')

x_tensor = torch.tensor(df['Sales sum'].values).float()
y_tensor = torch.tensor(df['Sales count'].values).float()

dataset = TensorDataset(x_tensor, y_tensor)

lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
train_dataset, val_dataset = random_split(dataset, lengths)

train_loader = DataLoader(dataset=train_dataset)
val_loader = DataLoader(dataset=val_dataset)

class LayerLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Instead of our custom parameters, we use a Linear layer with single input and single output
        self.linear = nn.Linear(1, 1)
                
    def forward(self, x):
        # Now it only takes a call to the layer to make predictions
        return self.linear(x)
    
model = LayerLinearRegression()
# Checks model's parameters
#print(model.state_dict())   
lr = 1e-3
n_epochs = 20

loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step

# Creates the train_step function for our model, loss function and optimizer
train_step = make_train_step(model, loss_fn, optimizer)
training_losses = []
validation_losses = []
#print(model.state_dict())   
# For each epoch...
for epoch in range(n_epochs):

    losses = []
    # Uses loader to fetch one mini-batch for training
    for x_batch, y_batch in train_loader:
        # NOW, sends the mini-batch data to the device
        # so it matches location of the MODEL
        # x_batch = x_batch.to(device)
        # y_batch = y_batch.to(device)
        # One stpe of training
        loss = train_step(x_batch, y_batch)
        losses.append(loss)
    training_loss = np.mean(losses)
    training_losses.append(training_loss)

        
    # After finishing training steps for all mini-batches,
    # it is time for evaluation!
    # Ewaluacja jest już tutaj nie potrzebna bo odbywa sie w evaluation.py. Można jednak włączyć podgląd ewaluacji dla poszczególnych epok.    
    # # We tell PyTorch to NOT use autograd...
    # # Do you remember why?
    with torch.no_grad():
        val_losses = []
        # Uses loader to fetch one mini-batch for validation
        for x_val, y_val in val_loader:
            # Again, sends data to same device as model
            # x_val = x_val.to(device)
            # y_val = y_val.to(device)
            
            model.eval()
            # Makes predictions
            yhat = model(x_val)
            # Computes validation loss
            val_loss = loss_fn(y_val, yhat)
            val_losses.append(val_loss.item())
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    # print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")
print(f"{validation_loss:.4f}")
# torch.save({
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': lr,
#             }, 'model.pt')