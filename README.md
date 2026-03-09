# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
<img width="746" height="701" alt="DP5 5" src="https://github.com/user-attachments/assets/90b2bdcc-230f-45d7-88d8-6ff4bc65e570" />



## DESIGN STEPS
### STEP 1:

Load and normalize data, create sequences.

### STEP 2:
Convert data to tensors and set up DataLoader.

### STEP 3:
Define the RNN model architecture

### STEP 4:
Summarize, compile with loss and optimizer.

### STEP 5:
Train the model with loss tracking.

### STEP 6:
Predict on test data, plot actual vs. predicted prices.



## PROGRAM

### Name:PARVEEN SULTHANA J

### Register Number:212224040233

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

## Step 1: Load and Preprocess Data
# Load training and test datasets
df_train = pd.read_csv('/content/trainset.csv')
df_test = pd.read_csv('/content/testset.csv')
df_train.head()

# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

## Step 2: Define RNN Model
class RNNModel(nn.Module):
  def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel,self).__init__()
    self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc=nn.Linear(hidden_size,output_size)
  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

!pip install torchinfo

from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

criterion =nn.MSELoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    for epoch in range(epochs):
      total_loss=0
      for x_batch,y_batch in train_loader:
        x_batch,y_batch=x_batch.to(device),y_batch.to(device)
        optimizer.zero_grad()
        outputs=model(x_batch)
        loss=criterion(outputs,y_batch)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
      train_losses.append(total_loss/len(train_loader))
      print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
      # Plot training loss
    print('Name: PARVEEN SULTHANA J')
    print('Register Number: 212224040233')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()
train_model(model,train_loader,criterion,optimizer)

## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name: PARVEEN SULTHANA J')
print('Register Number: 212224040233')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')


```

### OUTPUT

## Training Loss Over Epochs Plot
<img width="317" height="466" alt="DP5 1" src="https://github.com/user-attachments/assets/b076c0dc-4a63-45a9-9fd5-7a3c8872f92e" />
<img width="746" height="544" alt="DP5 2" src="https://github.com/user-attachments/assets/d18df529-5a36-418a-b769-9ca50de5d51a" />


## True Stock Price, Predicted Stock Price vs time
<img width="1144" height="635" alt="DP5 3" src="https://github.com/user-attachments/assets/1f4dc947-0667-46f2-abbc-c417ba931d90" />



### Predictions
<img width="301" height="56" alt="DP5 4" src="https://github.com/user-attachments/assets/de12b1b8-d4d4-44bd-9a1c-a941b384cbaa" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
