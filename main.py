import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import ShakespeareDataset
from model import RNNModel, LSTMModel

# Parameters
SEQ_LENGTH = 100
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
N_LAYERS = 2
EPOCHS = 10
LEARNING_RATE = 0.001

# Load data
with open('shakespeare.txt', 'r') as f:
    text = f.read()

dataset = ShakespeareDataset(text, SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate models
vocab_size = len(dataset.chars)
rnn_model = RNNModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS)
lstm_model = LSTMModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

def train_model(model, optimizer):
    model.train()
    hidden = model.init_hidden(BATCH_SIZE)
    total_loss = 0

    for seq, target in dataloader:
        optimizer.zero_grad()
        output, hidden = model(seq, hidden)
        loss = criterion(output, target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

for epoch in range(EPOCHS):
    rnn_loss = train_model(rnn_model, rnn_optimizer)
    lstm_loss = train_model(lstm_model, lstm_optimizer)
    print(f'Epoch {epoch+1}, RNN Loss: {rnn_loss}, LSTM Loss: {lstm_loss}')

import matplotlib.pyplot as plt

rnn_losses = []
lstm_losses = []

for epoch in range(EPOCHS):
    rnn_loss = train_model(rnn_model, rnn_optimizer)
    lstm_loss = train_model(lstm_model, lstm_optimizer)
    rnn_losses.append(rnn_loss)
    lstm_losses.append(lstm_loss)
    print(f'Epoch {epoch+1}, RNN Loss: {rnn_loss}, LSTM Loss: {lstm_loss}')

plt.plot(rnn_losses, label='RNN Loss')
plt.plot(lstm_losses, label='LSTM Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
