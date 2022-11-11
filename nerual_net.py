# conda activate base

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from helpers import hoursSinceEpoch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

#
# CONFIG DATASET
#

df = pd.read_csv("lunden.csv", sep=';')

# Drop missing records
df.dropna(subset=["Dygnsmedel"], inplace=True)
df.reset_index(drop=True, inplace=True)

dataset = df.iloc[:, 1].values
dataset = dataset.astype(float)

train_window_end = 718
train_dataset = dataset[0:train_window_end]

scaler = MinMaxScaler(feature_range=(-1, 1))

train_data_normalized = scaler.fit_transform(train_dataset.reshape(-1, 1))
train_data_normalized = torch.FloatTensor(
    train_data_normalized).view(-1).to(device)

#
# TRAIN
#

train_window = 100


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


train_inout_seq = create_inout_sequences(train_data_normalized, train_window)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=500, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(device))

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

#
# TEST
#


from_save = False
model_path = "./model_5"
model = LSTM().to(device)

if (from_save == False):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                 torch.zeros(1, 1, model.hidden_layer_size).to(device))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    torch.save(model.state_dict(), model_path)
    print("Model saved to: ", model_path)
else:
    model.load_state_dict(torch.load(model_path))


fut_pred = 100

test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()


for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:]).to(device)
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))
        test_inputs.append(model(seq).item())

test_inputs[fut_pred:]
actual_predictions = scaler.inverse_transform(
    np.array(test_inputs[train_window:]).reshape(-1, 1))
real_values = dataset[train_window_end:train_window_end + fut_pred]


def printResult(list):
    for i in enumerate(list):
        print(f'{i[0]}: ', f'P  {i[1][0]}   ',  f'R   {i[1][1]}')


print(actual_predictions.flatten())
print(real_values)
plt.plot(actual_predictions, real_values)
plt.show()
# print("(Predicted Value:Real value)", printResult(
#     list(zip(actual_predictions.flatten().round(1).tolist(), real_values))))
