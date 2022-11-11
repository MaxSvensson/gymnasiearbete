import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

#
# CONFIG DATASET
#

df = pd.read_csv("lerum.csv", sep=';')

# Drop missing records
df.dropna(subset=["Dygnsmedel"], inplace=True)
df.reset_index(drop=True, inplace=True)

dataset = df.iloc[:, 1].values
dataset = dataset.astype(float)

N_T = 361 * 14

N = 300  # number of samples
L = N_T//N  # length of each sample

formatted_dataset = np.array(
    [dataset[L*i:L*(i+1)] for i in range(N)]).astype(np.float32)

#
# LSTM
#


class LSTM(nn.Module):
    def __init__(self, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)

    def forward(self, y, future_preds=0):
        outputs, num_samples = [], y.size(0)
        h_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(num_samples, self.hidden_layers,
                           dtype=torch.float32)
        c_t2 = torch.zeros(num_samples, self.hidden_layers,
                           dtype=torch.float32)

        for time_step in y.split(1, dim=1):
            # N, 1
            # initial hidden and cell states
            h_t, c_t = self.lstm1(time_step, (h_t, c_t))
            # new hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)  # output from the last FC layer
            outputs.append(output)

        for i in range(future_preds):
            # this only generates future predictions if we pass in future_preds>0
            # mirrors the code above, using last output/prediction as input
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        # transform list to tensor
        outputs = torch.cat(outputs, dim=1)
        return outputs


#
# TRAIN
#

train_input = torch.from_numpy(
    formatted_dataset[3:, :-1])
train_target = torch.from_numpy(
    formatted_dataset[3:, 1:])

test_input = torch.from_numpy(formatted_dataset[:3, :-1])
test_target = torch.from_numpy(formatted_dataset[:3, 1:])

model = LSTM()
criterion = nn.MSELoss()
optimiser = torch.optim.LBFGS(model.parameters(), lr=0.0025)


def training_loop(n_epochs, model, optimiser, loss_fn,
                  train_input, train_target, test_input, test_target):
    for i in range(n_epochs):
        def closure():
            optimiser.zero_grad()
            out = model(train_input)
            loss = loss_fn(out, train_target)
            loss.backward()
            return loss
        optimiser.step(closure)
        with torch.no_grad():
            future = 14
            pred = model(test_input, future_preds=future)
            # use all pred samples, but only go to 999
            loss = loss_fn(pred[:, :-future], test_target)
            y = pred.detach().numpy()

        # draw figures
        plt.figure(figsize=(12, 6))
        plt.title(f"Step {i+1}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[1]  # 999

        def draw(yi, colour):
            plt.plot(np.arange(n), yi[:n], colour, linewidth=2.0)
            plt.plot(np.arange(n, n+future), yi[n:], colour+":", linewidth=2.0)
        draw(y[0], 'r')
        draw(y[1], 'b')
        draw(y[2], 'g')
        plt.savefig("model_6/predict%d.png" % i, dpi=200)
        plt.close()
        # print the loss
        out = model(train_input)
        loss_print = loss_fn(out, train_target)
        print("Step: {}, Loss: {}".format(i, loss_print))


training_loop(1000, model, optimiser, criterion, train_input,
              train_target, test_input, test_target)
