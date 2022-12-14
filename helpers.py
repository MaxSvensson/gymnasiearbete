import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Network and Training


# LSTM Model
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

    def predict(self, x):
        # Use the forward() method to make predictions on the input data
        return self.forward(x)


# Plotting
def training_loop(n_epochs, model, optimiser, loss_fn,
                  train_input, train_target, test_input, test_target, model_name, file_name):
    generation = create_new_generation_folder(model_name)
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

        # dr figures
        if ((i+1) % 25 == 0) or (i == 0):
            plt.figure(figsize=(12, 6))
            plt.title(f"{file_name} | Step {i+1}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            n = train_input.shape[1]  # 999

            def draw(yi, colour):
                plt.plot(np.arange(n), yi[:n], colour, linewidth=2.0)
                plt.plot(np.arange(n, n+future),
                         yi[n:], colour+":", linewidth=2.0)

            def draw_data(data, max_lines=3):
                # Hard-coded colors array
                colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
                if not data.any():
                    return
                for i, d in enumerate(data):
                    if (i >= max_lines):
                        return
                    draw(d, colors[i % len(colors)])

            draw_data(y, max_lines=3)

            plt.savefig(
                f"{model_name}/gen_{generation}/predict{i+1}.png", dpi=200)
            plt.close()
            # print the loss
            out = model(train_input)
            loss_print = loss_fn(out, train_target)
            print("Step: {}, Loss: {}".format(i+1, loss_print))


# General

# Create a new generation folder
def create_new_generation_folder(folder_location):
    # Get a list of all subdirectories in the specified folder location
    subdirectories = [x[0] for x in os.walk(folder_location)]

    # Filter the list to include only directories that start with "gen_"
    gen_directories = [d for d in subdirectories if d.startswith(
        f"{folder_location}/gen_")]

    # Remove the folder_location part of the path
    gen_directories = [d.replace(f"{folder_location}/", "")
                       for d in gen_directories]

    # If there are no gen_ directories, create a new directory named "gen_1"
    if len(gen_directories) == 0:
        os.mkdir(os.path.join(folder_location, "gen_1"))
        return 1

    # Return the number of every gen_directory
    gen_directories = [int(d.replace("gen_", "")) for d in gen_directories]

    # Sort the gen_ directories by name
    gen_directories.sort()

    # Get the name of the latest gen_ directory (which will be the last element in the sorted list)
    gen_num = gen_directories[-1]

    # Create a new directory named "gen_X+1" where X is the generation number of the latest directory
    new_gen_directory = os.path.join(
        folder_location, "gen_{}".format(gen_num+1))
    os.mkdir(new_gen_directory)

    return gen_num+1


# Gets the latest generation folder
def get_current_generation(folder_location):
    # Get a list of all subdirectories in the specified folder location
    subdirectories = [x[0] for x in os.walk(folder_location)]

    # Filter the list to include only directories that start with "gen_"
    gen_directories = [d for d in subdirectories if d.startswith(
        f"{folder_location}/gen_")]

    # Remove the folder_location part of the path
    gen_directories = [d.replace(f"{folder_location}/", "")
                       for d in gen_directories]

    # If there are no gen_ directories, return 0
    if len(gen_directories) == 0:
        return 0

    # Return the number of every gen_directory
    gen_directories = [int(d.replace("gen_", "")) for d in gen_directories]

    # Sort the gen_ directories by name
    gen_directories.sort()

    # Get the name of the latest gen_ directory (which will be the last element in the sorted list)
    gen_num = gen_directories[-1]

    return gen_num


# Creates a folder if it doesn't exist
def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' was created.")
        return True
    return False


# Checks if a file exists
def check_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        # If the file does not exist, create it
        open(file_path, "w").close()
        # Return False because the file was created and is empty
        return False

    # Check if the file is empty
    if os.stat(file_path).st_size == 0:
        # If the file is empty, return False
        return False

    # If the file exists and is not empty, return True
    return True


# Move file to a folder
def move_file(file_path, destination):
    if os.path.exists(destination):
        os.remove(destination)
    os.rename(file_path, destination)
