# conda activate base

import os
import sys
import pandas as pd
import numpy as np
from helpers import LSTM, check_file, create_folder_if_not_exists, create_new_generation_folder, get_current_generation, training_loop
import torch
import torch.nn as nn


# Model to train
model_name = "gothenburg_daily"

# Setup
folder_created = create_folder_if_not_exists(model_name)
if folder_created:
    create_folder_if_not_exists(f"{model_name}/data")
    sys.exit()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


# Train
data_folder = f"{model_name}/data"
if not os.path.exists(data_folder) or not os.listdir(data_folder):
    print("Data folder does not exist or is empty. Exiting.")
    sys.exit()

for csv_file in os.listdir(data_folder):
    # Check that the file is a CSV file
    if not csv_file.endswith(".csv"):
        continue

    print(f"Processing file: {csv_file}")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(data_folder, csv_file), sep=';')

    # Drop missing records
    df.dropna(subset=["Dygnsmedel"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Convert the temperature data to a numpy array
    dataset = df.iloc[:, 1].values
    dataset = dataset.astype(float)

    N_T = dataset.size

    # Split the data into training and testing sets
    L = 200
    N = N_T//L

    formatted_dataset = np.array(
        [dataset[L*i:L*(i+1)] for i in range(N)]).astype(np.float32)

    # Split the formatted dataset into input and target data for training and testing
    train_input = torch.from_numpy(formatted_dataset[3:, :-1])
    train_target = torch.from_numpy(formatted_dataset[3:, 1:])

    test_input = torch.from_numpy(formatted_dataset[:3, :-1])
    test_target = torch.from_numpy(formatted_dataset[:3, 1:])

    # Create the LSTM model
    model = LSTM()

    # Load the model if it exists
    model_path = f"{model_name}/model.pt"
    result = check_file(model_path)
    if result:
        model.load_state_dict(torch.load(model_path))

    # Loss function
    criterion = nn.MSELoss()
    # Optimizer
    optimiser = torch.optim.LBFGS(model.parameters(), lr=0.0025)

    # Train the model
    training_loop(100, model, optimiser, criterion, train_input,
                  train_target, test_input, test_target, model_name)

    print(f"Finished training Generation {get_current_generation(model_name)}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
