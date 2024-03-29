import os
import sys
import pandas as pd
import numpy as np
from helpers import LSTM, check_file, create_folder_if_not_exists, get_current_generation, move_file, training_loop
import torch
import torch.nn as nn


# Model to train
model_name = "gothenburg_daily_3"

# Setup
folder_created = create_folder_if_not_exists(model_name)
if folder_created:
    create_folder_if_not_exists(f"{model_name}/data")
    create_folder_if_not_exists(f"{model_name}/processed_data")
    sys.exit()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


# Train
data_folder = f"{model_name}/data"
processed_data_folder = f"{model_name}/processed_data"
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
    df = df.dropna()
    df = df[df['Dygnsmedel'] != "Nan"]
    df = df.reset_index(drop=True)

    # Convert the temperature data to a numpy array
    dataset = df.iloc[:, 1].values
    dataset = dataset.astype(float)

    N_T = dataset.size

    # Split the data into training and testing sets
    L = 30
    N = N_T//L

    formatted_dataset = np.array(
        [dataset[L*i:L*(i+1)] for i in range(N)]).astype(np.float32)

    # Split the formatted dataset into input and target data for training and testing
    data_len = len(formatted_dataset)
    train_input = torch.from_numpy(formatted_dataset[data_len-1:, :-1])
    train_target = torch.from_numpy(formatted_dataset[data_len-1:, 1:])

    test_input = torch.from_numpy(formatted_dataset[:data_len-1, :-1])
    test_target = torch.from_numpy(formatted_dataset[:data_len-1, 1:])

    # Create the LSTM model
    model = LSTM()

    # Load the model  it exists
    model_path = f"{model_name}/model.pt"
    result = check_file(model_path)
    if result:
        model.load_state_dict(torch.load(model_path))

    # Loss function
    criterion = nn.MSELoss()
    # Optimizer
    optimiser = torch.optim.LBFGS(model.parameters(), lr=0.0005)

    # Train the model
    training_loop(100, model, optimiser, criterion, train_input,
                  train_target, test_input, test_target, model_name, csv_file)

    print(f"Finished training Generation {get_current_generation(model_name)}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)

    # Check if processed data folder exists
    if not os.path.exists(processed_data_folder):
        create_folder_if_not_exists(f"{model_name}/processed_data")
        print("Processed data folder created.")

    # Move cvs file to processed_data folder
    move_file(os.path.join(data_folder, csv_file),
              f"{model_name}/processed_data/{csv_file}")
