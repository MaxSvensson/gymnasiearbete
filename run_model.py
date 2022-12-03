from helpers import LSTM, check_file
import torch

model_name = "gothenburg_daily"
temperatures = [
    1.2,
    1.4,
    2.5,
    1.8,
    0.9,
    0.9,
    2.0,
    5.2,
    3.9,
    2.7,
    3.0,
    1.6,
    2.1,
    1.3,
    0.6,
    3.6,
    4.4,
    3.3,
    4.2,
    6.5,
    5.9,
    7.1,
    6.0,
    5.6,
    7.8,
    9.2,
    8.3,
    5.4,
    5.4,
    3.8,
    4.5,
    3.4,
    4.4,
    5.0,
    2.6,
    5.5,
    4.5,
    1.3,
    3.6,
    3.9,
    3.2,
    5.3,
    5.9,
    9.8,
    11.8,
    12.1,
    10.4,
    6.7,
    7.0,
    5.4,
    4.1,
    4.7,
    4.8,
    5.2,
    7.4,
    8.6,
    7.8,
    6.6,
    8.0,
    7.2,
    4.9,
    6.3,
    4.6,
    5.6,
    5.9,
    9.9,
    15.1,
    10.8,
    12.3,
    13.6,
    13.7,
    10.2,
    11.0,
    10.9,
    11.2,
    10.0,
    9.7,
    9.7,
    11.6,
    10.6,
    11.4,
    12.5,
    10.6,
    10.7,
    12.4,
    12.2,
    15.2,
    13.3,
    15.4,
    16.3,
    18.0,
    18.3,
    18.1,
    18.2,
    15.7,
    14.4,
    15.3,
    16.2,
    14.9,
    14.4,
    15.0,
    13.4,
    14.2,
    20.0,
    20.6,
    20.8,
    17.9,
    15.2,
    16.1,
    16.0,
    15.1,
    15.1,
    15.6,
    15.7,
    16.3,
    16.5,
    16.8,
    15.8,
    18.7,
    16.0,
    8.8,
    10.5,
    16.9,
    13.9,
    18.6,
    22.1,
    23.4,
    22.9,
    23.5,
    20.8,
    18.2,
    18.0,
    17.6,
    17.7,
    18.1,
    18.3,
    20.0,
    21.9,
    21.0,
    19.4,
    18.6,
    16.1,
    16.6,
    17.8,
    15.9,
    14.4,
    14.5,
    14.3,
    15.3,
    17.7,
    17.4,
    15.6,
    17.5,
    15.2,
    15.8,
    17.3,
    17.4,
    15.5,
    16.1,
    13.8,
    13.4,
    15.1,
    15.4,
    14.9,
    14.0,
    13.9,
    15.0,
    14.8,
    17.7,
    13.6,
    14.4,
    14.3,
    16.4,
    15.3,
    15.4,
    14.7,
    14.0,
    14.5,
    12.8,
    11.7,
    12.4,
    14.0,
    17.2,
    17.3,
    17.1,
    16.8,
    16.6,
    14.4,
    12.2,
    14.2,
    11.2,
    10.0,
    11.4,
    10.1,
    9.6,
    9.6,
    12.0,
    12.6,
    11.4,
]

# run model
model = LSTM()

# Load the model  it exists
model_path = f"{model_name}/model.pt"
result = check_file(model_path)
if result:
    model.load_state_dict(torch.load(model_path))

# Fill the new input tensor with the given data
new_input = torch.tensor(temperatures).view(-1, 199)

# Run the model
# Use the model to make predictions on the new input data
predictions = model.predict(new_input)

# Get the first 14 predictions
predictions = predictions[0][:14]
print(predictions)
