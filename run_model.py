from helpers import LSTM, check_file
import torch

model_name = "gothenburg_daily_3"
temperatures = [
    3.8,
    5.9,
    5.5,
    3.7,
    1.2,
    -1.1,
    3.0,
    2.0,
    0.9,
    0.8,
    2.5,
    4.9,
    6.9,
    6.2,
    2.1,
    5.1,
    4.3,
    3.3,
    5.1,
    1.7,
    3.1,
    1.9,
    4.7,
    5.8,
    4.8,
    5.9,
    6.2,
    4.3,
    6.1,
    5.2,
    # -1.0,
    # 0.6,
    # -0.6,
    # 3.3,
    # 4.5,
    # 3.9,
    # 3.4,
    # 3.7,
    # 5.2,
    # 4.9,
    # 4.2,
    # 2.3,
    # 3.3,
    # 3.7,
    # 4.4,
    # 4.6,
    # 3.3,
    # 3.1,
    # 3.5,
    # 2.9,
    # 2.3,
    # 2.0,
    # 0.1,
    # 3.8,
    # 3.9,
    # 3.3,
    # 1.0,
    # 2.7,
    # 3.3,
    # 3.3,
]

model = LSTM()

# Load the model  it exists
model_path = f"{model_name}/model.pt"
result = check_file(model_path)
if result:
    model.load_state_dict(torch.load(model_path))

# Fill the new input tensor with the given data
new_input = torch.tensor(temperatures).view(-1, 30)

# Run the model
# Use the model to make predictions on the new input data
predictions = model.predict(new_input)

# Get the first 14 predictions
predictions = predictions[0][:100]
print(predictions)
