from helpers import LSTM, check_file
import torch
from flask import Flask, request

model_name = "gothenburg_daily_3"
model = LSTM()

# Load the model  it exists
model_path = f"{model_name}/model.pt"
result = check_file(model_path)
if result:
    model.load_state_dict(torch.load(model_path))

# Serer

app = Flask(__name__)


@app.route("/", methods=['POST'])
def run_model():
    temps = request.form["data"].split(",")
    temps = [float(temp) for temp in temps]
    new_input = torch.tensor(temps).view(-1, 30)
    predictions = model.predict(new_input)
    return predictions[0][:30].tolist()


if __name__ == "__main__":
    app.run()
