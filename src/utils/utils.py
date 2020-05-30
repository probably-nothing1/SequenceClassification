import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

from src.models import LSTM

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, device, num_layers, hidden_size, embedding=False):
    model = LSTM(num_layers, hidden_size, embedding).to(device)
    return model.load_state_dict(torch.load(path))
