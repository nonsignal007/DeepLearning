
import torch
import torch.nn as nn
from data_loader import DataLoad
from model import MLP
from trainer import Train

import yaml

with open('config.yml') as f:
    config = yaml.safe_load(f)

model = MLP(config['INPUT_DIM'], config['OUTPUT_DIM'])
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

train = Train(model=model, optimizer=optimizer, criterion=criterion, train=True, EPOCHS=config['EPOCHS'], config = config)
