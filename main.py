
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torchvision

import math
import tqdm

from model import MLPModel
from utils import get_image, image_to_tensor, show_data

BATCH_SIZE = 16

trainset = torchvision.datasets.ImageFolder('./dataset', transform=image_to_tensor)
validset = torchvision.datasets.ImageFolder('./dataset', transform=image_to_tensor)

train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset, batch_size = BATCH_SIZE, shuffle=False)

# show_data(train_loader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Environment: device')

model = MLPModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(model)

losses = []
prev_loss = 1.0
EPOCHS = 50

for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f'train model ({epoch}/{EPOCHS} epoch)'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model.forward(inputs) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    batch_size = len(train_loader)
    avg_loss = running_loss / batch_size
    losses.append(avg_loss)
    print('[%d] loss: %.6f, diff: %.6f' % (epoch + 1, avg_loss, prev_loss))
    
    if abs(prev_loss - avg_loss) < 1e-5: break
    prev_loss = (prev_loss + avg_loss) / 2

# save model
torch.save(model, './model.pth')