import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from dataset import *
from copy import copy

import warnings

warnings.filterwarnings('ignore')


def train(args):

  num_train = len(os.listdir(os.path.join(args.train_dir, 'rock/'))) + \
          len(os.listdir(os.path.join(args.train_dir,'paper/'))) + \
          len(os.listdir(os.path.join(args.train_dir, 'scissors')))
  
  num_val = len(os.listdir(os.path.join(args.val_dir, 'rock/'))) + \
          len(os.listdir(os.path.join(args.val_dir, 'paper/'))) + \
          len(os.listdir(os.path.join(args.val_dir, 'scissors/'))) 

  transform = transforms.Compose([ToTensor()])
  dataset_train = CustomDataset(args.train_dir, transform=transform)
  loader_train = DataLoader(dataset_train, batch_size = args.batchsize, \
          shuffle=True, collate_fn=dataset_train.custom_collate_fn, num_workers=8)
  
  dataset_val = CustomDataset(args.val_dir, transform=transform)
  loader_val = DataLoader(dataset_val, batch_size=num_val, \
          shuffle=True, collate_fn=dataset_val.custom_collate_fn, num_workers=8)
  
  # Define Model
  model = nn.Sequential(nn.Conv2d(1, 32, 2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(32, 64, 2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(64, 128, 2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(128, 256, 2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(256, 256, 2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(256, 128, 2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(128, 64, 2, padding=0),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=1),
                        torch.nn.Flatten(),
                        nn.Linear(64, 1000, bias = True),
                        nn.Dropout(0.75),
                        nn.Linear(1000, 3, bias = True),
                       )
  
  soft = nn.Softmax(dim=1)
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("Current device:", device)
  
  model.to(device)
  
  # Define the loss
  criterion = nn.CrossEntropyLoss().to(device)
  
  # Define the optimizer
  optim = torch.optim.Adam(model.parameters(), lr = 0.0001)
  
  best_epoch = 0
  accuracy_save = np.array(0)
  for epoch in range(args.epochs):

    model.train()
    train_loss = []
    correct_train = 0
    correct_val = 0
    correct_batch = 0

    for batch, data in enumerate(loader_train, 1):
      label = data['label'].to(device)
      input = data['input'].to(device)

      output = model(input)
      label_pred = soft(output).argmax(1)
  
      optim.zero_grad()
  
      loss = criterion(output, label)
      loss.backward()
  
      optim.step()
  
      correct_train += (label == label_pred).float().sum()
  
      train_loss += [loss.item()]

    accuracy_train = correct_train / num_train
  
    correct_val = 0
    accuracy_tmp = np.array(0)
    with torch.no_grad():

      model.eval() 
      val_loss = []
      for batch, data in enumerate(loader_val, 1):

        label_val = data['label'].to(device)
        input_val = data['input'].to(device)
  
        output_val = model(input_val)
  
        label_val_pred = soft(output_val).argmax(1)
  
        correct_val += (label_val == label_val_pred).float().sum()
  
        loss = criterion(output_val, label_val)
        val_loss += [loss.item()]
  
      accuracy_val = correct_val / num_val
  
      # Save the best model wrt val accuracy
      accuracy_tmp = accuracy_val.cpu().numpy()
      if accuracy_save < accuracy_tmp:
        best_epoch = epoch
        accuracy_save = accuracy_tmp.copy()
        torch.save(model.state_dict(), 'param.data')
        print(".......model updated (epoch = ", epoch+1, ")")
        
    print("epoch: %04d / %04d | train loss: %.5f | train accuracy: %.4f | validation loss: %.5f | validation accuracy: %.4f" %
          (epoch+1, args.epochs, np.mean(train_loss), accuracy_train, np.mean(val_loss), accuracy_val))

  print("Model with the best validation accuracy is saved.")
  print("Best epoch: ", best_epoch)
  print("Best validation accuracy: ", accuracy_save)
  print("Done.")
    
