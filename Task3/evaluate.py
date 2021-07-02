''' YOU CAN MODIFY EVERYTHING BELOW '''

import sys
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from dataset import *


def main(test_dir, result):
  new_test = test_dir

  ## comment out this block
  new_test = './new_test/'
  preprocess(test_dir, './new_test/', eval=True) 
  ##
  
  num_test = len(os.listdir(new_test))
  ''' YOU SHOULD SUBMIT param.data FOR THE INFERENCE WITH TEST DATA'''
  model_ckpt = './param.data'

  transform = transforms.Compose([ToTensor()])
  dataset_test = CustomDataset(new_test, transform=transform, test=True)
  loader_test = DataLoader(dataset_test, batch_size=num_test, \
          collate_fn=dataset_test.custom_collate_fn, num_workers=8)

  # Define Model
  model = nn.Sequential(nn.Conv2d(1, 32, 2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(32, 64, 2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(64, 128, 2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(128, 256, 2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(256, 512, 2, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=1),
                        nn.Dropout(0.3),
                        nn.Conv2d(512, 256, 2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Dropout(0.3),
                        nn.Conv2d(256, 128, 2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(128, 64, 2, padding=0),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=1),
                        torch.nn.Flatten(),
                        nn.Linear(64, 1000, bias = True),
                        nn.Dropout(0.5),
                        nn.Linear(1000, 3, bias = True),
                       )
  
  soft = nn.Softmax(dim=1)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("Current device:", device)

  if model_ckpt:
    state_dict = torch.load(model_ckpt, map_location=torch.device(device))
    state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
    model.load_state_dict(state_dict)

  correct_test = 0
  model.eval()
  with torch.no_grad():
    if torch.cuda.is_available():
      model = model.cuda()
    test_loss = []

    for batch, data in enumerate(loader_test, 1):

      input_test = data['input'].to(device)
      fname_test = data['filename']
      output_test = model(input_test)
  
      label_test_pred = soft(output_test).argmax(1)
      
  ''' WRITE THE TEST SET PREDICTION RESULT. FOLLOW THE INSTRUCTION BELOW. '''
  # Print one prediction result in a line.
  # One prediction result should have the format as {filename, label_prediction}.
  # 'label_prediction' should ONLY contain 0 or 1 or 2 (NOT 0.0 or 1.0E-15).
  # 0: Paper, 1: Rock, 2: Scissors
  # See an example file provided in the eTL. Format is VERY IMPORTANT.
 
  label_test_pred = label_test_pred.cpu().numpy().tolist()

  f = open(result, "w")
  for i in range(len(fname_test)):
    f.write(fname_test[i] + ",")
    f.write(str(label_test_pred[i]) + "\n")
  f.close()
  print("Done.")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

