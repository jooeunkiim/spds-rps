import os

import numpy as np
import torch
import torch.nn as nn
import cv2
import natsort

from skimage.color import rgb2gray
import imageio

# Data Loader
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, transform=None):#fdir, pdir, sdir, transform=None):
    self.fist_dir = os.path.join(data_dir,'rock/')
    self.palm_dir = os.path.join(data_dir,'paper/')
    self.swing_dir = os.path.join(data_dir,'scissors/')

    self.transform = transform

    lst_fist = os.listdir(self.fist_dir)
    lst_palm = os.listdir(self.palm_dir)
    lst_swing = os.listdir(self.swing_dir)

    lst_fist = [f for f in lst_fist if f.endswith(".png")]
    lst_palm = [f for f in lst_palm if f.endswith(".png")]
    lst_swing = [f for f in lst_swing if f.endswith(".png")]

    self.lst_dir = [self.fist_dir] * len(lst_fist) + [self.palm_dir] * len(lst_palm) + [self.swing_dir] * len(lst_swing)
    self.lst_prs = natsort.natsorted(lst_fist) + natsort.natsorted(lst_palm) + natsort.natsorted(lst_swing)
 
  def __len__(self):
    return len(self.lst_prs)

  def __getitem__(self, index): 
    self.img_dir = self.lst_dir[index]
    self.img_name = self.lst_prs[index]

    return [self.img_dir, self.img_name] 
    
  def custom_collate_fn(self, data):

    inputImages = []
    outputVectors = []

    for sample in data:
      prs_img = imageio.imread(os.path.join(sample[0] + sample[1]))
      gray_img = rgb2gray(prs_img)
      fname = sample[0] + sample[1]

      gray_img = cv2.resize(gray_img, (89, 100))
      if gray_img.ndim == 2:
        gray_img = gray_img[:, :, np.newaxis]

      inputImages.append(gray_img.reshape(89, 100, 1))

      dir_split = sample[0].split('/')
      if dir_split[-2] == 'rock':
        outputVectors.append(np.array(1))
      elif dir_split[-2] == 'paper':
        outputVectors.append(np.array(0))
      elif dir_split[-2] == 'scissors':
        outputVectors.append(np.array(2))

    data = {'input': inputImages, 'label': outputVectors, 'filename': fname}

    if self.transform:
      data = self.transform(data)

    return data


class ToTensor(object):
  def __call__(self, data):
    label, input = data['label'], data['input']

    input_tensor = torch.empty(len(input),89,100)
    label_tensor = torch.empty(len(input))
    for i in range(len(input)):
      input[i] = input[i].transpose((2, 0, 1)).astype(np.float32)
      input_tensor[i] = torch.from_numpy(input[i])
      label_tensor[i] = torch.from_numpy(label[i])
    input_tensor = torch.unsqueeze(input_tensor, 1)

    data = {'label': label_tensor.long(), 'input': input_tensor}

    return data

