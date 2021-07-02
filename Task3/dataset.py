import os

import numpy as np
import torch
import torch.nn as nn
import cv2
import natsort

from skimage.color import rgb2gray
from scipy import ndimage
import imageio
import shutil

lower = np.array([0, 10, 60], dtype=np.uint8)
upper = np.array([20, 150, 255], dtype=np.uint8)
lower_white = np.array([230, 230, 230], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
lower_skin = np.array([0, 50, 50], dtype=np.uint8)
upper_skin = np.array([200, 250, 255], dtype=np.uint8)
angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, 45, -45]

bk_dir = '../Data/backgrounds/'
num_bk = 15
backgrounds = [] 
for i in range(1, 11):
  img = cv2.imread(bk_dir+str(i)+'.jpg')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  if img.shape[0] > img.shape[1]:
    bk = cv2.resize(img, (400, 300))[:,50:350,:]
  else:
    bk = cv2.resize(img, (300, 400))[50:350,:,:]
  backgrounds.append(bk)

def fill_bk(img, i, b):
  if i > 3:
    img = ndimage.rotate(img, angles[i-1], mode='constant', cval=255)[75:375, 75:375]
  elif i > 0:
      img = cv2.rotate(img, angles[i-1])
  if b > 9:
    return cv2.resize(img, (100, 100))
  mask = cv2.inRange(img, lower_white, upper_white) 
  mask = cv2.bitwise_not(mask)
  bk = backgrounds[b]
  fg_masked = cv2.bitwise_and(img, img, mask=mask)
  mask = cv2.bitwise_not(mask)
  bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
  final = cv2.bitwise_or(fg_masked, bk_masked)
  return cv2.resize(final, (100,100))

def crop(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsv, lower, upper)
    img_hand = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_area = 0
    maxcnt = None
    for cnt in contours :
        area = cv2.contourArea(cnt)
        if max_area < area:
            max_area = area
            maxcnt = cnt
    hull = cv2.convexHull(maxcnt)
    x_max, y_max = np.max(hull, axis=0)[0]
    x_min, y_min = np.min(hull, axis=0)[0]
    half = max(y_max-y_min, x_max-x_min)//2
    y_mid = (y_min+y_max)//2
    x_mid = (x_min+x_max)//2
    if y_mid-half >= 0 and x_mid-half >= 0:
        cropped_img = img_hsv[y_mid-half:y_mid+half, x_mid-half:x_mid+half]
    else:
        cropped_img = img_hsv[y_min:y_max, x_min:x_max]
    cropped_img_bgr = cv2.cvtColor(cropped_img, cv2.COLOR_HSV2RGB)
    img = cv2.resize(cropped_img_bgr, (100, 100), interpolation=cv2.INTER_AREA)
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsv, lower_skin, upper_skin) 
    bk = np.full(img.shape, fill_value=255).astype(np.uint8)
    fg_masked = cv2.bitwise_and(img, img, mask=mask)
    mask = cv2.bitwise_not(mask)
    bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
    img = cv2.bitwise_or(fg_masked, bk_masked)
    return img

def preprocess(data_dir, target_dir, eval=False):
  if os.path.isdir(target_dir):
    shutil.rmtree(target_dir, ignore_errors=True)
  os.mkdir(target_dir)
  if eval:
    for data in [f for f in os.listdir(data_dir) if f.endswith(".jpg")]:
      img = cv2.imread(os.path.join(data_dir+data))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = crop(img)
      img = cv2.resize(img, (100, 100))
      cv2.imwrite(target_dir+data, img)
  else:
    for hand in ['rock/', 'paper/', 'scissors/']:
      os.mkdir(target_dir + hand)
      dr = os.path.join(data_dir+hand)
      for data in [f for f in os.listdir(dr) if f.endswith(".jpg")]:
        img = cv2.imread(os.path.join(dr+data))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop(img)
        img = cv2.resize(img, (100, 100))
        cv2.imwrite(target_dir+hand+data, img)
  return target_dir

# Data Loader
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, transform=None, train = False, test=False):
    if test:
      lst = os.listdir(data_dir)
      lst = [f for f in lst if f.endswith(".jpg")]
      self.lst_dir = [data_dir] * len(lst)
      self.lst_prs = natsort.natsorted(lst)
    else:
      self.fist_dir = os.path.join(data_dir,'rock/')
      self.palm_dir = os.path.join(data_dir,'paper/')
      self.swing_dir = os.path.join(data_dir,'scissors/')

      lst_fist = os.listdir(self.fist_dir)
      lst_palm = os.listdir(self.palm_dir)
      lst_swing = os.listdir(self.swing_dir)

      lst_fist = [f for f in lst_fist if (f.endswith(".jpg") or f.endswith(".png"))]
      lst_palm = [f for f in lst_palm if (f.endswith(".jpg") or f.endswith(".png"))]
      lst_swing = [f for f in lst_swing if (f.endswith(".jpg") or f.endswith(".png"))]

      self.lst_dir = [self.fist_dir] * len(lst_fist) + [self.palm_dir] * len(lst_palm) + [self.swing_dir] * len(lst_swing)
      self.lst_prs = natsort.natsorted(lst_fist) + natsort.natsorted(lst_palm) + natsort.natsorted(lst_swing)
    
    self.transform = transform
    self.train = train
    self.test = test
    self.example = 0

  
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
      img = cv2.imread(os.path.join(sample[0] + sample[1]))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      
      if self.train:
        i = np.random.randint(len(angles)+1)
        j = np.random.randint(num_bk)
        img = fill_bk(img, i, j)
        if self.example < 10:
          self.example += 1
          cv2.imwrite('./sample/'+str(self.example)+'.jpg', img)

      alpha = 1.5 
      beta = 20 
      img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      img = cv2.resize(img, (100, 100))
      if img.ndim == 2:
        img = img[:, :, np.newaxis]
      img = img.reshape(100, 100, 1)

      inputImages.append(img / 255.0)

      if self.test:
        outputVectors.append(sample[1])
      else:
        dir_split = sample[0].split('/')
        if dir_split[-2] == 'rock':
          outputVectors.append(np.array(1))
        elif dir_split[-2] == 'paper':
          outputVectors.append(np.array(0))
        elif dir_split[-2] == 'scissors':
          outputVectors.append(np.array(2))
    
    if self.test:
      data = {'input': inputImages, 'filename': outputVectors}
    else:
      data = {'input': inputImages, 'label': outputVectors}

    if self.transform:
      data = self.transform(data)

    return data

class ToTensor(object):
  def __call__(self, data):
    if 'label' in data:
      label, input = data['label'], data['input']

      input_tensor = torch.empty(len(input),100,100)
      label_tensor = torch.empty(len(input))
      for i in range(len(input)):
        input[i] = input[i].transpose((2, 0, 1)).astype(np.float32)
        input_tensor[i] = torch.from_numpy(input[i])
        label_tensor[i] = torch.from_numpy(label[i])
      input_tensor = torch.unsqueeze(input_tensor, 1)

      return {'label': label_tensor.long(), 'input': input_tensor}
    else:
      filename, input = data['filename'], data['input']
      input_tensor = torch.empty(len(input),100,100)
      for i in range(len(input)):
        input[i] = input[i].transpose((2, 0, 1)).astype(np.float32)
        input_tensor[i] = torch.from_numpy(input[i]) 
      input_tensor = torch.unsqueeze(input_tensor, 1)
      
      return {'filename': filename, 'input': input_tensor}

