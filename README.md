# SPDS Final Project Report

2021-22861 김주은 (spds092)



## Task 1

#### dataset.py

1. tensorflow RPS 데이터셋의 기본 이미지 크기가 $300\times300$ 으로 너무 크기 때문에, ```class CustomDataset``` 내부의 ```collate_fn``` 함수에서 각 이미지의 크기를 다음과 같이 $89\times100$ 으로 resize 하도록 했다.

```python
for sample in data:
  prs_img = imageio.imread(os.path.join(sample[0] + sample[1]))
  gray_img = rgb2gray(prs_img)
  fname = sample[0] + sample[1]

  gray_img = cv2.resize(gray_img, (89, 100))
  if gray_img.ndim == 2:
    gray_img = gray_img[:, :, np.newaxis]

    inputImages.append(gray_img.reshape(89, 100, 1))
...
```



#### train.py

1. 학습을 더 안정적으로 진행할 수 있도록 optimizer의 learning rate를 조정했다.

```python
optim = torch.optim.Adam(model.parameters(), lr = 0.0001)
```



#### result

training accuracy: 99.96%

validation accuracy: 93.82%



## Task3

#### NOTES

1. train 모델을 재현할 때 고려해야 하는 사항

   validation을 전처리하는데 걸리는 시간을 줄이기 위해 `process`라는 함수를 dataset.py 내부에 정의하였다. 이 함수를 실행하면 새로운 디렉토리 val_processed가 생기고 여기에 있는 데이터를 기준으로 validation을 하게 된다. --new_dir 뒤에 새로 저장할 디렉토리를 주면 되고, default로는 ../Data/val_processed/ 로 설정되어있다. 만약에 새로운 디렉토리를 지정하였다면 main.py를 실행할 때 --new_dir new_directory라는 argument를 지정해주면 된다.

   ```bash
   srun python validation.py --new_dir new_validation_directory(optional)
   ```

   

2. **evaluate 전 실행해야 하는 사항**

    train 과정에서 validation set을 따로 전처리 해주었기 때문에 evaluate set 또한 전처리를 해주어야 한다.

   방법 1) evaluate.py 내부에 전처리하는 과정을 포함해두었다. 이 방법을 사용하기를 추천한다. 이 방법을 사용한다면 기존 코드에서 아무 것도 수정하지 않아도 된다.

   ​		다만 evaluate set이 너무 커서 10분을 초과하는 경우는 방법 2를 참고하면 된다.

   방법 2) 1번에서 언급했던 것과 동일한 방법으로 원래 evaluate set, 처리한 evaluate set을 저장할 디렉토리를 지정해주고 validation.py를 실행한다. 이 때 evaluate.py 내부에 processing하는 코드를 주석처리해주면 중복 작업이 없다.

   ```python
   	## comment out this block
     new_test = './new_test/'
     preprocess(test_dir, './new_test/', eval=True) 
     ##
   ```

   다음으로 command line에서 다음 명령어를 실행한다.

   ```bash
   srun python validation.py --val_dir evaluate_set_directory --new_dir new_validation_directory
   ```

   이 작업을 완료하고 나면 데이터셋을 새로 저장한 디렉토리를 evaluate.py를 실행할 때 지정해주면 된다.



#### dataset.py

- train set (tensorflow RPS dataset) 전처리

  1. augmentation

     SPDS-RPS 데이터셋은 모양과 조명, 배경, 각도가 모두 다양한데에 비해 TF-RPS 데이터셋은 일정한 각도와 조명, 크기 그리고 배경에서 미세하게 다른 모습을 보였다. 이에 따라 validation set과 분포를 어느 정도 맞춰주기 위해서 다음과 같은 함수를 정의하고, 매 train epoch마다 random한 각도와 배경을 합성해주는 함수 ```fill_bk``` 를 정의하였다. 이를 위해 배경으로 줄 이미지를 Data/backgrounds/ 폴더에 넣어두었다.

     ```python
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
     ```

     

  2. normalize & resize

     다음 1번 단계에서 처리된 train 이미지들을 data loader에서 배치 단위로 학습하는 과정에서 거치는 함수인 ```custom_collate_fn``` 내부에서 contrast와 brightness를 추가해주어 이미지 내의 경계선에 대한 구분을 조금 더 명확하게 할 수 있도록 하고, gray scale로 바꾼 다음  RGB 값의 최대값은 255.0으로 나누어서 normalize 해주었다.

     ```python
     if self.train:
         i = np.random.randint(len(angles)+1)
         j = np.random.randint(num_bk)
         img = fill_bk(img, i, j)
     
         alpha = 1.5
         beta = 20
         img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
     
         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
         img = cv2.resize(img, (100, 100))
         if img.ndim == 2:
           img = img[:, :, np.newaxis]
           img = img.reshape(100, 100, 1)
     
           inputImages.append(img / 255.0)
     ```

     



- validation set (SPDS-RPS dataset) 전처리

  1. resize

     주어진 SPDS-RPS 데이터셋의 해상도가 너무 높아 매 epoch마다 validation을 진행하려면 시간이 너무 많이 소요되었다. Training 시간에 validation score를 확인하지 않고 train set 기준의 정확도가 가장 높은 모델이 overfitting 되었을 가능성이 있기 때문에, validation set에 최적으로 train 된 모델을 얻기 위해서 training을 시작할 때 다음과 같이 새로운 validation directory를 지정하고 거기에 preprocess한 이미지들을 저장한 다음, 새로 저장한 이미지들을 가지고 train을 진행했다.

     

     이를 위한 함수 ```preprocess``` 는 dataset.py 내부에 다음과 같이 정의하고, 각 이미지를 $100\times100$ 의 크기로 조정해서 저장하도록 했다.

     ```python
     def preprocess(data_dir, target_dir):
       if os.path.isdir(target_dir):
         shutil.rmtree(target_dir, ignore_errors=True)
       os.mkdir(target_dir)
       for hand in ['r/', 'p/', 's/']:
         os.mkdir(target_dir + hand)
         dr = os.path.join(data_dir+hand)
         for data in [f for f in os.listdir(dr) if f.endswith(".jpg")]:
           img = cv2.imread(os.path.join(dr+data))
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           img = crop(img)
           img = cv2.resize(img, (100, 100))
           cv2.imwrite(target_dir+hand+data, img)
       return target_dir
     ```

     

  2. crop and remove background

     1번에서 정의한 전처리 과정에서 validation 이미지들을 처리해주는 함수 `crop` 을 새로 만들었다. `crop` 에서는 일단 이미지의 contour을 찾고, skin color이라고 판단되는 HSV 값 사이에 있는 contour의 면적 중 가장 큰 것을 기준으로 가능하다면 정방형, 아니라면 그 모양 그대로 crop 하고 $100\times100$ 으로 사이즈를 줄여준다. 그런 다음, 다시 한 번 skin color의 범위 안에 있는 모든 pixel을 제외한 나머지 부분을 mask를 이용하여 흰색 배경으로 바꿔주었다. 이런 과정을 통해 SPDS-RPS validation set이 TF-RPS train set과 비슷한 분포를 갖도록 조정해주었다.

     ```python
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
     ```

     

#### train.py

1. Model 구조

   기존에 주어진 모델에서 kernel 크기와 약간의 레이어를 변경한 형태를 사용했다.

   

2. optimizer

   여러 가지 optimizer를 시도해본 결과, Adam은 trainset을 너무 빨리 학습해버려서 train set에 overfitting 되고 validation set의 정확도가 상대적으로 낮았다. 따라서 SGD optimizer를 사용했다.

   ```python
   optim = torch.optim.SGD(model.parameters(), lr = 0.01)
   ```

   

#### evaluate.py

1. data loader 변경

   label 별 디렉토리가 별도로 존재하지 않는 test set을 load할 수 있도록 기존에 존재하던 CustomDataset가 test라는 인자를 받도록 하고, 그 때는 하위 r/p/s 디렉토리가 아니라 직접 파일들을 받아올 수 있도록 하였다.

   ```python
   class CustomDataset(torch.utils.data.Dataset):
     def __init__(self, data_dir, transform=None, train = False, test=False):
       if test:
         lst = os.listdir(data_dir)
         lst = [f for f in lst if f.endswith(".jpg")]
         self.lst_dir = [data_dir] * len(lst)
         self.lst_prs = natsort.natsorted(lst)
   ```

   또한 label이 존재하지 않기 때문에 label 대신 filename을 받아와서 data 딕셔너리에 저장해서 결과 파일을 출력할 수 있도록 했다.

   ```
     def custom_collate_fn(self, data):
       inputImages = []
       outputVectors = []
       for sample in data:
         img = cv2.imread(os.path.join(sample[0] + sample[1]))
         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         ...
         if self.test:
           outputVectors.append(sample[1])    
       if self.test:
         data = {'input': inputImages, 'filename': outputVectors}
   ```

   

#### result

training accuracy: 97.06%

validation accuracy: 54.08%