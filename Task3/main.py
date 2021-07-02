import argparse
import time

from train import *

def parse_args():

  parser = argparse.ArgumentParser(description="SPDS_FinalPJT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--batchsize", default=32, type=int, dest="batchsize") 
  parser.add_argument("--epochs", default=10, type=int, dest="epochs")
  parser.add_argument("--train_dir", default="../Data/RPS/train/", type=str, dest="train_dir") 
  parser.add_argument("--val_dir", default="../Data/validation/", type=str, dest="val_dir")
  parser.add_argument("--new_dir", default="../Data/val_processed/", type=str, dest="new_dir")

  return parser.parse_args()


def main():
  args = parse_args()
  train(args)

if __name__ == '__main__':
  s = time.time()
  main()
  print("Total time: ", round(time.time()-s, 2), "seconds")
