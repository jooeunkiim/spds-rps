import argparse

from train import *

def parse_args():

  parser = argparse.ArgumentParser(description="SPDS_FinalPJT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--batchsize", default=25, type=int, dest="batchsize") 
  parser.add_argument("--epochs", default=10, type=int, dest="epochs")
  parser.add_argument("--train_dir", default="../Data/RPS/train/", type=str, dest="train_dir") 
  parser.add_argument("--val_dir", default="../Data/RPS/validation/", type=str, dest="val_dir")

  return parser.parse_args()


def main():
  args = parse_args()
  train(args)

if __name__ == '__main__':
  main()
