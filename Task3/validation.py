from dataset import *
import argparse

def parse_args():

  parser = argparse.ArgumentParser(description="validation_preprocessing", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--val_dir", default="../Data/validation/", type=str, dest="val_dir")
  parser.add_argument("--new_dir", default="../Data/val_processed/", type=str, dest="new_dir")

  return parser.parse_args()

def main():
  args = parse_args()
  return preprocess(args.val_dir, args.new_dir, eval=True) 

if __name__ == '__main__': 
  print("Dataset preprocessing finished in directory: ", main())