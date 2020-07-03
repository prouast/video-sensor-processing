import argparse
import os
import shutil
import logging
import oreba_dis
import oreba_sha

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
  datefmt='%H:%M:%S', level=logging.INFO)

class DataOrganiser:

  def __init__(self, src_dir, organise_dir, dataset, organise_subfolders):
    self.src_dir = src_dir
    self.dataset = dataset
    self.organise_dir = organise_dir
    self.organise_subfolders = organise_subfolders

  def organise(self):

    if self.dataset == "OREBA-DIS":
      train_ids = oreba_dis.TRAIN_IDS
      valid_ids = oreba_dis.VALID_IDS
      test_ids = oreba_dis.TEST_IDS
    elif self.dataset == "OREBA-SHA":
      train_ids = oreba_sha.TRAIN_IDS
      valid_ids = oreba_sha.VALID_IDS
      test_ids = oreba_sha.TEST_IDS

    train_dir = os.path.join(self.organise_dir, "train")
    valid_dir = os.path.join(self.organise_dir, "valid")
    test_dir = os.path.join(self.organise_dir, "test")

    all_files = os.listdir(self.src_dir)
    train_files = [f for f in all_files if any(id + "_" in f for id in train_ids)]
    valid_files = [f for f in all_files if any(id + "_" in f for id in valid_ids)]
    test_files = [f for f in all_files if any(id + "_" in f for id in test_ids)]

    assert len(list(set(train_files) & set(valid_files))) == 0, \
      "Overlap between train and valid"
    assert len(list(set(train_files) & set(test_files))) == 0, \
      "Overlap between train and test"
    assert len(list(set(valid_files) & set(test_files))) == 0, \
      "Overlap between valid and test"

    def copy_to_dir(file, origin, dest):
      if not os.path.exists(dest):
        os.makedirs(dest)
      origin_file = os.path.join(origin, file)
      if os.path.isfile(origin_file):
        shutil.copy(origin_file, dest)
      else:
        raise RuntimeError('File {} does not exist'.format(origin_file))

    for file in train_files:
      copy_to_dir(file, self.src_dir, train_dir)

    for file in valid_files:
      copy_to_dir(file, self.src_dir, valid_dir)
      if self.organise_subfolders:
        subdir = os.path.join(valid_dir + "_sub", file)
        copy_to_dir(file, self.src_dir, subdir)

    for file in test_files:
      copy_to_dir(file, self.src_dir, test_dir)
      if self.organise_subfolders:
        subdir = os.path.join(test_dir + "_sub", file)
        copy_to_dir(file, self.src_dir, subdir)

    logging.info("Done organising")

def main(args=None):
  organiser = DataOrganiser(src_dir=args.src_dir,
    organise_dir=args.organise_dir, dataset=args.dataset,
    organise_subfolders=args.organise_subfolders)
  organiser.organise()

def str2bool(v):
  """Boolean type for argparse"""
  if isinstance(v, bool):
     return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Organise exported files')
  parser.add_argument('--src_dir', type=str, default='OREBA-DIS', nargs='?', help='Directory to search for data')
  parser.add_argument('--dataset', choices=('OREBA-DIS', 'OREBA-SHA'), default='OREBA-DIS', nargs='?', help='Which dataset is used')
  parser.add_argument('--organise_dir', type=str, default='Organised', nargs='?', help='Directory to copy train, val and test sets using data organiser')
  parser.add_argument('--organise_subfolders', type=str2bool, default=False, nargs='?', help='Create sub folder per each file in validation and test set')
  args = parser.parse_args()
  main(args)
