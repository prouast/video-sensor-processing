"""Convert video files to tfrecords."""

import argparse
import os
import math
import logging
import cv2
import numpy as np
import xml.etree.cElementTree as etree
from enum import Enum
import oreba_dis
import oreba_sha
from data_organiser import DataOrganiser

FILENAME_SUFFIX = 'tfrecord'

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%m/%d/%Y %I:%M:%S %p')

def read_video(filename, exp_fps):
  """Convert a video file to numpy array"""
  assert os.path.isfile(filename), "Couldn't find video file"
  # Read video and its properties with opencv
  cap = cv2.VideoCapture(filename)
  num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  # Determine how many frames to drop
  keep_every = fps/exp_fps
  if not keep_every.is_integer():
    raise RuntimeError(
      'Cannot get from {0} fps to {1} fps by dropping out frames.'
        .format(fps, exp_fps))
  # Determine number of frames
  num = math.ceil(num_frames/keep_every)
  timestamps = np.empty(num, np.dtype('float32'))
  frames = np.empty((num, width, height, 3), np.dtype('uint8'))
  # Convert frames to np arrays, get timestamps
  i = 0
  j = 0
  ret = True
  while (i < num_frames and ret):
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    ret, frame = cap.read()
    if i % keep_every == 0:
      timestamps[j] = timestamp
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frames[j] = frame
      j += 1
    i += 1
  cap.release()
  return timestamps, frames, num_frames

def calc_opt_flow(frames):
  """Calculate optical flow as Dual TV-L1"""
  frame_1 = frames[0]
  prvs = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY)
  flows = np.empty((frames.shape[0]-1, frames.shape[1], frames.shape[2], 2),
    np.dtype('float32'))
  i = 0
  for frame_2 in frames[1:]:
    next = cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY)
    dual_tv = cv2.optflow.createOptFlow_DualTVL1()
     	#dual_tv = cv2.DualTVL1OpticalFlow_create()
    flow = dual_tv.calc(prvs, next, None)
    flows[i] = flow
    i += 1
  return flows

def main(args=None):
  """Main"""
  # Identify dataset
  if args.dataset == "OREBA-DIS":
    dataset = oreba_dis.Dataset(args.src_dir, args.exp_dir, args.resolution,
      args.label_spec, args.label_spec_inherit)
  elif args.dataset == "OREBA-SHA":
    dataset = oreba_sha.Dataset(args.src_dir, args.exp_dir, args.resolution,
      args.label_spec, args.label_spec_inherit)
  else:
    raise ValueError("Dataset {} not implemented!".format(args.dataset))

  if not os.path.exists(args.exp_dir):
    os.makedirs(args.exp_dir)

  # Session ids
  ids = dataset.ids()
  logging.info("Found {0} participant directories.".format(str(len(ids))))

  # Create a separate TFRecords file for each video
  for id in ids:
    id_s = '_'.join([str(x) for x in id]) if isinstance(id, tuple) else id
    logging.info("Working on {}".format(id_s))

    # Exclude id if necessary
    if not dataset.check(id):
      continue

    # Output filename
    pp_s = "" +                                           \
      ("_opt_flow" if args.exp_optical_flow else "") +    \
      ("_" + str(args.exp_fps) + "_fps")
    out_filename = os.path.join(args.exp_dir,
      args.dataset + "_" + id_s + pp_s + "." + FILENAME_SUFFIX)

    # Check if file already generated
    if os.path.exists(out_filename):
      logging.info("Dataset file already exists. Skipping {0}.".format(id))
      continue

    # Read video
    video_filename = dataset.video_filename(id)
    timestamps, frames, num_orig = read_video(video_filename, args.exp_fps)

    # Compute optical flow and remove first frame
    flows = None
    if args.exp_optical_flow:
      flows = calc_opt_flow(frames)
      frames = frames[1:]
      timestamps = timestamps[1:]

    # Fetch labels
    labels = dataset.labels(id, timestamps)

    # Write
    dataset.write(out_filename, id, timestamps, frames, flows, labels)

  if args.organise_data:
    organiser = DataOrganiser(src_dir=args.exp_dir,
      organise_dir=args.organise_dir, dataset=args.dataset,
      organise_subfolders=args.organise_subfolders)
    organiser.organise()

  # Print info
  dataset.done()

  logging.info("Finished converting the dataset!")

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

class Resolution(Enum):
  res_140p = '140p'
  res_250p = '250p'
  def __str__(self):
    return self.value

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--src_dir', type=str, default='OREBA_Dataset_Public_1_0/oreba_dis/recordings', help='Recordings directory')
  parser.add_argument('--exp_dir', type=str, default='export', help='Directory for data export')
  parser.add_argument('--dataset', choices=('OREBA-DIS', 'OREBA-SHA'), default='OREBA-DIS', nargs='?', help='Which dataset is used')
  parser.add_argument('--label_spec', type=str, default='label_spec/OREBA_only_intake.xml', help='Filename of label specification')
  parser.add_argument('--resolution', type=Resolution, choices=list(Resolution), default='140p', help='Resolution for video')
  parser.add_argument('--exp_fps', type=float, default=8, help='Store video frames using this framerate')
  parser.add_argument('--exp_optical_flow', type=str2bool, default=False, help='Calculate optical flow')
  parser.add_argument('--label_spec_inherit', type=str2bool, default=True, help='Inherit label specification, e.g., if Serve not included, always keep sublabels as Idle')
  parser.add_argument('--organise_data', type=str2bool, default=False, nargs='?', help='If True, organise data in train, valid, test subfolders')
  parser.add_argument('--organise_dir', type=str, default='Organised', nargs='?', help='Directory to copy train, val and test sets using data organiser')
  parser.add_argument('--organise_subfolders', type=str2bool, default=False, nargs='?', help='Create sub folder per each file in validation and test set')
  args = parser.parse_args()
  main(args)
