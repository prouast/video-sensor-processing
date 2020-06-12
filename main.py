"""Convert video files to tfrecords."""

import argparse
import datetime as dt
import os
import math
import logging
import sys
from collections import Counter
import csv
import cv2
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import xml.etree.cElementTree as etree
from enum import Enum

FILENAME_SUFFIX = 'tfrecord'
DEFAULT_LABEL = 'Idle'

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%m/%d/%Y %I:%M:%S %p')

def _int64_feature(value):
  """Return int64 feature"""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Return bytes feature"""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
  """Return float feature"""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _get_class_names(label_spec):
  """Get class names from label master file"""
  assert os.path.isfile(label_spec), "Couldn't find label master file"
  names_1 = []; names_2 = []; names_3 = []; names_4 = []
  tree = etree.parse(label_spec)
  categories = tree.getroot()
  for tag in categories[0]:
    names_1.append(tag.attrib['name'])
  for tag in categories[1]:
    names_2.append(tag.attrib['name'])
  for tag in categories[2]:
    names_3.append(tag.attrib['name'])
  for tag in categories[3]:
    names_4.append(tag.attrib['name'])
  return names_1, names_2, names_3, names_4

def _add_to_tfrecord(tfrecord_writer, video_id, frames, flows,
  labels_1, labels_2, labels_3, labels_4, exp_optical_flow, offset=0):
  """Add a batch of frames to tfrecords file"""
  logging.info("Writing {0} examples with {1} labels for video {2}..."
    .format(frames.shape, labels_1.shape, video_id))
  # Assertions
  assert (frames.shape[0]==labels_1.shape[0]), \
    "Frame and label length must match!"
  if exp_optical_flow:
    assert (frames.shape[0]==flows.shape[0]), \
      "Frame and optical flow length must match!"
  # Grabbing the dimensions
  num = frames.shape[0]
  # Writing
  for index in range(num):
    image_raw = frames[index].tostring()
    if exp_optical_flow:
      flow_arr = flows[index].ravel()
      example = tf.train.Example(features=tf.train.Features(feature={
        'example/video_id': _bytes_feature(video_id.encode('utf-8')),
        'example/seq_no': _int64_feature(index),
        'example/label_1': _bytes_feature(labels_1[index].encode()),
        'example/label_2': _bytes_feature(labels_1[index].encode()),
        'example/label_3': _bytes_feature(labels_1[index].encode()),
        'example/label_4': _bytes_feature(labels_1[index].encode()),
        'example/image': _bytes_feature(image_raw),
        'example/flow': _floats_feature(flow_arr)
      }))
    else:
      example = tf.train.Example(features=tf.train.Features(feature={
        'example/video_id': _bytes_feature(video_id.encode('utf-8')),
        'example/seq_no': _int64_feature(index),
        'example/label_1': _bytes_feature(labels_1[index].encode()),
        'example/label_2': _bytes_feature(labels_1[index].encode()),
        'example/label_3': _bytes_feature(labels_1[index].encode()),
        'example/label_4': _bytes_feature(labels_1[index].encode()),
        'example/image': _bytes_feature(image_raw)
      }))
    tfrecord_writer.write(example.SerializeToString())
  return offset + num

def _video_to_numpy(filename, exp_fps):
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

def _calc_opt_flow(frames):
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

def _timestamp_to_frame(time, timestamps):
  """Convert formatted timestamp to frame number"""
  t = dt.datetime.strptime(time, '%M:%S.%f')
  ms = t.minute * 60 * 1000 + t.second * 1000 + t.microsecond/1000
  return np.argmax(timestamps >= ms)

def _label_to_numpy(filename, num, timestamps, names_1, names_2, names_3,
  names_4, inherit_label_spec):
  """Read labels from csv to numpy array"""
  assert os.path.isfile(filename), "Couldn't find label file"
  # Set up np arrays
  labels_1 = np.empty(num, dtype='U25'); labels_1.fill(DEFAULT_LABEL)
  labels_2 = np.empty(num, dtype='U25'); labels_2.fill(DEFAULT_LABEL)
  labels_3 = np.empty(num, dtype='U25'); labels_3.fill(DEFAULT_LABEL)
  labels_4 = np.empty(num, dtype='U25'); labels_4.fill(DEFAULT_LABEL)
  # Read the file
  with open(filename) as dest_f:
    next(dest_f)
    for row in csv.reader(dest_f, delimiter=','):
      start_frame = _timestamp_to_frame(row[0], timestamps)
      end_frame = _timestamp_to_frame(row[1], timestamps)
      if row[4] in names_1:
        labels_1[start_frame:end_frame] = row[4]
      elif inherit_label_spec:
        continue
      if row[5] in names_2:
        labels_2[start_frame:end_frame] = row[5]
      if row[6] in names_3:
        labels_3[start_frame:end_frame] = row[6]
      if row[7] in names_4:
        labels_4[start_frame:end_frame] = row[7]
  return labels_1, labels_2, labels_3, labels_4

def _label_to_int(label, class_names):
  """Convert from label name to int based on label master file"""
  return class_names.index(label)

def _add_to_class_counts(class_counts, labels):
  """Add increment to class counts"""
  unique, counts = np.unique(labels, return_counts=True)
  new_class_counts = Counter(dict(zip(unique, counts)))
  return class_counts + new_class_counts

def _write_label_file(names_1, names_2, names_3, names_4, counts_1, counts_2,
  counts_3, counts_4, directory, filename):
  """Writes a file with the list of class names"""
  labels_filename = os.path.join(directory, filename)
  categories = etree.Element("categories")
  category_1 = etree.SubElement(categories, "category", id="1", name="Default")
  for name in names_1:
    id = _label_to_int(name, names_1)
    count = counts_1[name]
    etree.SubElement(category_1, "tag", id=str(id), name=name, count=str(count))
  category_2 = etree.SubElement(categories, "category", id="2", name="Intake")
  for name in names_2:
    id = _label_to_int(name, names_2)
    count = counts_2[name]
    etree.SubElement(category_2, "tag", id=str(id), name=name, count=str(count))
  category_3 = etree.SubElement(categories, "category", id="3", name="Hand")
  for name in names_3:
    id = _label_to_int(name, names_3)
    count = counts_3[name]
    etree.SubElement(category_3, "tag", id=str(id), name=name, count=str(count))
  category_4 = etree.SubElement(categories, "category", id="4", name="Utensil")
  for name in names_4:
    id = _label_to_int(name, names_4)
    count = counts_4[name]
    etree.SubElement(category_4, "tag", id=str(id), name=name, count=str(count))
  tree = etree.ElementTree(categories)
  tree.write(labels_filename)

def main(args=None):
  """Main"""
  # Set number of opencv threads to sequential to run on HPC
  #cv2.setNumThreads(0)
  # Scan for participant dirs
  ids = [x for x in next(os.walk(args.src_dir))[1]]
  if not ids:
    raise RuntimeError('No participant directories found.')
  logging.info("Found {0} participant directories.".format(str(len(ids))))
  # Keep track of classes
  names_1, names_2, names_3, names_4 = _get_class_names(
    os.path.join(args.src_dir, args.label_spec))
  counts_1 = Counter(); counts_2 = Counter(); counts_3 = Counter(); counts_4 = Counter()
  # video_suffix
  video_desc = "video_" + str(args.resolution)
  # Create a separate TFRecords file for each video
  for id in ids:
    video_filename = os.path.join(args.src_dir, id,
      id + "_" + video_desc + "." + args.video_suffix)
    out_filename = os.path.join(args.exp_dir, "OREBA_" + id + "." + FILENAME_SUFFIX)
    if tf.io.gfile.exists(out_filename):
      logging.info("Dataset file already exists. Skipping {0}.".format(id))
      continue
    if not os.path.exists(args.exp_dir):
      os.makedirs(args.exp_dir)
    logging.info("Working on {0}.".format(id))
    # Fetch frames
    timestamps, frames, num_orig = _video_to_numpy(video_filename, args.exp_fps)
    # Compute optical flow and remove first frame
    flows = None
    if args.exp_optical_flow:
      flows = _calc_opt_flow(frames)
      frames = frames[1:]
      timestamps = timestamps[1:]
    # Fetch label filename
    labels_filename = os.path.join(args.src_dir, id,
      id + "_" + "annotations.csv")
    # Fetch labels
    labels_1, labels_2, labels_3, labels_4 = _label_to_numpy(
      labels_filename, len(frames), timestamps,
      names_1, names_2, names_3, names_4, args.inherit_label_spec)
    # Update class names
    counts_1 = _add_to_class_counts(counts_1, labels_1)
    counts_2 = _add_to_class_counts(counts_2, labels_2)
    counts_3 = _add_to_class_counts(counts_3, labels_3)
    counts_4 = _add_to_class_counts(counts_4, labels_4)
    with tf.io.TFRecordWriter(out_filename) as tfrecord_writer:
      # Write to .tfrecords file
      num = _add_to_tfrecord(tfrecord_writer, id, frames, flows,
                   labels_1, labels_2, labels_3, labels_4,
                   args.exp_optical_flow)
      logging.info("Finished writing {0} examples from {1} original " \
        "frames.".format(num, num_orig))
  # Print info
  if not (counts_1['Idle'] == counts_2['Idle'] == counts_3['Idle'] == counts_4['Idle']):
    logging.warning("Idle counts are not equal for all classes. " +
      "Please check label spec and/or label files.")
  logging.info("Final class counts for category 1: {0}.".format(counts_1))
  logging.info("Final class counts for category 2: {0}.".format(counts_2))
  logging.info("Final class counts for category 3: {0}.".format(counts_3))
  logging.info("Final class counts for category 4: {0}.".format(counts_4))
  # Write a labels file
  _write_label_file(names_1, names_2, names_3, names_4,
            counts_1, counts_2, counts_3, counts_4,
            args.exp_dir, "label_summary.xml")
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
  parser.add_argument('--src_dir', type=str, default='data', help='Directory to search for videos and labels')
  parser.add_argument('--exp_dir', type=str, default='export', help='Directory for data export')
  parser.add_argument('--video_suffix', type=str, default='mp4', help='Suffix of video files')
  parser.add_argument('--label_spec', type=str, default='labels.xml', help='Filename of label specification')
  parser.add_argument('--resolution', type=Resolution, choices=list(Resolution), default='140p', help='Resolution for video')
  parser.add_argument('--exp_fps', type=float, default=8, help='Store video frames using this framerate')
  parser.add_argument('--exp_optical_flow', type=str2bool, default=False, help='Calculate optical flow')
  parser.add_argument('--inherit_label_spec', type=str2bool, default=True, help='Inherit label specification, e.g., if Serve not included, always keep sublabels as Idle')
  args = parser.parse_args()
  main(args)
