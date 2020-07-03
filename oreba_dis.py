"""OREBA-DIS dataset"""

from collections import Counter
import datetime as dt
import os
import logging
import csv
import tensorflow as tf
import numpy as np
import xml.etree.cElementTree as etree

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
  datefmt='%H:%M:%S', level=logging.INFO)

DEFAULT_LABEL = "Idle"
VIDEO_SUFFIX = 'mp4'

TRAIN_IDS = ['1001','1002','1003','1006','1007','1008','1012','1013','1014',
  '1017','1018','1019','1022','1023','1024','1027','1028','1029','1032',
  '1033','1035','1039','1040','1041','1045','1046','1047','1051','1052',
  '1053','1056','1057','1059','1063','1064','1067','1072','1073','1077',
  '1079','1080','1083','1084','1085','1088','1089','1090','1093','1094',
  '1095','1098','1099','1100','1103','1104','1105','1109','1110','1111',
  '1115','1116']
VALID_IDS = ['1004','1010','1015','1020','1025','1030','1036','1043','1048',
  '1054','1060','1068','1075','1081','1086','1091','1096','1101','1107',
  '1112']
TEST_IDS = ['1005','1011','1016','1021','1026','1031','1037','1044','1050',
  '1055','1061','1071','1076','1082','1087','1092','1097','1102','1108',
  '1113']

def _int64_feature(value):
  """Return int64 feature"""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Return bytes feature"""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
  """Return float feature"""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class Dataset():

  def __init__(self, src_dir, exp_dir, resolution, label_spec, label_spec_inherit):
    self.src_dir = src_dir
    self.exp_dir = exp_dir
    self.label_spec = label_spec
    self.label_spec_inherit = label_spec_inherit
    self.resolution = resolution
    # Class names
    self.names_1, self.names_2, self.names_3, self.names_4 = \
      self.__class_names()
    # Class counters
    self.counts_1, self.counts_2, self.counts_3, self.counts_4 = \
      Counter(), Counter(), Counter(), Counter()

  def __class_names(self):
    """Get class names from label master file"""
    assert os.path.isfile(self.label_spec), "Couldn't find label_spec file at {}".format(self.label_spec)
    names_1 = []; names_2 = []; names_3 = []; names_4 = []
    tree = etree.parse(self.label_spec)
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

  def __timestamp_to_frame(self, time, timestamps):
    """Convert formatted timestamp to frame number"""
    t = dt.datetime.strptime(time, '%M:%S.%f')
    ms = t.minute * 60 * 1000 + t.second * 1000 + t.microsecond/1000
    return np.argmax(timestamps >= ms)

  def __add_to_class_counts(self, class_counts, labels):
    """Add increment to class counts"""
    unique, counts = np.unique(labels, return_counts=True)
    new_class_counts = Counter(dict(zip(unique, counts)))
    return class_counts + new_class_counts

  def ids(self):
    ids = [x for x in next(os.walk(self.src_dir))[1]]
    if not ids:
      raise RuntimeError('No participant directories found.')
    return ids

  def check(self, id):
    return id != "1074_1"

  def video_filename(self, id):
    return os.path.join(self.src_dir, id, id + "_video_" + str(self.resolution) + "." + VIDEO_SUFFIX)

  def labels_filename(self, id):
    return os.path.join(self.src_dir, id, id + "_" + "annotations.csv")

  def labels(self, id, timestamps):
    """Read labels from csv to numpy array"""
    labels_filename = self.labels_filename(id)
    assert os.path.isfile(labels_filename), "Couldn't find label file"
    # Set up np arrays
    num = len(timestamps)
    labels_1 = np.empty(num, dtype='U25'); labels_1.fill(DEFAULT_LABEL)
    labels_2 = np.empty(num, dtype='U25'); labels_2.fill(DEFAULT_LABEL)
    labels_3 = np.empty(num, dtype='U25'); labels_3.fill(DEFAULT_LABEL)
    labels_4 = np.empty(num, dtype='U25'); labels_4.fill(DEFAULT_LABEL)
    # Read the file
    with open(labels_filename) as dest_f:
      next(dest_f)
      for row in csv.reader(dest_f, delimiter=','):
        start_frame = self.__timestamp_to_frame(row[0], timestamps)
        end_frame = self.__timestamp_to_frame(row[1], timestamps)
        if row[4] in self.names_1:
          labels_1[start_frame:end_frame] = row[4]
        elif self.label_spec_inherit:
          continue
        if row[5] in self.names_2:
          labels_2[start_frame:end_frame] = row[5]
        if row[6] in self.names_3:
          labels_3[start_frame:end_frame] = row[6]
        if row[7] in self.names_4:
          labels_4[start_frame:end_frame] = row[7]
      # Update class names
      self.counts_1 = self.__add_to_class_counts(self.counts_1, labels_1)
      self.counts_2 = self.__add_to_class_counts(self.counts_2, labels_2)
      self.counts_3 = self.__add_to_class_counts(self.counts_3, labels_3)
      self.counts_4 = self.__add_to_class_counts(self.counts_4, labels_4)
    return labels_1, labels_2, labels_3, labels_4

  def write(self, path, id, timestamps, frames, flows, labels):
    logging.info("Writing {0} examples with {1} labels for video {2}..."
      .format(frames.shape, labels[0].shape, id))
    # Assertions
    assert (frames.shape[0]==labels[0].shape[0]), \
      "Frame and label length must match!"
    if flows is not None:
      assert (frames.shape[0]==flows.shape[0]), \
        "Frame and optical flow length must match!"
    # Grabbing the dimensions
    num = frames.shape[0]
    # Writing
    with tf.io.TFRecordWriter(path) as tfrecord_writer:
      for index in range(num):
        image_raw = frames[index].tostring()
        if flows is not None:
          flow_arr = flows[index].ravel()
          example = tf.train.Example(features=tf.train.Features(feature={
            'example/video_id': _bytes_feature(id.encode('utf-8')),
            'example/seq_no': _int64_feature(index),
            'example/label_1': _bytes_feature(labels[0][index].encode()),
            'example/label_2': _bytes_feature(labels[1][index].encode()),
            'example/label_3': _bytes_feature(labels[2][index].encode()),
            'example/label_4': _bytes_feature(labels[3][index].encode()),
            'example/image': _bytes_feature(image_raw),
            'example/flow': _floats_feature(flow_arr)
          }))
        else:
          example = tf.train.Example(features=tf.train.Features(feature={
            'example/video_id': _bytes_feature(id.encode('utf-8')),
            'example/seq_no': _int64_feature(index),
            'example/label_1': _bytes_feature(labels[0][index].encode()),
            'example/label_2': _bytes_feature(labels[1][index].encode()),
            'example/label_3': _bytes_feature(labels[2][index].encode()),
            'example/label_4': _bytes_feature(labels[3][index].encode()),
            'example/image': _bytes_feature(image_raw)
          }))
        tfrecord_writer.write(example.SerializeToString())

  def done(self):
    logging.info("Done")
    if not (self.counts_1[DEFAULT_LABEL] == self.counts_2[DEFAULT_LABEL] == self.counts_3[DEFAULT_LABEL] == self.counts_4[DEFAULT_LABEL]):
      logging.warning("Idle counts are not equal for all classes. " +
        "Please check label spec and/or label files.")
    logging.info("Final number of frames for category 1: {0}.".format(self.counts_1))
    logging.info("Final number of frames for category 2: {0}.".format(self.counts_2))
    logging.info("Final number of frames for category 3: {0}.".format(self.counts_3))
    logging.info("Final number of frames for category 4: {0}.".format(self.counts_4))

  def get_train_ids(self):
    return TRAIN_IDS

  def get_valid_ids(self):
    return VALID_IDS

  def get_test_ids(self):
    return TEST_IDS
