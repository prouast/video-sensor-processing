"""OREBA-SHA dataset"""

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

TRAIN_IDS = ['1002','1003','1018','1022','1024','1029','1033','1039','1041',
  '1045','1051','1083','1084','1110','1111','1115','2000','2001','2002',
  '2007','2008','2009','2015','2016','2017','2021','2022','2023','2027',
  '2028','2029','2033','2034','2035','2040','2041','2042','2043','2047',
  '2048','2049','2052','2053','2054','2057','2058','2061','2062','2065',
  '2066','2067','2070','2071','2072','2074','2076','2077','2078','2081',
  '2082','2083','2087','2088','2091','2092','2093']
VALID_IDS = ['1025','1036','1043','1068','1075','2003','2010','2018','2024',
  '2030','2036','2045','2050','2055','2063','2068','2073','2079','2084',
  '2094']
TEST_IDS = ['1037','1071','2004','2005','2011','2013','2019','2020','2025',
  '2026','2032','2037','2039','2046','2051','2056','2069','2075','2080',
  '2085','2090']

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
    return id != "2038_2" and id != "2064_2" and id != "2086_2" and id != "2089_2"

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
