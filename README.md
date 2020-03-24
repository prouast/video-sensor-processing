# video-sensor-processing

Convert video files and labels from the OREBA dataset to `tfrecord` format.

## Prepare data for TensorFlow

We use `.tfrecord` files to feed data into TensorFlow.
This includes [TV-L<sup>1</sup> optical flow](https://pequan.lip6.fr/~bereziat/cours/master/vision/papers/zach07.pdf), which is stored in the file alongside with raw frames and labels.
A dependency due to optical flow calculation is [opencv](https://opencv.org).

## Usage

For each video file `x.mp4`, there must be a label file `x_annotations.csv`.
To generate files from data in folder `data`:

```
$ python video_to_tfrecord.py --src_dir=data
```

The following flags can be set:

| Argument | Description |
| --- | --- |
| --src_dir | Directory to search for videos and labels |
| --exp_dir | Directory for data export |
| --video_suffix | Suffix of video files (defaults to *.mp4) |
| --label_spec | Filename of label specification (in src_dir) |
| --height | Height of video frames |
| --width | Width of video frames |
| --exp_fps | Store video frames using this framerate (Should be able to divide original framerate by this) |
| --exp_optical_flow | Calculate optical flow (defaults to False) |
