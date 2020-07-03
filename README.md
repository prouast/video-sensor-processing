# video-sensor-processing

Processing of video sensor data from the [OREBA dataset](http://www.newcastle.edu.au/oreba) for intake gesture detection.
Convert video files and labels to `tfrecord` format.

## Prepare data for TensorFlow

We use `.tfrecord` files to feed data into TensorFlow.
This includes [TV-L<sup>1</sup> optical flow](https://pequan.lip6.fr/~bereziat/cours/master/vision/papers/zach07.pdf), which is stored in the file alongside with raw frames and labels.
A dependency due to optical flow calculation is [opencv](https://opencv.org).

## Usage

Make sure that all requirements are fulfilled

```
$ brew install opencv
$ pip install -r requirements.txt
```

Then call `main.py`, pointing to the recordings directory for OREBA-DIS or OREBA-SHA.

```
$ python main.py --src_dir=OREBA_Dataset_Public_1_0/oreba_dis/recordings
```

The following flags can be set:

| Argument | Description | Default |
| --- | --- | --- |
| --src_dir | Recordings directory | OREBA_Dataset_Public_1_0/oreba_dis/recordings |
| --exp_dir | Directory for data export | Export |
| --dataset | Which dataset is used {OREBA-DIS or OREBA-SHA} | OREBA-DIS |
| --label_spec | Filename of label specification | label_spec/OREBA_only_intake.xml |
| --resolution | Resolution of the video {140p or 250p} | 140p |
| --exp_fps | Store video frames using this framerate (In fps; Should be able to divide original framerate by this) | 8 |
| --exp_optical_flow | Calculate optical flow | False |

## Label specfication

Control what labels are included by selecting or editing the appropriate `label_spec` file.
Templates are available in the `label_spec` directory.
