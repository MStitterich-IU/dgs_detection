# DGS Detection
## Overview
The DGS detection project contains the prototype of a solution designed to detect and recognize sign language gestures through action
recognition. Using a LSTM model, it can be trained with previously recorded video frames which in turn can be used for live detection and
recognition of sign language gestures in front of a camera.

While the project was initially designed to work with the German sign language (or DGS for short), its functionality can easily be
transferred to any other sign language that utilizes hand and arm gestures for letters, words or full expressions.

## Features
- Recording of training data
- Building and training an LSTM model
- Evaluate the model's accuracy
- Live detection and recognition of trained gestures

## Dependencies
- [`Keras`](https://keras.io/) &ge; 3.3.3 
- [`MediaPipe`](https://developers.google.com/mediapipe) &ge; 0.10.14
- [`NumPy`](https://numpy.org/) &ge; 1.26.4
- [`OpenCV-Python`](https://opencv.org/) &ge; 4.10.0.82
- [`Scikit-learn`](https://scikit-learn.org/stable/) &ge; 1.5.0
- [`TensorBoard`](https://www.tensorflow.org/tensorboard) &ge; 2.16.2
- [`TensorFlow`](https://www.tensorflow.org/) &ge; 2.16.1

## Installation
