# DGS Detection
## Overview
The DGS detection project contains the prototype of a solution designed to detect and recognize sign language gestures through action
recognition. Using a LSTM model, it can be trained with previously recorded video frames which in turn can be used for live detection and
recognition of sign language gestures in front of a camera.

While the project was initially designed to work with the German sign language (or DGS for short), its functionality can easily be
transferred to any other sign language that utilizes hand and arm gestures for letters, words or full expressions.

## Features
- Recording of training data (single or multiple cameras)
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
- [`Tkinter`](https://docs.python.org/3/library/tkinter.html) &ge; 8.6.12

## Installation
The following steps will guide you through the installation process for the DGS detection prototype.

### Prerequesites
- Python 3.x: Please download and install the latest version from the [official website](https://www.python.org/)
- pip: Package installer for Python, usually included with the Python installation
- Tkinter: If not included with the Python installation, please use your package manager for download and installation

  ```
  # Example on Ubuntu / Debian:
  
  sudo apt-get install python3-tk
  ```
### Clone the Repository
Using your git installation, please clone the repository to your local machine:

```
git clone https://github.com/MStitterich-IU/dgs_detection.git
```

### (Optional) Create a Virtual Environment
In order to separate the required dependencies from potentially existing installations, please create a virtual environment inside the project directory.
```
# Example on Debian / Ubuntu

python3 -m venv .venv  # Creates the virtual environment as .venv
source .venv/bin/activate  # Activates the virtual environment
```
### Install the Dependencies
Using the requirements.txt file from this repository, install the dependencies using pip:

```
pip install -r requirements.txt
```
This will make sure that all required packages like NumPy, OpenCV, MediaPipe etc. are available in your environment.

## Usage
