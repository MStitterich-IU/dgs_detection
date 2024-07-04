# DGS Detection
## About
The DGS detection project contains the prototype of a solution designed to detect and recognize sign language gestures through action
recognition. Using a LSTM model, it can be trained with previously recorded video frames which in turn can be used for live detection and
recognition of sign language gestures in front of a camera.

While the project was initially designed to work with the German sign language (or DGS for short), its functionality can easily be
transferred to any other sign language that utilizes hand and arm gestures for letters, words or full expressions.

## Table of Contents

- [Features](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#features)  
- [Dependencies](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#dependencies)  
- [Installation](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#installation)  
  - [Prerequisites](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#prerequesites)  
  - [Clone the Repository](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#clone-the-repository)  
  - [(Optional) Create a Virtual Environment](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#optional-create-a-virtual-environment)  
  - [Install the Dependencies](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#install-the-dependencies)  
- [Usage](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#usage)  
  - [Record Data](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#record-data)  
  - [Train Model](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#train-model)  
  - [Evaluation](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#evaluation)  
  - [Live Gesture Recognition](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#live-gesture-recognition)  
- [Limitations](https://github.com/MStitterich-IU/dgs_detection?tab=readme-ov-file#limitations)  

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
The following steps will guide users through the installation process for the DGS detection prototype.

### Prerequisites
- Python 3.x: Please download and install the latest version from the [official website](https://www.python.org/)
- pip: Package installer for Python, usually included with the Python installation
- Tkinter: If not included with the Python installation, please use the local package manager for download and installation

  ```
  # Example on Ubuntu / Debian:
  
  sudo apt-get install python3-tk
  ```
### Clone the Repository
Using the local git installation, please clone the repository:

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
This will make sure that all required packages like NumPy, OpenCV, MediaPipe etc. are available in the environment.

## Usage
The steps below describe how to use the prototype for sign language recognition.

### Record Data
By executing `record_gestures.py` users will be presented with prompts, asking what gesture will be recorded and how many videos:

![Gesture Prompt](https://github.com/MStitterich-IU/dgs_detection/assets/119433042/be68253c-7108-491d-8299-9af5469b8152)
![Count Prompt](https://github.com/MStitterich-IU/dgs_detection/assets/119433042/f5fd62a7-d0fd-444b-bda6-572a9757779b)

This is followed by a prompt which simply lets users select where to store the previously recorded data.
Afterwards another window will open, showing the current image recorded by the camera. It will inform users what gesture is about to be 
recorded and which video number the recording is at.

![DataRecording](https://github.com/MStitterich-IU/dgs_detection/assets/119433042/db82d046-404f-4c0e-b700-8afc32aefe7a)

Now the application will loop through the amount of videos specified, letting users record the gesture multiple times in quick succession.
Between each video recording users will get two seconds of time to adjust their hand / arm position and pose to a neutral stance.  
If this preparation time is too long it can be skipped by pressing any keyon a keyboard.

### Train Model
Training the LSTM model can be done by executing `model_training.py`. It will prompt the user for various inputs, similarly to the "Record Data" step:

- Select the folder where the training data is stored
- Put in the number of required training iterations
- Select the folder for storing the trained model weights / files

With these inputs provided the prototype will start training for as many iterations as users have requested. Each model's weights will be stored in the
designated folder. Depending on the amount of training data, iterations and available computing power this may take a considerable amount of time. Each
iteration will save its results in a log file in the same folder as the one in which the models are stored.

### Evaluation
In order to figure out the prediction accuracy of a trained model the `model_evaluation.py' file can be executed. It will asks users to select a folder where
testing data has been stored. These recordings should **not** be the same as the training data used for training the model but separate recordings. In addition
users will be prompted to pick a model that they want to evaluate.  

The evaluation procedure will use the selected model to try and predict the gestures that were performed in the testing data. After it has finished going through
all the available testing data it will print out the accuracy score of the model's predictions.

### Live Gesture Recognition

## Limitations
