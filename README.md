# Hand Gesture Recognition

## Overview
This project aims to develop a real-time hand gesture recognition system using deep learning and computer vision. The system captures hand gestures via a webcam, processes the images using Mediapipe to extract landmarks, and classifies the gestures using a deep neural network model built with TensorFlow/Keras.

The system can be used in various applications such as human-computer interaction, sign language recognition, and virtual reality interfaces.

## Features
- **Real-Time Hand Gesture Recognition:** Uses webcam input for real-time predictions.
- **Deep Learning Model:** Trained using hand landmarks detected by Mediapipe.
- **Gesture Prediction:** Classifies gestures into predefined categories like 'thumbs_up', 'thumbs_down', 'stop', etc.
- **Dataset:** Images of hand gestures used for training and testing the model.

## Dataset
The dataset used for this project contains images of various hand gestures. The dataset is hosted on Google Drive and can be accessed through the link below:

[Google Drive Dataset Link](https://drive.google.com/drive/folders/1pQy06sIVzYRRCc6DVWTQ6zMu7ME7FYqz?usp=drive_link)

The dataset includes the following hand gestures:
- Thumbs Up
- Thumbs Down
- Stop
- I Love You

Each gesture is captured in different lighting conditions and angles to make the model more robust.

## Requirements
Before running the project, you need to install the necessary libraries:

- `tensorflow`
- `opencv-python`
- `mediapipe`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `numpy`

You can install these dependencies using `pip`:

```bash
pip install tensorflow opencv-python mediapipe scikit-learn matplotlib seaborn numpy
```


## Setup & Usage
**1. Clone the Repository**
Clone the repository to your local machine:
```bash
git clone https://github.com/BunnySanga/Hand-Gesture-Recognition
```

**2. Prepare the Dataset**
Download dataset

**3.Upadte path**
Update path of your dataset in the real time Implementation

**4.Run the Model**
Execute the real time .ipynb file to start real-time hand gesture recognition at the bottom of the Signals_Project:


## Model Architecture
The model is built using TensorFlow/Keras and takes Mediapipe’s hand landmarks as input for classification. Key layers include:
Input Layer: Accepts 2D hand landmark coordinates.
Hidden Layers: Dense layers with ReLU activation.
Output Layer: A softmax layer for multi-class classification.

## Results
The model achieves high accuracy in detecting and classifying gestures, even under varied lighting and angles. Metrics such as accuracy and loss can be found in the training logs.


---

### Author
- **Sanga Balanarsimha** – [GitHub](https://github.com/BunnySanga)

---
