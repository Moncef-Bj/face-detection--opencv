# Face Detection with OpenCV

A simple Python project to detect faces using OpenCV’s Haar Cascades. This project allows you to detect faces either in a static image or via a live webcam feed.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)


## Features
- Detects faces in images using `haarcascade_frontalface_default.xml`.
- Real-time face detection with your webcam.
- Draws bounding boxes around detected faces.

## Prerequisites
- **Python 3.7+** (recommended)
- **OpenCV** Python package

## Installation

1. **Clone or download** this repository:
   ```bash
   git clone https://github.com/Moncef-Bj/face-detection--opencv.git
   cd face-detection-opencv
   ```
2. **Install the required packages:**
   ```bash
   python -m venv venv
   source venv/bin/activate    # (Linux/Mac)
   venv\Scripts\activate       # (Windows)
   pip install -r requirements.txt
   ```
4. **Ensure the Haar Cascade file is present:**
 
   The file haarcascade_frontalface_default.xml should be located in the same directory as your script.

## Usage ##

There are two primary modes: image mode and webcam mode.

1. **Detect Faces in an Image**:
   
  ```bash
  python face_detection.py --image path/to/your/image.jpg
   ```

-Replace path/to/your/image.jpg with an actual image file path.

-The script will open a window showing the detected faces with bounding boxes.

-Close the window or press a key to exit.

2.**Detect Faces via Webcam** :

   ```bash
    python face_detection.py
   ```

 -This uses your computer’s default webcam.

 -Press q to exit the webcam window.

 




