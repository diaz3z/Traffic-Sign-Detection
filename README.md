# Traffic Sign Detection

This project implements a traffic sign detection system using YOLO (You Only Look Once) object detection model. It can identify and classify various traffic signs in images.

## Features

- Detects multiple traffic signs in images
- Classifies signs into 47 different categories
- Draws bounding boxes around detected signs
- Displays class labels for each detected sign
- Counts and reports the number of each sign type in an image

## Model Training

This project uses YOLOv8, a state-of-the-art object detection model, trained on a custom dataset of traffic signs.

### Training Process
1. The dataset was prepared and hosted on Roboflow.
2. YOLOv8-small (yolov8s) was used as the base model.
3. Training was performed for 40 epochs with an image size of 800x800 pixels.

### Dataset
The traffic sign detection dataset was created and managed using Roboflow. It contains various traffic sign images with corresponding annotations.

## Model Performance

After training, the model was validated on a separate validation set. For detailed performance metrics, please refer to the training logs in the `runs/detect/train4/` directory.

## Using the Trained Model

The best performing weights from the training process are saved as `best.pt` in the `runs/detect/train4/weights/` directory. These weights can be used for inference on new images or videos.

## Future Improvements

- Fine-tune the model on more diverse traffic sign data
- Experiment with different YOLOv8 model sizes (nano, medium, large)
- Implement data augmentation techniques to improve model robustness
- Optimize the model for real-time detection on edge devices


## Requirements

- Python 3.x
- OpenCV (cv2)
- Ultralytics YOLO
- cvzone
- NumPy
- Pandas



## Installation

1. Clone this repository:
```bash
git clone https://github.com/diaz3z/Traffic-Sign-Detection.git

```
2. Install the required packages:

```bash
pip install opencv-python ultralytics cvzone numpy pandas

```

3. Download the trained YOLO model weights file (`best.pt`) and place it in the `runs/train/weights/` directory.

## Usage

1. Place your input images in a directory (default is `images/`).

2. Update the `path` variable in the script to point to your image directory if needed.

3. Run the script:
```bash
python run.py

```
4. The script will process each image in the specified directory, displaying:
- The image with bounding boxes and labels for detected signs
- A count of each type of sign detected in the image

5. Press any key to move to the next image.

## Customization

- To modify the list of detectable sign classes, update the `class_list` in the script.
- Adjust the image resize dimensions in the script to change the display size.

## License

[Specify your license here]

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push to your fork and submit a pull request

## Acknowledgements

- This project uses the YOLO model from Ultralytics
