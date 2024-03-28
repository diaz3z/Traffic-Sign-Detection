import cv2
import numpy as np

# Load YOLOv5 model
model = torch.load("runs/train/weights/best.pt")
model.eval()

# Load class names
classes = ['0', 'ANIMALS', 'BVelocidad50', 'CONSTRUCTION', 'CYCLES CROSSING', 'DANGER', 'NO ENTRY', 'SCHOOL CROSSING',
            'STOP', 'bend', 'bend left', 'bus_stop', 'crosswalk', 'give way', 'give_way', 'go left', 'go right',
              'go right or straight', 'go straight', 'keep right', 'left_turn', 'no overtaking', 'no traffic both ways',
                'no_overtaking_truck', 'no_stop', 'no_waiting', 'priority at next intersection', 'priority road',
                  'restriction ends -overtaking-', 'right_turn', 'road narrows', 'road_main', 'road_rough', 'road_work',
                    'rough_road', 'round_about', 'roundabout', 'slippery road', 'speed limit 30', 'speed limit 50', 'speed_limit_30',
                      'speed_limit_50', 'stop', 'traffic signal', 'truck', 'uneven road', 'warning']

# Load class names from file (replace "your_classes_file.txt" with the actual file path)
# with open("your_classes_file.txt", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# Set input image size and confidence threshold
input_size = (416, 416)
confidence_threshold = 0.5

# Load image
image = cv2.imread("images/-10-12-2021-184513_jpg.rf.d6b163cfe69eec549774a0ced7a9565f.jpg")
image_height, image_width, _ = image.shape

# Create blob from image
blob = cv2.dnn.blobFromImage(image, 1/255.0, input_size, swapRB=True, crop=False)

# Set input blob for the network
net.setInput(blob)

# Get output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

# Forward pass
outputs = net.forward(output_layer_names)

# Process detection outputs
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > confidence_threshold:
            # Get bounding box coordinates
            center_x = int(detection[0] * image_width)
            center_y = int(detection[1] * image_height)
            w = int(detection[2] * image_width)
            h = int(detection[3] * image_height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Draw bounding box and label
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show result
cv2.imshow("Traffic Sign Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
