import cv2
from ultralytics import YOLO
import cvzone
import numpy as np
import pandas as pd
from collections import Counter  # Import Counter from collections module
import glob

model = YOLO(r"runs\train\weights\best.pt")


# my_file = open("coco.txt", "r")
# data = my_file.read()
class_list = ['0', 'ANIMALS', 'BVelocidad50', 'CONSTRUCTION', 'CYCLES CROSSING', 'DANGER', 'NO ENTRY', 'SCHOOL CROSSING', 'STOP', 'bend', 'bend left', 'bus_stop', 'crosswalk', 'give way', 'give_way', 'go left', 'go right', 'go right or straight', 'go straight', 'keep right', 'left_turn', 'no overtaking', 'no traffic both ways', 'no_overtaking_truck', 'no_stop', 'no_waiting', 'priority at next intersection', 'priority road', 'restriction ends -overtaking-', 'right_turn', 'road narrows', 'road_main', 'road_rough', 'road_work', 'rough_road', 'round_about', 'roundabout', 'slippery road', 'speed limit 30', 'speed limit 50', 'speed_limit_30', 'speed_limit_50', 'stop', 'traffic signal', 'truck', 'uneven road', 'warning']


def object(img):
    results = model.predict(img)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    object_classes = []

    for index, row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        obj_class = class_list[d]
        object_classes.append(obj_class)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cvzone.putTextRect(img, f'{obj_class}', (x2, y2), 1, 1)

    return object_classes

def count_objects_in_image(object_classes):
    counter = Counter(object_classes)
    print("Object Count in Image:")
    for obj, count in counter.items():
        print(f"{obj}: {count}")

path = r'C:\Users\mdzai\Desktop\Traffic Sign Detection\images\*.*'
for file in glob.glob(path):
    img = cv2.imread(file)
    img = cv2.resize(img, (1020, 500))
    object_classes = object(img)
    count_objects_in_image(object_classes)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()