from ultralytics import YOLO
import urllib
import numpy as np
import cv2

def detector(data, model):
    try:
        file_path = data["name"]
        bucket_name = data["bucket"]

        image_path = f"https://storage.googleapis.com/{bucket_name}/{file_path}"
    except Exception as e:
        print(f"Debug: {e}")
        raise Exception("Error decode file data")
    
    try:
        # Read the input image using OpenCV
        req = urllib.request.urlopen(image_path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)  # 'Load it as it is'
    except Exception as e:
        print(f"DEBUG: exception when trying to read image. Error message: {e}")
        raise Exception("Error when trying to read image")
    
    try:
        model = YOLO(model)
        res = model.predict(image, project="output", name=file_path, boxes=True, save=True, imgsz=600, conf=0.2, hide_labels=True)
        # save class label names
        names = res[0].names    # same as model.names

        # store number of objects detected per class label
        class_detections_values = []
        for k, v in names.items():
            class_detections_values.append(res[0].boxes.cls.tolist().count(k))
        # create dictionary of objects detected per class
        classes_detected = dict(zip(names.values(), class_detections_values))
    except Exception as e:
        print(f"DEBUG: exception when trying to predict image. Error message: {e}")
        raise Exception("Error when trying to predict image")

