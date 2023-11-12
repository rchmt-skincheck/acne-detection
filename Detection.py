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
        model.predict(image, project="output", name=file_path, boxes=True, save=True, imgsz=600, conf=0.2, hide_labels=True)
        result = model(image)
        count = result.pandas().xyxy[0].value_counts('acne')
        print(count)
    except Exception as e:
        print(f"DEBUG: exception when trying to predict image. Error message: {e}")
        raise Exception("Error when trying to predict image")