from ultralytics import YOLO
import urllib
import numpy as np
import cv2
import requests


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
        res = model.predict(
            image,
            project="output",
            name=file_path,
            boxes=True,
            save=True,
            imgsz=600,
            conf=0.2,
            hide_labels=True,
        )

        # run inference on the source image
        results = model(image)
        # get the model names list
        names = model.names
        # get the 'acne' class id
        acne_id = list(names)[list(names.values()).index("fore")]
        # count 'acne’ objects in the results
        count = results[0].boxes.cls.tolist().count(acne_id)
        print(count)

    except Exception as e:
        print(f"DEBUG: exception when trying to predict image. Error message: {e}")
        raise Exception("Error when trying to predict image")

    try:
        post_response = post_request(
            file_name=f"{bucket_name}/{file_path}",
            image_result=f"output/{file_path}/image0.jpg",
            acne_count=count,
        )
    except Exception as e:
        print(f"DEBUG: exception when trying to post request. Error message: {e}")
        raise Exception("Error when trying to post request")


def post_request(file_name, image_result, acne_count):
    with open(image_result, "rb") as image_file:
        files = {
            "file": (image_result, image_file, "image/jpeg"),
            "image_name": file_name,
            "count": acne_count,
        }
        response = requests.post(
            "https://skincheckai.id/api/internal/v1/acne-detection",
            file=files,
        )

    if response.status_code == 200:
        print("Permintaan berhasil!")
    else:
        print("Gagal. Kode status: ", response.status_code, " , payload ")
