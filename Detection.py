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
        comedone_id = list(names)[list(names.values()).index("comedone")]
        pustule_id = list(names)[list(names.values()).index("pustule")]
        nodule_id = list(names)[list(names.values()).index("nodule")]
        papule_id = list(names)[list(names.values()).index("papule")]

        # count 'acne’ objects in the results
        comedone_count = results[0].boxes.cls.tolist().count(comedone_id)
        pustule_count = results[0].boxes.cls.tolist().count(pustule_id)
        nodule_count = results[0].boxes.cls.tolist().count(nodule_id)
        papule_count = results[0].boxes.cls.tolist().count(papule_id)
        print(f"comedone: {comedone_count}")
        print(f"pustule: {pustule_count}")
        print(f"nodule: {nodule_count}")
        print(f"papule: {papule_count}")

    except Exception as e:
        print(f"DEBUG: exception when trying to predict image. Error message: {e}")
        raise Exception("Error when trying to predict image")

    try:
        post_request(
            image_name=f"/{bucket_name}/{file_path}",
            image_result=f"output/{file_path}/image0.jpg",
            comedone_count=comedone_count,
            pustule_count=pustule_count,
            nodule_count=nodule_count,
            papule_count=papule_count,
        )
    except Exception as e:
        print(f"DEBUG: exception when trying to post request. Error message: {e}")
        raise Exception("Error when trying to post request")


def post_request(image_name, image_result, comedone_count, pustule_count, nodule_count, papule_count):
    with open(image_result, "rb") as image_file:
        files = {
            "file": (image_result, image_file),
        }
        data = {
            "image_name": image_name,
            "count": {
                "comedone": int(comedone_count),
                "pustule": int(pustule_count),
                "nodule": int(nodule_count),
                "papule": int(papule_count)
            },
        }
        response = requests.post(
            "https://skincheckai.id/api/internal/v1/acne-detection",
            files=files,
            data=data,
        )
        print(f"DEBUG: {image_name}")

    if response.status_code == 200:
        print("Permintaan berhasil!")
    else:
        print(
            "Gagal. Kode status: ",
            response.status_code,
            " , payload: ",
            response.json(),
        )

detector("image.jpg", "best.pt")