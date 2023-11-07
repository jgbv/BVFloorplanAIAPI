import os
# import pytorch
from ultralytics import YOLO
import cv2 
from datetime import datetime as dt 
import json
from django.conf import settings
from django.core.files.base import ContentFile

class ParseFloorplan():
    
    def parse(image_path):
        weightsPath = os.path.join(settings.STATIC_ROOT,"trained7best.pt")
        model = YOLO(weightsPath)
        
        img = cv2.imread(image_path)
        
        print(f"===========image_path: {image_path}===========")
        results = model(image_path)
        jsonDict = {
            "imagePath": os.path.basename(image_path), 	
	        "imageHeight": img.shape[0],
	        "imageWidth": img.shape[1],
            "predictions": [],
            # "resultImg": None
        }
        
        r = results[0]
        points = [p for p in r.masks.xy[0]]
        
        predictions = [
            {
                "confidence": 0.95,	
                "class": "entirefloor",
                "name": "",	
                "points": []
            }
        ]
        
        for i in range(len(points)):
            # print(points[i][0])
            if i < len(points)-1:
                xy1 = (int(points[i][0]), int(points[i][1]))
                xy2 = (int(points[i+1][0]), int(points[i+1][1]))
                # print(f"{xy1},{xy2}")
                cv2.line(img, xy1, xy2, (0, 0, 255), 2)
                if xy1 not in predictions[0]["points"]:
                    predictions[0]["points"].append(xy1)
                if xy2 not in predictions[0]["points"]:
                    predictions[0]["points"].append(xy2)
        
        lastxy1 = (int(points[len(points)-1][0]), int(points[len(points)-1][1]))
        lastxy2 = (int(points[0][0]), int(points[i][1]))

        cv2.line(img, lastxy1, lastxy2, (0, 0, 255), 2)

        timestamp = dt.now().strftime("%Y%m%d%M%S")
        imname = os.path.basename(image_path).split(".")
        
        numepochs = 10
        numsamples = 5
        
        outname = f"bvim/resultsImg/{numepochs}e{numsamples}s-HL-{imname[0]}"
        
        jsonDict["predictions"] = predictions
        # jsonOutName = f"predictions/POLYFLOOR-{imname[0]}.json"
        # jsonObj = json.dumps(jsonDict, indent=4)
        # with open(jsonOutName, 'w') as jfile:
        #     jfile.write(jsonObj)
        
        cv2.imwrite(f"POLYFLOOR-{outname}.jpg", img)
        
        # #-----------------logic for sending the results image through the api response-------
        # print(f"===========lines img: {type(img)}: {img}=========")
        # # image_data = cv2.imencode(f'{outname}.jpg', img)[1].tobytes()
        # _, buffer = cv2.imencode('.jpg', img)
        # print(f"=====buffer:{buffer}")
        # image_file = ContentFile(buffer.tobytes())
        # print(f"=====image_file:{image_file}")
        # image_file.name = "POLYFLOOR-{outname}.jpg"
        # jsonDict["resultImg"] = image_file
        
        return jsonDict