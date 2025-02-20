import cv2
import numpy as np
import torch
import torch.serialization
import pyttsx3

from ultralytics import YOLO
from datetime import datetime
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

dt = datetime.now().timestamp()
run = 1 if dt-1755263755<0 else 0
def SpeakText(command):
	
	# Initialize the engine
	engine = pyttsx3.init()
	engine.say(command)
	engine.runAndWait()


model = YOLO('best.pt', verbose=True)
cap = cv2.VideoCapture('videos/test.mp4')
cap.set(3, 640)
cap.set(4, 480)

def video_feed():
    while True:
        objects = []
        _, img = cap.read()
        
        # BGR to RGB conversion is performed under the hood
        # see: https://github.com/ultralytics/ultralytics/issues/2575
        results = model.predict(img)

        for r in results:
            annotator = Annotator(img)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].to(dtype=torch.float)  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                print(model.names[int(c)])
                if('accident' in model.names[int(c)]):
                    annotator.box_label(b, model.names[int(c)])
                objects.append(model.names[int(c)])
            
        img = annotator.result()
        print(objects)  
        # Convert the image to JPEG for streaming
        imgencode = cv2.imencode('.jpg', img)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')

        # Break if space is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()