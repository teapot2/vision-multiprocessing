import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort
import time
import sys, os

from face_detection import Face_Detection
from face_match import Face_Match
from face_verify import Face_Verif

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    model = YOLO("yolov8n-pose.pt")

    tracker = Sort()

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        results = model(frame, stream=True)

        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.75)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)
            
            for xmin, ymin, xmax, ymax, track_id in tracks:
                # Proceso de detección facial
                id, face = Face_Detection (xmin, ymin, xmax, ymax, frame, backends = 'retinaface')
                # Proceso de buscar un match
                print('Estamos revisando quien eres!')
                id, face, match = Face_Match(id, face)
                # Proceso verifica si es la misma persona
                print('Se esta verificando su rostro')
                track_id = Face_Verif(id, face, match)

                cv2.putText(img=frame, text=f"Id: {track_id}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

        try:
            cv2.imshow('Ventana', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except cv2.error as e:
            print(f"Error: {e}")                
            
    cap.release()   