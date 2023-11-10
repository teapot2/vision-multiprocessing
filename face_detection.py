# Esta función recibe las coordenadas de un cuerpo y verifica si tiene rostro

# Backends para el reconocimiento facial
'''
backends = [
    'opencv', 
    'ssd', 
    'dlib', 
    'mtcnn', 
    'retinaface', 
    'mediapipe',
    'yolov8',
    'yunet',
    'fastmtcnn',
]   
'''

from deepface import DeepFace

import uuid
import cv2

# ! Este proceso solo se repite para la cantidad de personas 
def Face_Detection(xmin, ymin, xmax, ymax, frame):  
    # Recortar la imagen
    body = frame[ymin:ymax, xmin:xmax]

    # Buscamos una cara en el cuerpo
    result_faces = DeepFace.analyze(img_path = body, actions=['age'], enforce_detection=False, detector_backend = 'retinaface')
    
    # Encontramos una cara
    if result_faces is not None:                # Se comprueba si hay caras
        for faces in result_faces:              # Se repasa cada rostro detectado
            # Accede al campo 'region' de cada diccionario
            region = faces['region']  

            # Se toman las coordenadas del rostro
            # Ahora puedes acceder a los valores individuales dentro del diccionario 'region'        
            # Esquina superior izquierda
            x_min = int(region['x']) 
            y_min = int(region['y'])
            
            # Esquina inferior derecha
            x_max = int(region['x'] + region['w']) 
            y_max = int(region['y'] + region['h']) 

            face = body[y_min:y_max, x_min:x_max]

            '''
            # Mostrar la imagen en una ventana
            cv2.imshow('Titulo de la ventana', face)

            # Esperar a que se presione una tecla y luego cerrar la ventana
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''

            # Le asignamos un ID único
            id = str(uuid.uuid4()) 

            # Agregamos el path a una lista
            #imagen = (id, face) 
            #face_info.append(imagen)

    return id, face

    



    