# Esta función recibe face_find_info (id, array del rostro original, path de la imagen con coincidencia(*Si Existe*))

import cv2 
from deepface import DeepFace
import os
import pandas as pd

def Face_Verif(id, face, match):
    save_path = 'vision-multiprocessing/DB/'
    models = ["Facenet512"]

    # Verificamos si hay una imagen para hacer la verificación
    #? Hay algo en math?
    #* No hay imagen para verificación
    if match is None:
        # Guardamos la imagen en la base de datos con el id que se genero en un inicio
        path = save_path + id + '.png'
        cv2.imwrite(path, face)
        print('Usuario Nuevo con el id: ', id)

    #* Si hay imagen para verificación
    else:
        df_verif = DeepFace.verify(img1_path=face, img2_path=match, enforce_detection=False, model_name = models[0])

        # Indica si es la misma persona
        verification = df_verif["verified"]

        #? Es la misma persona?
        #* Si es la misma persona
        if verification == True:
            # Guardamos la imagen en la base de datos con el id que se genero en un inicio
            id = os.path.splitext(os.path.basename(match))[0]
            print('Bienvenido: ', id)

        else:
            # Guardamos la imagen en la base de datos con el id que se genero en un inicio
            path = save_path + id + '.png'
            cv2.imwrite(path, face)
            print('Usuario Nuevo con el id: ', id)

    return id


            