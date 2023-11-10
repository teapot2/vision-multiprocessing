# Recibe una dupla face_info (id, face array) y la base de datos de caras

from deepface import DeepFace
import pandas as pd


def Face_Match(id, face):
    face_db = 'vision-multiprocessing/DB'
    # Modelo de find de DeepFace
    models = ["Facenet512"]
    face_find_info = []

    # Iteramos para sacar los arrays de los rostros    

    # Buscamos rostros parecidos 
    df_find = DeepFace.find(img_path=face, db_path=face_db, enforce_detection=False, model_name = models[0])
    # Itero sobre los resultados
    for item in df_find:        
        # Se revisa si la instancia es un data frame
        if isinstance(item, pd.DataFrame):         
            # Convertimos los datos en una lista
            identity_list = item['identity'].tolist()

            # Verfica si la lista esta vacía
            #* No hay relación con ninguna imagen  
            if not identity_list or identity_list is None:
                # Indicamos que no hay match con ninguna imagen
                match = None
                # Guardamos el id, el array del rostro original y que no hizo match
                #imagen = (id, face, match) 
                #face_find_info.append(imagen)

            #* Si hay relación con una imagen  
            else:
                # Indicamos guardamos la imagen que tuvo el mejor match
                match = identity_list[0]
                # Guardamos el id, el array del rostro original y 
                # path del rostro con que hizo math
                #imagen = (id, face, match) 
                #face_find_info.append(imagen)
                
    return id, face, match

