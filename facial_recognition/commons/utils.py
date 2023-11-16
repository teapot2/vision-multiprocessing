import os
import cv2
import time
import logging
import pandas as pd
from deepface import DeepFace
from config import Config

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(module)s:%(funcName)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MODEL = "VGG-Face"
EUCLIDEAN_THRESHOLD_VALUE = 0.79


def extract_name_from_path(path):
    """Extracts the name from the file path."""
    return os.path.splitext(os.path.basename(path))[0]


def save_user_image(user_id, face):
    """Saves the user image with the given user ID."""
    save_path = Config.RECOGNITION_DB_PATH
    path = os.path.join(save_path, f"{user_id}.jpg")
    # cv2.imwrite(path, face * 255)
    logging.info(f"New user added with ID: {user_id}")


def run_facial_recognition_pipeline(frame):
    """Runs the facial recognition pipeline on the given frame."""
    face_find_info = []
    pipeline_start_time = time.time()

    try:
        # Extract faces from the frame
        face_objs = DeepFace.extract_faces(
            img_path=frame,
            target_size=(224, 224),
            detector_backend="ssd",
            enforce_detection=False,
        )

        for i, face_obj in enumerate(face_objs):
            face = face_obj["face"]
            face_coordinates = face_obj["facial_area"]

            x, y, w, h = (
                face_coordinates["x"],
                face_coordinates["y"],
                face_coordinates["w"],
                face_coordinates["h"],
            )

            # Perform face recognition
            dfs = DeepFace.find(
                img_path=face,
                db_path=Config.RECOGNITION_DB_PATH,
                enforce_detection=False,
                distance_metric="euclidean_l2",
                silent=True,
            )

            for df in dfs:
                # Filter results based on the Euclidean distance threshold
                df = df[df["VGG-Face_euclidean_l2"] < EUCLIDEAN_THRESHOLD_VALUE]
                identity_list = df["identity"].tolist()
                euclidean_l2 = df["VGG-Face_euclidean_l2"].tolist()

                # Retrieve the Euclidean distance
                if len(euclidean_l2) > 0:
                    euclidean_l2 = round(euclidean_l2[0], 4)

                # Check for a match
                if not identity_list or identity_list is None:
                    match = None
                    user_name = None
                else:
                    match = identity_list[0]
                    user_name = extract_name_from_path(match)

                # # Draw rectangle around the detected face and display info
                cv2.putText(
                    frame,
                    f"{str(user_name)} | {str(euclidean_l2)}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    1,
                )

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

                user_id = user_name

                # Save new user image or log the recognition
                if match is None:
                    save_user_image(user_id, face)
                else:
                    logging.info(f"Found user with ID: {user_id}")
                    save_user_image("name name existi " + user_id, face)

                cv2.imshow("frame", frame)
                cv2.waitKey(1)

                # Collect face recognition information
                image_info = (user_id, face, match)
                face_find_info.append(image_info)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    # Calculate and log the processing time
    pipeline_processing_time = time.time() - pipeline_start_time
    logging.debug(
        f"Pipeline processing time for: {round(pipeline_processing_time, 5)} seconds"
    )

    return
