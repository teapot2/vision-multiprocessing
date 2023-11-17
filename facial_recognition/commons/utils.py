import os
import cv2
import time
import logging
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


def draw_person_keypoints(frame, keypoint, color=(0, 0, 255), radius=2):
    """
    Draw keypoints on a given frame.

    Args:
        frame: The frame on which to draw the keypoints.
        keypoint: The list of keypoints to be drawn.
        color: The color of the keypoints (BGR format).
        radius: The radius of the keypoints.

    Returns:
        None
    """
    for point in keypoint:
        x, y = map(int, point)
        cv2.circle(frame, (x, y), radius, color, -1)


def draw_information(frame, face_coordinates, user_name, euclidean_l2, rectangle=False):
    """
    Draw user information on a frame, including the user's name, Euclidean distance, and an optional rectangle.

    Parameters:
    - frame (numpy.ndarray): The input frame.
    - face_coordinates (dict): Dictionary containing 'x', 'y', 'w', and 'h' coordinates of the detected face.
    - user_name (str): The name of the user.
    - euclidean_l2 (list): List containing the Euclidean distance.
    - rectangle (bool): Whether to draw a rectangle around the face (default: False).

    Returns:
    numpy.ndarray: The frame with drawn information.
    """
    x, y, w, h = (
        face_coordinates["x"],
        face_coordinates["y"],
        face_coordinates["w"],
        face_coordinates["h"],
    )

    if len(euclidean_l2) > 0:
        euclidean_l2 = round(euclidean_l2[0], 3)

    text = f"{user_name} | {euclidean_l2}"

    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
    )

    # Calculate the x-coordinate for centering the text
    text_x = x + (w - text_width) // 2

    # Draw black outline
    cv2.putText(
        frame,
        text,
        (text_x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # Draw yellow font
    cv2.putText(
        frame,
        text,
        (text_x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),  # Yellow color for the font
        1,
        cv2.LINE_AA,
    )

    if rectangle:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return frame


def draw_box(frame, box, color=(0, 255, 0), thickness=1):
    """
    Draw a rectangle (box) on the given frame.

    Args:
        frame: The frame on which to draw the rectangle.
        box: The coordinates of the rectangle (x1, y1, x2, y2).
        color: The color of the rectangle (BGR format).
        thickness: The thickness of the rectangle border.

    Returns:
        None
    """
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


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

                # Check for a match
                if not identity_list or identity_list is None:
                    match = None
                    user_name = None
                else:
                    match = identity_list[0]
                    user_name = extract_name_from_path(match)

                user_id = user_name

                # Save new user image or log the recognition
                if match is None:
                    save_user_image(user_id, face)
                else:
                    logging.info(f"Found user with ID: {user_id}")
                    save_user_image("name name existi " + user_id, face)

                # Collect face recognition information

                face_find_info.append(
                    {
                        "face_coordinates": face_coordinates,
                        "user_id": user_id,
                        "user_name": user_name,
                        "euclidean_l2": euclidean_l2,
                    }
                )

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    # Calculate and log the processing time
    pipeline_processing_time = time.time() - pipeline_start_time
    logging.debug(
        f"Pipeline processing time for: {round(pipeline_processing_time, 5)} seconds"
    )

    return face_find_info
