from config import Config
import requests
import logging
import json
import time
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(module)s:%(funcName)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

video_files = [
    {
        "camera_id": 0,
        "camera_location": "Interview Real Madrid",
        "camera_name": "Interview Real Madrid",
        "camera_url": os.path.join("data", "test_videos", "interview_1.mp4"),
        "camera_status": "online",
    },
    {
        "camera_id": 1,
        "camera_location": "test sf video",
        "camera_name": "test sf video",
        "camera_url": os.path.join("data", "test_videos", "Video_prueba.mp4"),
        "camera_status": "online",
    },
    {
        "camera_id": 2,
        "camera_location": "sf demo video",
        "camera_name": "sf demo video",
        "camera_url": os.path.join("data", "test_videos", "video3.avi"),
        "camera_status": "online",
    },
    {
        "camera_id": 3,
        "camera_location": "people walking",
        "camera_name": "people walking",
        "camera_url": os.path.join("data", "test_videos", "walking.mp4"),
        "camera_status": "online",
    },
]

# Select specific cameras for use
selected_streams = [
    video_files[0],
]


def get_cameras(update_cameras=False, offline=False):
    """
    Retrieve camera objects from the API.

    Returns:
        list: A list of camera objects
    """

    if offline:
        logging.info("Offline mode. Skipping camera retrieval from the API.")
        return selected_streams

    logging.info("Retrieving camera URLs from the API...")

    try:
        # Use 'raise_for_status()' to raise an HTTPError for bad responses
        res = requests.get(Config.CAMERA_API_URL)
        res.raise_for_status()
        response = res.json()
    except requests.RequestException as e:
        logging.critical(f"Failed to establish a connection to the API: {e}")
        sys.exit(1)
    except requests.HTTPError as e:
        logging.critical(f"HTTP error occurred: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.critical(f"Failed to decode JSON response: {e}")
        sys.exit(1)

    cameras = response.get("data", [])

    logging.info("Camera URLs retrieved successfully")

    if update_cameras:
        logging.info("Pinging cameras to check their status...")
        ping_cameras(cameras)

    return cameras


def update_camera_status(camera_id, data, i, size):
    """
    Update the camera status using the PUT method.

    Args:
        camera_id (int): The ID of the camera.
        data (dict): The data to be updated.
        i (int): Index of the camera.
        size (int): Total number of cameras.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    endpoint = f"{Config.CAMERA_API_URL}{camera_id}/"

    try:
        res = requests.put(endpoint, json=data)
        res.raise_for_status()

        if res.status_code == 200:
            logging.info(
                f"[{i + 1} / {size}] Camera status updated successfully for camera ID: {camera_id}"
            )
            return True
        else:
            logging.error(f"Failed to update camera status for camera ID: {camera_id}")
            return False
    except requests.RequestException as e:
        logging.error(
            f"Failed to update camera status for camera ID: {camera_id}. Error: {e}"
        )
        return False


def ping_cameras(cameras):
    """
    Ping cameras to check their status and update the status in the API.

    Args:
        cameras (list): List of camera objects.

    Returns:
        None
    """
    for i, camera in enumerate(cameras):
        url = camera["camera_url"]
        camera_name = camera["camera_name"]
        camera_id = camera["camera_id"]

        try:
            status = requests.head(url)
            status.raise_for_status()

            logging.info(f"[{i + 1} / {len(cameras)}] Camera {camera_name} is online")

            data = {"camera_status": "online"}
            update_camera_status(camera_id, data, i, len(cameras))

        except requests.RequestException as e:
            logging.warning(f"Camera {camera_name} is offline")
            logging.warning(f"Error message: {e}")

            data = {"camera_status": "offline"}
            update_camera_status(camera_id, data, i, len(cameras))

        time.sleep(0.5)

    logging.info("Camera status check complete.")


def send_user_api_request(data=None, user_id=None, method="GET"):
    try:
        api_url = Config.USERS_API_URL
        if user_id is not None:
            api_url += str(user_id)

        # Choose the HTTP method based on the provided 'method' parameter
        if method.upper() == "POST":
            res = requests.post(api_url, json=data)
        elif method.upper() == "PUT":
            res = requests.put(api_url, json=data)
        elif method.upper() == "GET":
            res = requests.get(api_url)
        else:
            raise ValueError("Unsupported HTTP method. Use 'POST', 'PUT', or 'GET'.")

        res.raise_for_status()
        response = res.json()

        print(response)
        # Check for a successful response (status code 2xx)
        if res.status_code // 100 == 2:
            print(f"API request to '{api_url}' successful!")
        else:
            print(
                f"Error in API request to '{api_url}'. Status code: {res.status_code}, Error message: {response}"
            )

    except requests.RequestException as e:
        print(f"Error in API request to '{api_url}': {e}")


send_user_api_request(method="GET")
