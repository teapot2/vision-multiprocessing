import requests
import logging
import config
import json
import time
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(module)s:%(funcName)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_cameras(update_cameras=False):
    """
    Retrieve camera objects from the API.

    Returns:
        list: A list of camera objects
    """

    logging.info("Retrieving camera URLs from the API...")

    # Perform a GET request to obtain camera URLs

    try:
        res = requests.get("http://localhost:8000/api/cameras")
        response = json.loads(res.text)
    except Exception:
        logging.critical(f"Failed to establish a connection to the API: {e}")
        sys.exit(1)

    cameras = []

    for obj in response["data"]:
        cameras.append(obj)

    logging.info("Camera URLs retrieved successfully")

    if update_cameras == True:
        logging.info("Pinging cameras to check their status...")
        ping_cameras(cameras=response["data"])

    return cameras


def update_camera_status(camera_id, data, i, size):
    """
    Update the camera status using the PUT method.

    Args:
        camera_id (int): The ID of the camera.
        data (dict): The data to be updated.

    Returns:
        bool: True if the update was successful, False otherwise.
    """

    # Define the API endpoint for updating the camera status
    endpoint = f"http://localhost:8000/api/cameras/{camera_id}/"

    try:
        res = requests.put(endpoint, json=data)
        if res.status_code == 200:
            logging.info(
                f"[{i + 1} / {size}] Camera status updated successfully for camera ID: {camera_id}"
            )
            return True
        else:
            logging.error(f"Failed to update camera status for camera ID: {camera_id}")
            return False
    except Exception as e:
        logging.error(
            f"Failed to update camera status for camera ID: {camera_id}. Error: {e}"
        )
        return False


def ping_cameras(cameras):
    for i, camera in enumerate(cameras):
        url = camera["camera_url"]
        camera_name = camera["camera_name"]
        camera_id = camera["camera_id"]

        try:
            status = requests.head(url)
            logging.info(f"[{i + 1} / {len(cameras)}] Camera {camera_name} is online")

            # Update the camera status using the PUT method
            data = {"camera_status": "online"}
            update_camera_status(camera_id, data, i, len(cameras))

        except Exception as e:
            logging.warning(f"Camera {camera_name} is offline")
            logging.warning(f"Error message: {e}")
            data = {"camera_status": "offline"}
            update_camera_status(camera_id, data, i, len(cameras))

    logging.info("Camera status check complete.")

    time.sleep(0.5)

    return
