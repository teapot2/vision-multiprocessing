import requests
import json
import time
import sys


def get_cameras(update_cameras=False):
    """
    Retrieve camera objects from the API.

    Returns:
        list: A list of camera objects
    """

    print("\033[94m\n[i] Attempting to retrieve camera URLs...\033[0m")

    # Perform a GET request to obtain camera URLs

    try:
        res = requests.get("http://localhost:8000/api/cameras")
        response = json.loads(res.text)
    except Exception:
        print(
            "\033[91m[ERROR] Connection to the API could not be established. Verify if the server is running.\033[0m"
        )
        sys.exit(1)

    cameras = []

    for obj in response["data"]:
        cameras.append(obj)

    print("\033[94m[i] Camera URLs retrieved successfully.\033[0m")

    if update_cameras == True:
        print("\033[94m[i] Pinging cameras...\n\033[0m")
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
            print(f"[{i + 1} / {size}] Camera status updated successfully...")
            return True
        else:
            print(
                f"\033[91m[ERROR] Camera status update was unsuccessful. Please try again.\033[0m"
            )
            return False
    except Exception as e:
        print(f"\033[91m[ERROR] {e}\033[0m")
        return False


def ping_cameras(cameras):
    for i, camera in enumerate(cameras):
        url = camera["camera_url"]
        camera_name = camera["camera_name"]
        camera_id = camera["camera_id"]

        try:
            status = requests.head(url)
            print(f"[{i + 1} / {len(cameras)}] Camera {camera_name} online...")

            # Update the camera status using the PUT method
            data = {"camera_status": "online"}
            update_camera_status(camera_id, data, i, len(cameras))

        except Exception as e:
            print(f"\033[91m[ERROR] Camera {camera_name} offline...\033[0m")
            print(f"\033[91m[ERROR] {e}\033[0m")
            data = {"camera_status": "offline"}
            update_camera_status(camera_id, data, i, len(cameras))

    print("\033[92m\n[i] Complete.\033[0m")

    time.sleep(0.5)

    return
