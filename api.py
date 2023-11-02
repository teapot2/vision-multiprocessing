import requests
import json
import sys


def get_cameras():
    """
    Retrieve camera objects from the API.

    Returns:
        list: A list of camera objects
    """

    # Perform a GET request to obtain camera URLs

    print("\033[94m\nAttempting to retrieve camera URLs...\n\033[0m")

    try:
        res = requests.get("http://localhost:8000/api/cameras")
        response = json.loads(res.text)
    except Exception:
        print(
            "\033[91mERROR: Connection to the API could not be established. Verify if the server is running.\033[0m"
        )
        sys.exit(1)

    cameras = []

    for obj in response["data"]:
        cameras.append(obj)

    print("\033[92mCamera URLs retrieved successfully.\033[0m")

    print("\033[94m\nPinging cameras...\n\033[0m")

    ping_cameras(cameras=response["data"])

    return cameras


def update_camera_status(camera_id, data):
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
            print("\033[92mCamera status updated successfully...\033[0m")
            return True
        else:
            print(
                "\033[91mERROR: Camera status update was unsuccessful. Please try again.\033[0m"
            )
            return False
    except Exception as e:
        print(f"\033[91mERROR: {e}\033[0m")
        return False


def ping_cameras(cameras):
    for camera in cameras:
        url = camera["camera_url"]
        camera_name = camera["camera_name"]
        camera_id = camera["camera_id"]

        try:
            status = requests.head(url)
            print(f"\033[92mCamera {camera_name} online...\033[0m")

            # Update the camera status using the PUT method
            data = {"camera_status": "online"}
            update_camera_status(camera_id, data)

        except:
            print(f"\033[91mERROR: Camera {camera_name} offline...\033[0m")
            data = {"camera_status": "offline"}
            update_camera_status(camera_id, data)

    print("\033[92m\nComplete.\033[0m")

    return
