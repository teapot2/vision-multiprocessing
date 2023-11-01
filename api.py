import requests
import json
import sys


def get_cameras():
    """
    Retrieve camera objects from the API.

    Returns:
        tuple: (A list of camera URLs, A list of camera names) 
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

    urls = []
    names = []

    for obj in response["data"]:
        print(f"\033[94mRetrieved camera {obj['camera_name']}...\033[0m")
        names.append(obj["camera_name"])
        urls.append(obj["camera_url"])

    print("\033[92m\nCamera URLs retrieved successfully.\033[0m")
    return urls, names
