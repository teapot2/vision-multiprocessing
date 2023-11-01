import requests
import json
import sys


def get_urls():
    """
    Retrieve camera URLs from the API.

    Returns:
        list: A list of camera URLs.
    """

    # Perform a GET request to obtain camera URLs

    print("\033[94m\nAttempting to retrieve camera URLs...\033[0m")

    try:
        res = requests.get("http://localhost:8000/api/cameras")
        response = json.loads(res.text)
    except Exception:
        print(
            "\033[91mERROR: Connection to the API could not be established. Verify if the server is running.\033[0m"
        )
        sys.exit(1)

    urls = []

    for obj in response["data"]:
        urls.append(obj["camera_url"])

    print("\033[92mCamera URLs retrieved successfully.\033[0m")
    return urls
