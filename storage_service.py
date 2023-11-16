from config import Config
import logging
import time
import cv2
import os

fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(module)s:%(funcName)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Function to store video data
def store_video_data(frames, camera_id, fps):
    """
    Store segmented and compressed video data to a designated storage location.

    Args:
        frames: The frame data to be stored.
        camera_id (int): The ID of the camera.
        fps (int): Frames per second.

    Returns:
        None

    Stores the segmented and compressed video data to the specified storage location or file system.
    """
    try:
        storage_path = f"storage/camera_{camera_id}/"
        current_date = time.strftime("%Y-%m-%d")
        current_time = time.strftime("%H-%M-%S")
        filename = f"video_{current_date}_{current_time}.mp4"

        logging.info(
            f"Storing video data for stream {camera_id} at: {storage_path + filename}"
        )

        os.makedirs(storage_path, exist_ok=True)

        out = cv2.VideoWriter(
            storage_path + filename,
            fourcc,
            fps,
            (Config.FRAME_WIDTH, Config.FRAME_HEIGHT),
        )

        for frame in frames:
            out.write(frame)

        logging.info(f"Video for stream {camera_id} segment stored successfully.")

    except Exception as e:
        logging.error(
            f"An error occurred while storing the video segment for stream {camera_id}: {e}"
        )

    finally:
        if "out" in locals():
            out.release()
