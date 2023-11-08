import logging
import config
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
        frame: The frame data to be stored.

    Returns:
        None

    Stores the segmented and compressed video data to the specified storage location or file system.
    """
    logging.debug(f"Storing video data for stream {index} - {name}...")

    try:
        storage_path = f"storage/camera_{camera_id}/"
        current_date = time.strftime("%Y-%m-%d")
        current_time = time.strftime("%H-%M-%S")
        filename = f"video_{current_date}_{current_time}.mp4"

        logging.debug(
            f"Storing video data for stream {index} at: {storage_path + filename}"
        )

        os.makedirs(storage_path, exist_ok=True)

        out = cv2.VideoWriter(
            storage_path + filename,
            fourcc,
            fps,
            (config.FRAME_WIDTH, config.FRAME_HEIGHT),
        )

        for frame in frames:
            out.write(frame)

        logging.info(
            f"Video for stream {camera_id} segment stored successfully: {storage_path + filename}"
        )

    except Exception as e:
        logging.error(
            f"An error occurred while storing the video segment for stream {camera_id}: {e}"
        )

    finally:
        out.release()
