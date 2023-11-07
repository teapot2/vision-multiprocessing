import config
import time
import cv2
import os

fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")


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
    try:
        storage_path = f"C:/Users/Sebastián/Documents/Tec/SF/multiprocessing_test/storage/{camera_id}/"
        current_date = time.strftime("%Y-%m-%d")
        current_time = time.strftime("%H-%M-%S")
        filename = f"video_{current_date}_{current_time}.mp4"

        os.makedirs(storage_path, exist_ok=True)

        out = cv2.VideoWriter(
            storage_path + filename,
            fourcc,
            fps,
            (config.FRAME_WIDTH, config.FRAME_HEIGHT),
        )

        for frame in frames:
            out.write(frame)

        print(f"Video for stream {camera_id} segment stored successfully: {storage_path + filename}")

    except Exception as e:
        print(f"An error occurred while storing the video segment: {e}")

    finally:
        out.release()
