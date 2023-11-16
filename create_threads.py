from multiprocessing import Process, shared_memory, Manager
import multiprocessing
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import argparse
import logging
import random
import signal
import torch
import time
import cv2
import sys
import os

from facial_recognition.commons.utils import run_facial_recognition_pipeline
from storage_service import store_video_data
from api import get_cameras, ping_cameras
from gui import process_status_gui
from config import Config
from sort import Sort


# Lambda function for generating shared memory stream names based on index
generate_shm_stream_name = lambda index: f"camera_{index}_stream"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(module)s:%(funcName)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

model = YOLO(os.path.join("models", "yolov8m-pose.pt"))
tracker = Sort()


# Function for sending a frame to shared memory
def send_frame_to_shared_memory(frame, shm):
    """
    Send a frame to the shared memory segment.

    Args:
        frame: The frame to be sent.
        shm: The shared memory segment.

    Returns:
        None

    Copies the frame data into the shared memory segment to be accessed by other processes.
    """
    shared_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
    shared_array[:] = frame[:]


# Handle Ctr+C interruptions
def signal_handler(sig, frame):
    """
    Function to handle the keyboard interrupt signal (Ctrl+C) and perform cleanup operations.

    Args:
        sig: The signal number.
        frame: The current stack frame.

    Returns:
        None

    This function is triggered when a keyboard interrupt (Ctrl+C) is detected. It terminates
    all active processes, closes threads, and exits the program gracefully.
    """

    logging.error("Interruption detected. Cleaning up processes and threads...")

    for p in processes:
        p.terminate()

    close_threads(urls)
    sys.exit(0)


# Function for processing camera streams
## TODO implement multiprocess logging
def process_camera(index, url, name, camera_id, shared_dict):
    """
    Process the camera streams.

    Args:
        index: The index of the camera stream.
        url: The URL of the camera stream.
        shared_dict: The shared dictionary for storing camera stream information.

    Returns:
        None

    Captures frames from the camera stream, processes them, and updates the shared dictionary with relevant information.
    """

    cap = cv2.VideoCapture(url)
    frames = []

    logging.debug(f"Starting process for stream at index {index}")

    try:
        logging.debug(f"Creating SharedMemory for stream at index {index}")
        shm = shared_memory.SharedMemory(
            create=True,
            size=Config.FRAME_SIZE_BYTES,
            name=generate_shm_stream_name(index),
        )
    except FileExistsError:
        logging.warning(
            f"Shared memory segment already exists for stream at index {index}"
        )
        shm = shared_memory.SharedMemory(name=generate_shm_stream_name(index))

    try:
        storage_start_time = time.time()
        processing_start_time = time.time()

        while True:
            ret, frame = cap.read()

            if ret:
                function_start_time = time.time()

                # Normalize frame to specified dimensions
                frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
                # frames.append(frame)

                # Store frames as video locally
                segmentation_interval = Config.VIDEO_SEGMENTATION_INTERVAL
                elapsed_storage_time = function_start_time - storage_start_time
                elapsed_processing_time = function_start_time - processing_start_time

                # frame = cv2.flip(frame, -1)

                # if elapsed_storage_time >= Config.VIDEO_SEGMENTATION_INTERVAL:
                #     estimated_fps = len(frames) / (elapsed_storage_time)
                #     store_video_data(frames, camera_id, int(estimated_fps))
                #     frames.clear()
                #     storage_start_time = function_start_time

                # Vision processing

                if elapsed_processing_time >= Config.PROCESSING_SEGMENTATION_INTERVAL:
                    estimated_fps = len(frames) / (elapsed_processing_time)

                    logging.debug(f"Performing recognition for stream {camera_id}")
                    results = model(frame, verbose=False)

                    for res in results:
                        filtered_indices = np.where(
                            res.boxes.conf.cpu().numpy() > 0.75
                        )[0]
                        boxes = (
                            res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
                        )

                        tracks = tracker.update(boxes)
                        tracks = tracks.astype(int)

                        for box, track_id in zip(boxes, tracks):
                            x1, y1, x2, y2 = box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                            # Add track ID as text on the frame
                            cv2.putText(
                                frame,
                                f"Track ID: {track_id[-1]}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                1,
                            )

                        res = run_facial_recognition_pipeline(frame)

                    processing_start_time = function_start_time

                send_frame_to_shared_memory(frame, shm)

                # Update the shared dictionary with relevant information
                shared_dict[index] = {
                    "execution_time": time.time() - function_start_time,
                    "camera_name": name,
                    "stream_name": generate_shm_stream_name(index),
                    "faces_detected": 0,
                }

    except Exception as e:
        logging.error(f"Error occurred while processing stream at index {index}: {e}")

        # Close the shared memory segment in case of an error
        terminate_shm(shm)

    finally:
        terminate_shm(shm)
        cv2.destroyAllWindows()


def terminate_shm(shm):
    """
    Close and unlink the shared memory segment.

    Args:
        shm: The shared memory segment.

    Returns:
        None
    """
    try:
        shm.close()
        shm.unlink()
        logging.debug("Shared memory segment closed and unlinked successfully.")
    except FileNotFoundError:
        logging.warning("FileNotFoundError: Shared memory segment not found.")
    except Exception as e:
        logging.error(f"An error occurred while closing the shared memory segment: {e}")


# Function to close threads and shared memory segments
def close_threads(urls):
    """
    Close threads and shared memory segments.

    Args:
        urls: The list of URLs for the camera streams.

    Returns:
        None

    Closes all active threads and shared memory segments associated with the camera streams.
    """
    for i, url in enumerate(urls):
        try:
            shm_name = generate_shm_stream_name(i)
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
            logging.debug(f"Shared memory segment {shm_name} closed successfully.")
        except FileNotFoundError:
            logging.debug(
                f"Shared memory segment {shm_name} could not be closed because it was not found."
            )
        except Exception as e:
            logging.error(
                f"Error occurred while closing shared memory segment {shm_name}: {e}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-U",
        "--update",
        action="store_true",
        default=False,
        help="Skip camera updates when this flag is specified",
    )

    parser.add_argument(
        "-M",
        "--monitor",
        action="store_true",
        default=False,
        help="Display monitoring information on a GUI during the process",
    )

    parser.add_argument(
        "--loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="Set the desired log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Run in offline mode; skip actions that require an online connection",
    )

    args = parser.parse_args()

    # Set the log level based on the provided --loglevel argument
    numeric_level = getattr(logging, args.loglevel, None)

    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.loglevel)

    logging.getLogger().setLevel(numeric_level)

    processes = []
    signal.signal(signal.SIGINT, signal_handler)

    # Configure the GPU device for YOLO if CUDA is available on the system
    if torch.cuda.is_available():
        device_id = 0  # You may adjust this based on your specific use case
        device_name = torch.cuda.get_device_name(device_id)
        logging.debug(
            f"CUDA is available on the system. Using {device_name} (ID {device_id})."
        )
        torch.cuda.set_device(device_id)
    else:
        logging.debug("CUDA is not available. Using CPU for computations.")

    # Use 'spawn' start method for multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_start_method("spawn")

    # Use a manager for shared dictionary
    with Manager() as manager:
        shared_dict = manager.dict()

        cameras = get_cameras(update_cameras=args.update, offline=args.offline)
        urls = [
            camera["camera_url"]
            for camera in cameras
            if camera["camera_status"] != "offline"
        ]
        names = [
            camera["camera_name"]
            for camera in cameras
            if camera["camera_status"] != "offline"
        ]
        ids = [
            camera["camera_id"]
            for camera in cameras
            if camera["camera_status"] != "offline"
        ]

        # Close any existing threads
        close_threads(urls)

        for i, (url, name, camera_id) in enumerate(zip(urls, names, ids)):
            logging.debug(f"Appending process for stream index: {i}")

            p = Process(
                target=process_camera, args=(i, url, name, camera_id, shared_dict)
            )
            p.start()
            processes.append(p)

        monitor_p = Process(target=process_status_gui, args=(shared_dict, args.monitor))
        monitor_p.start()
        processes.append(monitor_p)

        for p in processes:
            p.join()

        # Close all shared memory segments
        close_threads(urls)
