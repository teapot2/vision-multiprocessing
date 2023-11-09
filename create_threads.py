from multiprocessing import Process, shared_memory, Manager
from api import get_cameras, ping_cameras
from storage_service import store_video_data
import numpy as np
import argparse
import logging
import random
import config
import signal
import time
import cv2
import sys
import os


# Lambda function for generating shared memory stream names based on index
generate_shm_stream_name = lambda index: f"camera_{index}_stream"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(module)s:%(funcName)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
            size=config.FRAME_SIZE_BYTES,
            name=generate_shm_stream_name(index),
        )
    except FileExistsError:
        logging.warning(
            f"Shared memory segment already exists for stream at index {index}"
        )
        shm = shared_memory.SharedMemory(name=generate_shm_stream_name(index))

    try:
        storage_start_time = time.time()

        while True:
            ret, frame = cap.read()

            if ret:
                function_start_time = time.time()

                # Normalize frame to specified dimensions
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                frames.append(frame)

                # Store frames as video locally
                segmentation_interval = config.VIDEO_SEGMENTATION_INTERVAL
                elapsed_time = function_start_time - storage_start_time

                if elapsed_time >= config.VIDEO_SEGMENTATION_INTERVAL:
                    estimated_fps = len(frames) / (elapsed_time)
                    store_video_data(frames, camera_id, int(estimated_fps))
                    frames.clear()
                    storage_start_time = function_start_time

                # Vision processing logic goes here

                send_frame_to_shared_memory(frame, shm)

                # Update the shared dictionary with relevant information
                shared_dict[index] = {
                    "execution_time": f"{time.time() - function_start_time:.5f} s",
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


# Function for monitoring the status of the camera processes
def monitor_process_status(shared_dict, monitor):
    """
    Monitor the status of the camera processes.

    Args:
        shared_dict: The shared dictionary containing camera stream information.
        monitor: A boolean value indicating whether to display monitoring information.

    Returns:
        None

    Continuously monitors and displays the status of the camera processes in the terminal.
    """

    while True:
        if monitor:
            clear_command = "cls" if os.name == "nt" else "clear"
            os.system(clear_command)

            print(
                "\033[1m{:<11} {:<21} {:<21} {:<15} {:<9}\033[0m".format(
                    "process_id",
                    "camera_name",
                    "stream_name",
                    "execution_time",
                    "faces_detected",
                )
            )
            print("\033[1;37m{}\033[0m".format("=" * 90))

            # Print the data with appropriate formatting and colors
            for key, value in shared_dict.items():
                print(
                    "\033[92m{:<11} \033[0m {:<21} {:<21} {:<15} \033[0m {:<9}".format(
                        key,
                        value["camera_name"],
                        value["stream_name"],
                        value["execution_time"],
                        value["faces_detected"],
                    )
                )

            time.sleep(0.1)  # Delay for clarity


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
            logging.debug(f"Shared memory segment {shm_name} could not be closed because it was not found.")
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
        help="Specify this flag to skip camera updates",
    )

    parser.add_argument(
        "-M",
        "--monitor",
        action="store_true",
        default=False,
        help="Specify this flag to show monitoring info on the console during the process",
    )

    parser.add_argument(
        "--loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Specify the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    args = parser.parse_args()

    # Set the log level based on the provided --loglevel argument
    numeric_level = getattr(logging, args.loglevel, None)

    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.loglevel)

    logging.getLogger().setLevel(numeric_level)

    processes = []
    signal.signal(signal.SIGINT, signal_handler)

    # Use a manager for shared dictionary
    with Manager() as manager:
        shared_dict = manager.dict()

        cameras = get_cameras(update_cameras=args.update)
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
            p = Process(
                target=process_camera, args=(i, url, name, camera_id, shared_dict)
            )
            p.start()
            processes.append(p)

        monitor_p = Process(
            target=monitor_process_status, args=(shared_dict, args.monitor)
        )
        monitor_p.start()
        processes.append(monitor_p)

        for p in processes:
            p.join()

        # Close all shared memory segments
        close_threads(urls)
