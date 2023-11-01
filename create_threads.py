from multiprocessing import Process, shared_memory, Manager
from urls import urls
import config
import cv2
import numpy as np
import random
import time
import os

generate_shm_stream_name = lambda index: f"camera_{index}_stream"


def send_frame_to_shared_memory(frame, shm):
    shared_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
    shared_array[:] = frame[:]


def process_camera(index, url, shared_dict):
    cap = cv2.VideoCapture(url)

    shm = shared_memory.SharedMemory(
        create=True, size=config.FRAME_SIZE_BYTES, name=generate_shm_stream_name(index)
    )

    while True:
        try:
            ret, frame = cap.read()

            if ret:
                start_time = time.time()

                # Normalize frame to allocate memory based on given specifications
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

                # ------------------ Vision Processing goes here ----------------- #

                send_frame_to_shared_memory(frame, shm)

                # Update the shared dictionary with the variable generated inside the process
                shared_dict[index] = {
                    "execution_time": f"{time.time() - start_time:.5f} s",
                    "camera_name": generate_shm_stream_name(index),
                    "faces_detected": 0,
                }

        except Exception as e:
            print(e)


def monitor_process_status(shared_dict):
    while True:
        os.system("cls" if os.name == "nt" else "clear")

        print(
            "\033[1m{:<11} {:<21} {:<15} {:<9}\033[0m".format(
                "process_id", "camera_name", "execution_time", "faces_detected"
            )
        )
        print("\033[1;37m{}\033[0m".format("=" * 70))

        for key, value in shared_dict.items():
            print(
                "\033[92m{:<11} \033[0m {:<21} \033[93m{:<15} \033[0m {:<9}".format(
                    key,
                    value["camera_name"],
                    value["execution_time"],
                    value["faces_detected"],
                )
            )

        time.sleep(0.1)  # Delay for clarity


def close_threads(urls):
    for i in range(len(urls)):
        try:
            shm = shared_memory.SharedMemory(name=generate_shm_stream_name(i))
            shm.close()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    processes = []

    with Manager() as manager:
        shared_dict = manager.dict()

        close_threads(urls)

        for i, url in enumerate(urls):
            p = Process(target=process_camera, args=(i, url, shared_dict))
            p.start()
            processes.append(p)

        monitor_p = Process(target=monitor_process_status, args=(shared_dict,))
        monitor_p.start()
        processes.append(monitor_p)

        for p in processes:
            p.join()

        close_threads(urls)
