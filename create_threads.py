from multiprocessing import Process, shared_memory
from urls import urls
import config
import cv2
import numpy as np
import random
import time


def send_frame_to_shared_memory(frame, shm):
    shared_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
    shared_array[:] = frame[:]


def process_camera(index, shm_name, url):
    cap = cv2.VideoCapture(url)

    shm = shared_memory.SharedMemory(
        create=True, size=config.FRAME_SIZE_BYTES, name=shm_name
    )

    while True:
        try:
            ret, frame = cap.read()

            if ret:
                start_time = time.time()

                # Normalize frame to allocate memory based on given specifications
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

                # Vision Processing goes here

                print(
                    shm.name,
                    " |-| ",
                    f"Executed in {time.time() - start_time:.5f} s",
                    " |-| ",
                    frame.nbytes,
                    frame.dtype,
                    frame.shape,
                )

                send_frame_to_shared_memory(frame, shm)

        except Exception as e:
            print(e)


def closeThreads(urls):
    for i in range(len(urls)):
        try:
            shm = shared_memory.SharedMemory(name=f"camera_thread_{i}")
            shm.close()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    processes = []

    closeThreads(urls)

    for i, url in enumerate(urls):
        shm_name = f"camera_thread_{i}"
        p = Process(target=process_camera, args=(i, shm_name, url))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    closeThreads(urls)
