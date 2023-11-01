from multiprocessing import shared_memory
from urls import urls
import config
import cv2
import numpy as np
import time

try:
    while True:
        for i, url in enumerate(urls):
            shm = shared_memory.SharedMemory(
                name=f"camera_thread_{i}", size=config.FRAME_SIZE_BYTES
            )

            frame = np.ndarray(
                (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3),
                dtype=np.uint8,
                buffer=shm.buf,
            )

            if frame is not None:
                cv2.imshow(f"camera_thread_{i}", frame)

                # Not workinc, use Ctrl+C
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

except Exception as e:
    print(e)
finally:
    cv2.destroyAllWindows()
