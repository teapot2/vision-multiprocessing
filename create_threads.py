from multiprocessing import Process, shared_memory, Manager
from api import get_cameras, ping_cameras
import config
import cv2
import numpy as np
import random
import time
import os
import signal
import sys


from ultralytics import YOLO
from sort import Sort
from face_detection import Face_Detection
from face_match import Face_Match
from face_verify import Face_Verif

# Lambda function for generating shared memory stream names based on index
generate_shm_stream_name = lambda index: f"camera_{index}_stream"


model = YOLO("yolov8n-pose.pt")
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

    print(
        "\n\033[91mInterruption detected. Cleaning up processes and threads...\033[0m"
    )

    for p in processes:
        p.terminate()

    close_threads(urls)
    sys.exit(0)


# Function for processing camera streams
def process_camera(index, url, name, shared_dict):
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
    
    #cap = cv2.VideoCapture()

    cap = cv2.VideoCapture(url)

    # Establecer la frecuencia de cuadros objetivo
    fps_target = 1
    time_per_frame = 1.0 / fps_target


    try:
        # Create a shared memory segment for the frame
        shm = shared_memory.SharedMemory(
            create=True,
            size=config.FRAME_SIZE_BYTES,
            name=generate_shm_stream_name(index),
        )
    except FileExistsError:
        print(
            f"\033[93m[WARNING] Shared memory segment already exists for camera at index {index}\033[0m"
        )
        shm = shared_memory.SharedMemory(name=generate_shm_stream_name(index))

    try:
        while True:
            ret, frame = cap.read()

            if ret:
                start_time = time.time()

                # Normalize frame to specified dimensions
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

                 # Calcular cuánto tiempo tomar el procesamiento y dormir si es necesario
                processing_time = time.time() - start_time
                time_to_wait = time_per_frame - processing_time
                if time_to_wait > 0:
                    time.sleep(time_to_wait)

                    #! -----------------------------------------------------------------------------------
                    # Vision processing logic goes here
                    frame = cv2.flip(frame, -1)
                    
                    results = model(frame, stream=True)
                    
                    for res in results:
                        filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.75)[0]
                        boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
                        tracks = tracker.update(boxes)
                        tracks = tracks.astype(int)
                        print(tracks)
                        
                        
                        if tracks is not None:
                            print('------------------------------------------------------------')
                            
                            for xmin, ymin, xmax, ymax, track_id in tracks:
                                # Proceso de detección facial
                                ''' 
                                print(xmin, ymin, xmax, ymax)
                                id, face = Face_Detection (xmin, ymin, xmax, ymax, frame)
                                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                                print(face, 'Cara: ')
                                
                                if face is not None:
                                    # Proceso de buscar un match
                                    print('Estamos revisando quien eres!')
                                    
                                    id, face, match = Face_Match(id, face)
                                    
                                    # Proceso verifica si es la misma persona
                                    print('Se esta verificando su rostro')
                                    track_id = Face_Verif(id, face, match)

                                    
                                else:
                                    print('No he detectado una cara') '''
                                
                                cv2.putText(img=frame, text=f"Id: {track_id}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
                                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)
                
                   
                #! -----------------------------------------------------------------------------------
                send_frame_to_shared_memory(frame, shm)

                # Update the shared dictionary with relevant information
                shared_dict[index] = {
                    "execution_time": f"{time.time() - start_time:.5f} s",
                    "camera_name": name,
                    "stream_name": generate_shm_stream_name(index),
                    "faces_detected": 0,
                }

    except Exception as e:
        print(
            f"\033[91mError occurred while processing camera at index {index}: {e}\033[0m"
        )

        # Close the shared memory segment in case of an error
        shm.close()
        shm.unlink()


# Function for monitoring the status of the camera processes
def monitor_process_status(shared_dict):
    """
    Monitor the status of the camera processes.

    Args:
        shared_dict: The shared dictionary containing camera stream information.

    Returns:
        None

    Continuously monitors and displays the status of the camera processes in the terminal.
    """

    while True:
        os.system("cls" if os.name == "nt" else "clear")

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

    for i in range(len(urls)):
        try:
            shm = shared_memory.SharedMemory(name=generate_shm_stream_name(i))
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    processes = []

    signal.signal(signal.SIGINT, signal_handler)

    # Use a manager for shared dictionary
    with Manager() as manager:
        shared_dict = manager.dict()

        cameras = get_cameras(skip_update=True)

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

        # Close any existing threads
        close_threads(urls)

        for i, (url, name) in enumerate(zip(urls, names)):
            p = Process(target=process_camera, args=(i, url, name, shared_dict))
            p.start()
            processes.append(p)

        monitor_p = Process(target=monitor_process_status, args=(shared_dict,))
        monitor_p.start()
        processes.append(monitor_p)

        for p in processes:
            p.join()

        # Close all shared memory segments
        close_threads(urls)
