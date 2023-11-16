import os


class Config:
    # Width of the video frame
    FRAME_WIDTH = 854

    # Height of the video frame
    FRAME_HEIGHT = 480

    # Size of the video frame in bytes (width * height * 3 channels for RGB)
    FRAME_SIZE_BYTES = FRAME_HEIGHT * FRAME_WIDTH * 3

    # Interval for video segmentation
    VIDEO_SEGMENTATION_INTERVAL = 10

    # Interval for processing segmentation
    PROCESSING_SEGMENTATION_INTERVAL = 0.3

    # URL for accessing camera API
    CAMERA_API_URL = "http://localhost:8000/api/cameras/"

    # Path for the recognition database
    RECOGNITION_DB_PATH = os.path.join("data", "db")
