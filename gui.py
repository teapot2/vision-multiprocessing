from ttkthemes import ThemedTk
from tkinter import ttk
import tkinter as tk
import time


# Function for monitoring the status of the camera processes
def process_status_gui(shared_dict, monitor):
    """
    Monitor the status of the camera processes.

    Args:
        shared_dict: The shared dictionary containing camera stream information.
        monitor: A boolean value indicating whether to display monitoring information.

    Returns:
        None

    Continuously monitors and displays the status of the camera processes in the terminal.
    """

    def update_gui():
        if monitor:
            for widget in frame.winfo_children():
                widget.destroy()

            header_labels = [
                "ID",
                "Camera Name",
                "Stream Name",
                "Execution Time",
                "Faces Detected",
            ]

            for idx, label in enumerate(header_labels):
                ttk.Label(frame, text=label, font=("Arial", 12, "bold")).grid(
                    row=0, column=idx, padx=5, pady=5
                )

            row = 1
            for key, value in shared_dict.items():
                ttk.Label(frame, text=str(key)).grid(row=row, column=0, padx=5, pady=5)
                ttk.Label(
                    frame, text=value["camera_name"], anchor="w", justify="left"
                ).grid(row=row, column=1, padx=5, pady=5)
                ttk.Label(
                    frame, text=value["stream_name"], anchor="w", justify="left"
                ).grid(row=row, column=2, padx=5, pady=5)
                ttk.Label(frame, text=value["execution_time"]).grid(
                    row=row, column=3, padx=5, pady=5
                )
                ttk.Label(frame, text=str(value["faces_detected"])).grid(
                    row=row, column=4, padx=5, pady=5
                )
                row += 1

            root.after(100, update_gui)  # Update every 100 milliseconds

    root = ThemedTk(theme="")
    root.title("Camera Process Status")

    frame = ttk.Frame(root, padding=(5, 5, 5, 5))
    frame.grid(row=0, column=0, sticky="nsew")

    frame.pack_propagate(0)  # Disable propagation

    update_gui()

    root.mainloop()
