import matplotlib.pyplot as plt
import tkinter as tk
import logging
from collections import deque
from ttkthemes import ThemedTk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)

# Use a deque to store historical execution times for each iteration
historical_execution_times = {}


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

            fig, ax = plt.subplots()

            for key, value in shared_dict.items():
                # Plot historical execution times specific to each row
                execution_time = float(value["execution_time"])

                if key not in historical_execution_times:
                    historical_execution_times[key] = deque(maxlen=10)

                # Append the execution time to the deque
                historical_execution_times[key].append(execution_time)

                # Plot the historical execution times for each row
                ax.plot(
                    list(historical_execution_times[key]), marker="o", label=str(key)
                )

                ttk.Label(frame, text=str(key)).grid(
                    row=key + 1, column=0, padx=5, pady=5
                )
                ttk.Label(
                    frame, text=value["camera_name"], anchor="w", justify="left"
                ).grid(row=key + 1, column=1, padx=5, pady=5)
                ttk.Label(
                    frame, text=value["stream_name"], anchor="w", justify="left"
                ).grid(row=key + 1, column=2, padx=5, pady=5)
                ttk.Label(frame, text=value["execution_time"]).grid(
                    row=key + 1, column=3, padx=5, pady=5
                )
                ttk.Label(frame, text=str(value["faces_detected"])).grid(
                    row=key + 1, column=4, padx=5, pady=5
                )

            ax.set_title("Historical Execution Times")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Execution Time (s)")
            ax.legend()

            # Create a Tkinter canvas to embed the Matplotlib plot
            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=1, column=0)

            # Update every 100 milliseconds
            root.after(100, update_gui)

    if monitor:
        root = ThemedTk(theme="")
        root.title("Camera Process Status")

        frame = ttk.Frame(root, padding=(5, 5, 5, 5))
        frame.grid(row=0, column=0, sticky="nsew")

        frame.pack_propagate(0)  # Disable propagation

        update_gui()

        root.mainloop()
