# Camera Stream Processing

This Python script processes camera streams, stores video data, and monitors the camera processes in a graphical user interface (GUI).

## Prerequisites

- Python

## Installation

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can run the script with the following command:

```bash
python create_threads.py
```

### Command-line arguments

- `-U, --update`: Specify this flag to skip camera updates.
- `-M, --monitor`: Specify this flag to show monitoring information on a GUI during the process.
- `--loglevel`: Specify the log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- `--offline`: Use offline URLs (Saved videos, local streams, etc.)

Example:

```bash
python main_script.py -U -M --loglevel DEBUG
```

## Features

- Capable of handling camera streams from multiple sources simultaneously.
- Provides local storage functionality for video data.
- Offers a convenient graphical user interface (GUI) for monitoring the status of camera processes.

## Authors

- [Sebastián Segovia](https://github.com/teapot2)