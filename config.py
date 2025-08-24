# config.py
import os

# Get the absolute path of the main project directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to the data folder
DATA_PATH = os.path.join(ROOT_DIR, 'data')

# --- THIS IS THE MISSING VARIABLE ---
# Make sure this line exists and is spelled correctly
AVENUE_VIDEO_PATH = os.path.join(DATA_PATH, 'Avenue Dataset', 'testing_videos', '01.avi')

TEST_VIDEO_PATH = os.path.join(DATA_PATH, 'test_video.mp4')