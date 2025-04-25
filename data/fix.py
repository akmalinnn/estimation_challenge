import os
import json
import re

# Define file paths
folder_path = "flow_diff"
json_file = "data_filtered.json"

# Load JSON data
with open(json_file, "r") as f:
    json_data = json.load(f)

# Get list of all files in the folder
all_files = os.listdir(folder_path)

# Delete all .jpg files
for file in all_files:
    if file.endswith(".jpg"):
        os.remove(os.path.join(folder_path, file))
        print(f"Deleted: {file}")

# Extract unique video identifiers from JSON keys
json_videos = {re.match(r"(\d+_\d+)_frame", key).group(1) for key in json_data.keys()}

# Get list of .npy files in the folder
npy_files = {file for file in all_files if file.endswith(".npy")}

# Extract video identifiers from .npy filenames
def extract_video_id(npy_filename):
    match = re.search(r"diff_flow_(\d+_\d+)_frame", npy_filename)
    return match.group(1) if match else None

# Find .npy files that should be kept
valid_npy_files = {file for file in npy_files if extract_video_id(file) in json_videos}

# Delete unwanted .npy files
for file in npy_files - valid_npy_files:
    os.remove(os.path.join(folder_path, file))
    print(f"Deleted: {file}")

print("Cleanup completed.")
