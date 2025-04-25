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

# Extract unique video identifiers from JSON keys (ignoring frame numbers)
json_videos = {re.match(r"(\d+_\d+)", key).group(1) for key in json_data.keys() if re.match(r"(\d+_\d+)", key)}

# Get list of .npy files in the folder
npy_files = {file for file in all_files if file.endswith(".npy")}

# Extract video identifiers from .npy filenames (ignoring frame numbers)
def extract_video_id(npy_filename):
    match = re.search(r"diff_flow_(\d+_\d+)", npy_filename)
    return match.group(1) if match else None

# Find .npy files that should be kept
valid_npy_files = {file for file in npy_files if extract_video_id(file) in json_videos}

# Delete unwanted .npy files
for file in npy_files - valid_npy_files:
    os.remove(os.path.join(folder_path, file))
    print(f"Deleted: {file}")

# Remove JSON keys if no corresponding .npy file exists
valid_video_ids = {extract_video_id(file) for file in valid_npy_files}
json_data = {key: value for key, value in json_data.items() if re.match(r"(\d+_\d+)", key).group(1) in valid_video_ids}

# Save updated JSON file
with open(json_file, "w") as f:
    json.dump(json_data, f, indent=4)

print("Cleanup completed.")
