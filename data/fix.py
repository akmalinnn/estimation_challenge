import os
import json

# Define file paths
folder_path = "flow"
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

# Get list of .npy files in the folder
npy_files = {file for file in all_files if file.endswith(".npy")}

# Get valid filenames from JSON (convert .jpg to .npy)
valid_npy_files = {file.replace(".jpg", ".npy") for file in json_data.keys()}

# Find .npy files that are not in the JSON and delete them
for file in npy_files:
    if file not in valid_npy_files:
        os.remove(os.path.join(folder_path, file))
        print(f"Deleted: {file}")

print("Cleanup completed.")
