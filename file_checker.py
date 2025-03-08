import os
import json

# Define file paths
folder_path = "flow"
json_file = "data_filtered.json"

# Load JSON data
with open(json_file, "r") as f:
    json_data = json.load(f)

# Get list of jpg files in the folder
folder_files = {file for file in os.listdir(folder_path) if file.endswith(".jpg")}

# Get list of jpg files from JSON
json_files = set(json_data.keys())

# Find discrepancies
missing_from_json = sorted(folder_files - json_files)  # In folder but not in JSON
missing_from_folder = sorted(json_files - folder_files)  # In JSON but not in folder

# Print results
if missing_from_json:
    print("Files in 'flow' folder but NOT in 'data_filtered.json':")
    for file in missing_from_json:
        print(file)
else:
    print("All files in 'flow' folder are present in 'data_filtered.json'.")

if missing_from_folder:
    print("Files listed in 'data_filtered.json' but NOT found in 'flow' folder:")
    for file in missing_from_folder:
        print(file)
else:
    print("All files in 'data_filtered.json' are present in 'flow' folder.")
