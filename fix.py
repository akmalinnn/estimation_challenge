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

# Delete files that are not in JSON
for file in missing_from_json:
    os.remove(os.path.join(folder_path, file))
    print(f"Deleted: {file}")

# Remove entries from JSON that are missing from folder
for file in missing_from_folder:
    del json_data[file]

# Save the fixed JSON
with open(json_file, "w") as f:
    json.dump(json_data, f, indent=4)

print("Cleanup completed.")
