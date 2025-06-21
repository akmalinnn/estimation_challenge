import os

# Directory where the files are located (use "." for current directory)
directory = "flow_diff"

for filename in os.listdir(directory):
    if "_comma" in filename:
        new_name = filename.replace("_comma", "")
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")
