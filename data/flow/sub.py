import os
import shutil

# Get the directory where Main.py is located
main_dir = os.path.dirname(os.path.abspath(__file__))

# Iterate through all subdirectories
for subdir, _, files in os.walk(main_dir):
    if subdir == main_dir:
        continue  # Skip the main directory itself

    for file in files:
        src_path = os.path.join(subdir, file)
        dest_path = os.path.join(main_dir, file)

        # Ensure no name conflict
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(file)
            dest_path = os.path.join(main_dir, f"{name}_{counter}{ext}")
            counter += 1

        shutil.move(src_path, dest_path)

    # Remove the empty folder
    os.rmdir(subdir)

print("All files have been moved successfully!")
