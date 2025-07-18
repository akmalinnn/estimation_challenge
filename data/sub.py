import os
import shutil

flow_diff_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow_diff")
for subdir, _, files in os.walk(flow_diff_dir):
    if subdir == flow_diff_dir:
        continue  

    for file in files:
        if file.endswith(".npy"):
            src_path = os.path.join(subdir, file)
            dest_path = os.path.join(flow_diff_dir, file)
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(file)
                dest_path = os.path.join(flow_diff_dir, f"{name}_{counter}{ext}")
                counter += 1

            shutil.move(src_path, dest_path)
            

    if not os.listdir(subdir):
        os.rmdir(subdir)
        print(f"Removed empty folder: {subdir}")


