import json
from pathlib import Path
import random
from collections import OrderedDict
import re

def create_video_aware_json(of_dir, label_json, output_json):
    of_dir = Path(of_dir)
    labels = json.load(open(label_json))
    files = sorted(of_dir.glob("diff_flow_*.npy"))  # Get files sorted by name
    
    result = OrderedDict()
    samples = []
    
    # Extract video identifiers (assuming format like diff_flow_videoID_frameXXXX.npy)
    def get_video_id(filename):
        match = re.match(r'diff_flow_(.*?)_frame_\d+\.png\.npy', filename.name)
        return match.group(1) if match else None
    
    # Process files in order, skipping across video boundaries
    i = 0
    while i < len(files) - 2:
        file1, file2, file3 = files[i], files[i+1], files[i+2]
        
        # Check if all three files are from same video
        vid1 = get_video_id(file1)
        vid2 = get_video_id(file2)
        vid3 = get_video_id(file3)
        
        if vid1 == vid2 == vid3:
            img1 = file1.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"
            img2 = file2.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"
            img3 = file3.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"
            
            if all(img in labels for img in [img1, img2, img3]):
                speed_diff = ((labels[img2]["speed"] + labels[img3]["speed"]) / 2) - \
                            ((labels[img1]["speed"] + labels[img2]["speed"]) / 2)
                samples.append((file1.name, float(speed_diff)))
                i += 1  # Move to next frame
            else:
                i += 1  # Missing label, skip
        else:
            i += 1  # Different video, skip
    
    # Random split assignment
    random_indices = list(range(len(samples)))
    random.shuffle(random_indices)
    
    train_end = int(0.7 * len(samples))
    val_end = train_end + int(0.2 * len(samples))
    
    split_types = []
    for i in range(len(samples)):
        if i < train_end:
            split_types.append("training")
        elif i < val_end:
            split_types.append("validation")
        else:
            split_types.append("testing")
    
    # Create sorted output with random splits
    for idx, (filename, speed) in enumerate(samples):
        result[filename] = {
            "speed": speed,
            "data_type": split_types[random_indices[idx]]
        }
    
    # Write sorted JSON
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Created video-aware JSON with {len(samples)} samples")
    print(f"Training: {train_end}, Validation: {val_end-train_end}, Testing: {len(samples)-val_end}")


create_video_aware_json(
    of_dir="comma_flow_diff",
        label_json="data_filtered_copy.json",
        output_json="flow_diff.json"
)

