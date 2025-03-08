import json

# Load the JSON file
with open("filtered_combined.json", "r") as f:
    data = json.load(f)

filtered_data = data.copy()  # Start with a copy of the original data
sequence_frames = {}

# Group frames by sequence
for key in data.keys():
    sequence_id = key.rsplit("_", 1)[0]  # Extract sequence part (e.g., "00_0003_frame")
    if sequence_id not in sequence_frames:
        sequence_frames[sequence_id] = []
    sequence_frames[sequence_id].append(key)

# Remove the last frame from each sequence
for sequence_id, frames in sequence_frames.items():
    last_frame = max(frames)  # Get the last frame (assuming lexicographical order)
    filtered_data.pop(last_frame, None)  # Remove it from the data

# Save the modified JSON
with open("data_filtered.json", "w") as f:
    json.dump(filtered_data, f, indent=4)

print("Last frames removed successfully!")
