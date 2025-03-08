import json

# Load the JSON file
with open("combined.json", "r") as f:
    data = json.load(f)

filtered_data = {}
sequence_tracker = {}

for key in sorted(data.keys()):  # Sorting to process in order
    sequence_id = key.rsplit("_", 1)[0]  # Extract the sequence part (e.g., "00_0003_frame")

    if sequence_id not in sequence_tracker:
        sequence_tracker[sequence_id] = True  # Mark first occurrence, but skip adding it
        continue  # Skip the first frame

    filtered_data[key] = data[key]  # Add all non-first frames

# Save the filtered JSON
with open("combined_filtered.json", "w") as f:
    json.dump(filtered_data, f, indent=4)

print("Processed successfully!")
