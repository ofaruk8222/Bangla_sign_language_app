import os
import json

DATA_DIR = r"D:\4-I\Projects\Thesis\Sample testing with english\ASL Dataset"
METADATA_FILE = os.path.join(DATA_DIR, "WLASL_v0.3.json")

# Load metadata
with open(METADATA_FILE, 'r') as f:
    data = json.load(f)

print(f"Total glosses in JSON: {len(data)}")
print(f"\nFirst gloss: {data[0]['gloss']}")
print(f"Number of instances: {len(data[0]['instances'])}")
print(f"First video_id: {data[0]['instances'][0]['video_id']}")

# Check what files actually exist
all_files = os.listdir(DATA_DIR)
mp4_files = [f for f in all_files if f.endswith('.mp4')]
print(f"\nTotal .mp4 files in folder: {len(mp4_files)}")
if mp4_files:
    print(f"First 5 files: {mp4_files[:5]}")

# Check if the first video exists
first_video_id = str(data[0]['instances'][0]['video_id'])
first_video_path = os.path.join(DATA_DIR, f"{first_video_id}.mp4")
print(f"\nLooking for: {first_video_path}")
print(f"File exists: {os.path.exists(first_video_path)}")