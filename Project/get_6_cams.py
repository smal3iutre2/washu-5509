import json
import os
import shutil
import random
# ==========================================
# This script extracts 6 camera images from a specified sample in the dataset and saves them to a new folder.
# please ensure the paths to the json files and dataset root are correctly set before running the script.
# ==========================================
json_dir = r".\data\sets\nuscenes\v1.0-mini"
dataset_root = r".\data\sets\nuscenes"
dest_dir = r".\cvProject"


try:
    with open(os.path.join(json_dir, 'sample.json'), 'r', encoding='utf-8') as f:
        samples = json.load(f)
    with open(os.path.join(json_dir, 'sample_data.json'), 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
except FileNotFoundError:
    print("can't find the json files, please check your paths.")
    exit()


max_idx = len(samples) - 1
print(f" the current dataset contains {max_idx + 1} keyframes.")
user_input = input(f"please enter the sample index (0 - {max_idx}), or press Enter for random selection: ")

if user_input.strip() == "":

    sample_idx = random.randint(0, max_idx)
    print(f" random sample index: {sample_idx}")
else:

    try:
        sample_idx = int(user_input)
        if sample_idx < 0 or sample_idx > max_idx:
            print(f"unvalid index, using default sample index 10.")
            sample_idx = 10
    except ValueError:
        print("unvalid input, using default sample index 10.")
        sample_idx = 10
        
target_sample = samples[sample_idx] 
target_token = target_sample['token']


scene_folder = os.path.join(dest_dir, f"saved_scenes", f"scene_{sample_idx}")
os.makedirs(scene_folder, exist_ok=True) 


print(f"saved in: {scene_folder}\n")

cams_found = 0
for sd in sample_data:
    if sd['sample_token'] == target_token and sd.get('is_key_frame') and "CAM" in sd.get('filename', ''):
        
        sensor_name = ""
        sensor_list = ["CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT", "CAM_BACK"]
        for name in sensor_list:
            if name in sd['filename']:
                sensor_name = name
                break 
        
        if sensor_name:
            original_img_path = os.path.join(dataset_root, sd['filename'])
            new_img_path = os.path.join(scene_folder, f"{sensor_name}.jpg")
            
            try:
                shutil.copy(original_img_path, new_img_path)
                print(f"successfully extracted to folder: {sensor_name}.jpg")
                cams_found += 1
            except FileNotFoundError:
                print(f" original image not found: {original_img_path}")
                
        if cams_found == 6:
            break

