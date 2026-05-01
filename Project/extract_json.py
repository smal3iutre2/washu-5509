import json
import os

# This script extracts 6 camera images from a specified sample in the dataset and saves them to a new folder.
json_dir = r"C:\Users\xuyub\Desktop\cvProject\v1.0-mini\v1.0-mini"




try:
    with open(os.path.join(json_dir, 'sample_data.json'), 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
        
    with open(os.path.join(json_dir, 'calibrated_sensor.json'), 'r', encoding='utf-8') as f:
        calibrated_sensors = json.load(f)
except FileNotFoundError:
    print("can't find the json files, please check your paths.")
    exit()


cameras = [
    "CAM_FRONT", "CAM_BACK", 
    "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
    "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]



for cam_name in cameras:
    
    sensor_token = None
    for sd in sample_data:
       
        if cam_name in sd.get('filename', ''):  
            sensor_token = sd['calibrated_sensor_token']
            break

   
    if sensor_token:
        for cs in calibrated_sensors:
            if cs['token'] == sensor_token:
                print(f"================ {cam_name} ================")
                print(f"real_intrinsic = {cs['camera_intrinsic']}")
                print(f"real_translation = {cs['translation']}")
                print(f"real_rotation = {cs['rotation']}\n")
                break