# BEV Pure Vision Panoramic Perception System (v1.0)

This project achieves 360° Bird's Eye View (BEV) semantic fusion with zero distortion by leveraging nuScenes camera calibration parameters, YOLOv8 object detection, and Inverse Perspective Mapping (IPM), eliminating traditional image stitching techniques.

### 1. Environment and Dependency Setup
Please download the v1.0-mini dataset from the official nuScenes website (https://www.nuscenes.org/download) and extract it into the directory `./data/sets/nuscenes/` before running the code.
Make sure your dir looks like this
```text data/
└── sets/
    └── nuscenes/
        ├── maps/
        ├── samples/
        ├── sweeps/
        ├── v1.0-mini/
        ├── .gitkeep
        ├── .v1.0-mini.txt
        └── LICENSE
```

1. Ensure Python 3.8+ is installed.

2. Run the following command in terminal to install all dependencies at once:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the root directory contains the `v1.0-mini` dataset folder and the YOLO weight file yolov8n.pt (if the weight file is missing, it will be automatically downloaded on the first run).

### 2. Core Architecture Design

This project follows a minimalist "dual-script" pipeline:

- get_6_cams.py: **Data Synchronization & Sampler**. Responsible for parsing underlying JSON data and precisely extracting 6-view keyframes at the same absolute timestamp.
- 360_fusion.py: **Core Spatial Fusion Engine**. Handles 2D object detection and IPM 3D projection mapping.

### 3. Execution Guide (Standard Workflow)

**Step 1: Extract Synchronized Scene** Run in terminal:

```bash
python get_6_cams.py
```

- **Operation:** Follow the terminal prompt to enter a number between 0-403, or press Enter directly to trigger a random blind-box selection.
- **Output:** The system automatically generates a saved_scenes/scene_number folder in the project and stores 6 keyframes captured within the same millisecond.

**Step 2: Activate Panoramic Perception and Fusion** Run in terminal:

```bash
python 360_fusion.py
```

- **Operation:** Follow the terminal prompt to enter the scene number you extracted earlier (e.g., enter 123).
- **Output:** The system reads the images in that folder for computation and generates a radar chart (HD_Dashboard_Scene_xxx.jpg) in the saved_scenes/scene_number folder.

### 4. Result Verification

Double-click to open the generated image, and you will see:

1. **Top Console:** Original footage from 6 cameras and YOLO semantic detection boxes.
2. **Bottom Radar Chart:** All objects are mapped to the 3D physical ground plane with the ego vehicle (Ego) as the absolute center.
