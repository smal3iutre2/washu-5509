import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO
from sklearn.cluster import DBSCAN


def get_real_perspective_matrix(intrinsic, rotation_quat, translation):
    K = np.array(intrinsic)
    T_sensor2ego = np.array(translation).reshape(3, 1)
    w, x, y, z = rotation_quat
    R_sensor2ego = R.from_quat([x, y, z, w]).as_matrix()
    R_ego2sensor = R_sensor2ego.T
    T_ego2sensor = -np.dot(R_ego2sensor, T_sensor2ego)
    M = np.column_stack((R_ego2sensor[:, 0], R_ego2sensor[:, 1], T_ego2sensor[:, 0]))
    H = np.dot(K, M)
    return np.linalg.inv(H)

def pixel_to_ground(u, v, H_inv):
    pixel_pt = np.array([u, v, 1.0])
    ground_pt = np.dot(H_inv, pixel_pt)
    X = ground_pt[0] / ground_pt[2] 
    Y = ground_pt[1] / ground_pt[2] 
    return X, Y


CAM_PARAMS = {
    "CAM_FRONT": {
        "intrinsic": [[1266.417203046554, 0.0, 816.2670197447984], [0.0, 1266.417203046554, 491.50706579294757], [0.0, 0.0, 1.0]],
        "translation": [1.70079118954, 0.0159456324149, 1.51095763913],
        "rotation": [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755],
        "color": (0, 255, 0) 
    },
    "CAM_BACK": {
        "intrinsic": [[1259.5137405846733, 0.0, 807.2529053838625], [0.0, 1259.5137405846733, 501.19579884916527], [0.0, 0.0, 1.0]],
        "translation": [1.0148780988, -0.480568219723, 1.56239545128],
        "rotation": [0.12280980120078765, -0.132400842670559, -0.7004305821388234, 0.690496031265798],
        "color": (0, 0, 255) 
    },
    "CAM_FRONT_LEFT": {
        "intrinsic": [[1272.5979470598488, 0.0, 826.6154927353808], [0.0, 1272.5979470598488, 479.75165386361925], [0.0, 0.0, 1.0]],
        "translation": [1.52387798135, 0.494631336551, 1.50932822144],
        "rotation": [0.6757265034669446, -0.6736266522251881, 0.21214015046209478, -0.21122827103904068],
        "color": (255, 255, 0) 
    },
    "CAM_FRONT_RIGHT": {
        "intrinsic": [[1260.8474446004698, 0.0, 807.968244525554], [0.0, 1260.8474446004698, 495.3344268742088], [0.0, 0.0, 1.0]],
        "translation": [1.5508477543, -0.493404796419, 1.49574800619],
        "rotation": [0.2060347966337182, -0.2026940577919598, 0.6824507824531167, -0.6713610884174485],
        "color": (0, 255, 255) 
    },
    "CAM_BACK_LEFT": {
        "intrinsic": [[1256.7414812095406, 0.0, 792.1125740759628], [0.0, 1256.7414812095406, 492.7757465151356], [0.0, 0.0, 1.0]],
        "translation": [1.03569100218, 0.484795032713, 1.59097014818],
        "rotation": [0.6924185592174665, -0.7031619420114925, -0.11648342771943819, 0.11203317912370753],
        "color": (255, 0, 255) 
    },
    "CAM_BACK_RIGHT": {
        "intrinsic": [[1259.5137405846733, 0.0, 807.2529053838625], [0.0, 1259.5137405846733, 501.19579884916527], [0.0, 0.0, 1.0]],
        "translation": [1.0148780988, -0.480568219723, 1.56239545128],
        "rotation": [0.12280980120078765, -0.132400842670559, -0.7004305821388234, 0.690496031265798],
        "color": (255, 100, 100) 
    }
}



scene_idx = input("please enter the scene index you want to visualize (e.g., 10): ")
scene_folder = f"./cvProject/saved_scenes/scene_{scene_idx}"

if not os.path.exists(scene_folder):
    print(f"can't find the scene folder: {scene_folder}. Please run get_6_cams.py first to extract the 6 camera images for the specified scene index.")
    exit()
print(f"reading {scene_folder}")

model = YOLO("yolov8n.pt")

MAP_SIZE = 1200
EGO_X, EGO_Y = MAP_SIZE // 2, MAP_SIZE // 2 
PIXELS_PER_METER = 12 

bev_map = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)

cv2.line(bev_map, (EGO_X, 0), (EGO_X, MAP_SIZE), (50, 50, 50), 1)
cv2.line(bev_map, (0, EGO_Y), (MAP_SIZE, EGO_Y), (50, 50, 50), 1)
for radius_m in range(10, 60, 10):
    cv2.circle(bev_map, (EGO_X, EGO_Y), radius_m * PIXELS_PER_METER, (60, 60, 60), 1)

    cv2.putText(bev_map, f"{radius_m}m", (EGO_X + 10, EGO_Y - radius_m * PIXELS_PER_METER - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)


cv2.rectangle(bev_map, (EGO_X - 15, EGO_Y - 30), (EGO_X + 15, EGO_Y + 30), (255, 255, 255), -1)
cv2.putText(bev_map, "EGO", (EGO_X - 25, EGO_Y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


y_offset = 50
cv2.putText(bev_map, "Sensor Legend:", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
for cam, params in CAM_PARAMS.items():
    cv2.circle(bev_map, (45, y_offset), 10, params['color'], -1)
    cv2.putText(bev_map, cam, (65, y_offset + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    y_offset += 40


annotated_images = {}
all_detections = []

for cam_name, params in CAM_PARAMS.items():
    # image_path = f"{cam_name}.jpg"
    image_path = os.path.join(scene_folder, f"{cam_name}.jpg")
    img = cv2.imread(image_path)
    if img is None:
        img = np.zeros((900, 1600, 3), dtype=np.uint8)

    H_inv = get_real_perspective_matrix(params['intrinsic'], params['rotation'], params['translation'])
    dot_color = params['color']

    results = model(img, verbose=False)

    for box in results[0].boxes:
        class_name = model.names[int(box.cls[0].item())]
        if class_name in ['car', 'truck', 'bus']:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, y2

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), dot_color, 4)
            cv2.putText(img, cam_name.replace("CAM_", ""), (int(x1), int(y1)-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, dot_color, 3)
            # Apply Inverse Perspective Mapping (IPM):
            # Transform the 2D image coordinate (tire contact point where Z=0) into absolute 3D physical ground coordinates.
            # Returns longitudinal (X) and lateral (Y) distances in meters relative to the Ego vehicle.
            real_x_meters, real_y_meters = pixel_to_ground(cx, cy, H_inv)

            if abs(real_x_meters) < 50 and abs(real_y_meters) < 50:
                map_y = int(EGO_Y - (real_x_meters * PIXELS_PER_METER))
                map_x = int(EGO_X - (real_y_meters * PIXELS_PER_METER))

                all_detections.append({
                    "x": real_x_meters,
                    "y": real_y_meters,
                    "map_x": map_x,
                    "map_y": map_y,
                    "camera": cam_name,
                    "class": class_name,
                    "color": dot_color
                })

                cv2.circle(bev_map, (map_x, map_y), 5, dot_color, -1)

   
    annotated_images[cam_name] = cv2.resize(img, (800, 450))

# ===============================
# DBSCAN Fusion
# ===============================
if len(all_detections) > 0:
    points = np.array([[d["x"], d["y"]] for d in all_detections])

    clustering = DBSCAN(eps=2.0, min_samples=1).fit(points)
    labels = clustering.labels_

    for label in sorted(set(labels)):
        idxs = np.where(labels == label)[0]
        cluster_points = points[idxs]

        fused_x = np.mean(cluster_points[:, 0])
        fused_y = np.mean(cluster_points[:, 1])

        fused_map_y = int(EGO_Y - fused_x * PIXELS_PER_METER)
        fused_map_x = int(EGO_X - fused_y * PIXELS_PER_METER)

        cv2.line(bev_map, (EGO_X, EGO_Y), (fused_map_x, fused_map_y), (180, 180, 180), 2)

        cv2.circle(bev_map, (fused_map_x, fused_map_y), 14, (255, 255, 255), 2)
        cv2.circle(bev_map, (fused_map_x, fused_map_y), 6, (255, 255, 255), -1)

        cams = [all_detections[i]["camera"].replace("CAM_", "") for i in idxs]
        cams_text = ",".join(sorted(set(cams)))

        cv2.putText(
            bev_map,
            f"FUSED {label} ({cams_text})",
            (fused_map_x + 15, fused_map_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

row1 = cv2.hconcat([annotated_images["CAM_FRONT_LEFT"], annotated_images["CAM_FRONT"], annotated_images["CAM_FRONT_RIGHT"]])
row2 = cv2.hconcat([annotated_images["CAM_BACK_LEFT"], annotated_images["CAM_BACK"], annotated_images["CAM_BACK_RIGHT"]])
#
cams_panel = cv2.vconcat([row1, row2])


radar_panel = np.zeros((1200, 2400, 3), dtype=np.uint8)

radar_panel[0:1200, 600:1800] = bev_map 


dashboard = cv2.vconcat([cams_panel, radar_panel])


cv2.putText(dashboard, "AUTONOMOUS DRIVING 360 COMMAND CENTER", (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)


# save_path = "HD_Dashboard_Presentation.jpg"
save_path = os.path.join(scene_folder, f"HD_Dashboard_Scene_{scene_idx}.jpg")
# cv2.imwrite(save_path, dashboard)
cv2.imwrite(save_path, dashboard)
print(f"successfully saved to: {save_path}")



cv2.namedWindow("Preview (Press any key to close)", cv2.WINDOW_NORMAL)

cv2.imshow("Preview (Press any key to close)", cv2.resize(dashboard, (1200, 1050)))
cv2.waitKey(0)
cv2.destroyAllWindows()