import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d

# Load rectified stereo images (grayscale)
left = cv.imread("rectified_left.jpg", cv.IMREAD_GRAYSCALE)
right = cv.imread("rectified_right.jpg", cv.IMREAD_GRAYSCALE)
color = cv.imread("rectified_left.jpg")  # For point cloud color

# Stereo parameters
focal_length = 3154.3    # in pixels
baseline = 0.06          # in meters (6 cm)

# Compute disparity using StereoSGBM
stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16,       # Must be divisible by 16
    blockSize=9,             # Tuneable: smaller = more detail
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
)
disparity = stereo.compute(left, right).astype(np.float32) / 16.0

# Disparity filtering
disparity[disparity < 1] = np.nan  # Prevent division by zero or huge depths

# Depth calculation using the formula
depth_map = (focal_length * baseline) / disparity

# Save disparity and depth map visualizations
disp_vis = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
disp_vis = np.uint8(disp_vis)
cv.imwrite("disparity.jpg", disp_vis)

depth_vis = cv.normalize(np.nan_to_num(depth_map), None, 0, 255, cv.NORM_MINMAX)
depth_vis = np.uint8(depth_vis)
color_map = cv.applyColorMap(depth_vis, cv.COLORMAP_JET)
cv.imwrite("depth_map_colored.jpg", color_map)

# --------- Depth Analysis ---------

# 1. Click to view depth at a pixel
def show_depth(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        z = depth_map[y, x]
        if np.isnan(z):
            print(f"📍 Depth at ({x},{y}): Invalid")
        else:
            print(f"📍 Depth at ({x},{y}): {z:.2f} meters")

cv.imshow("Depth Map", color_map)
cv.setMouseCallback("Depth Map", show_depth)
cv.waitKey(0)
cv.destroyAllWindows()

# 2. ROI Depth Statistics
x1, y1, x2, y2 = 100, 100, 200, 200  # Change if needed
roi = depth_map[y1:y2, x1:x2]
roi_valid = roi[~np.isnan(roi)]
if roi_valid.size > 0:
    print("\n📊 ROI Depth Analysis:")
    print(f"  Min Depth:   {np.min(roi_valid):.2f} m")
    print(f"  Max Depth:   {np.max(roi_valid):.2f} m")
    print(f"  Mean Depth:  {np.mean(roi_valid):.2f} m")
    print(f"  Std Dev:     {np.std(roi_valid):.2f} m")
else:
    print("❌ No valid depth data in ROI.")

# 3. Histogram
depth_valid = depth_map[~np.isnan(depth_map)]
plt.hist(depth_valid.ravel(), bins=100, color='blue')
plt.title("Depth Histogram")
plt.xlabel("Depth (meters)")
plt.ylabel("Pixel Count")
plt.grid(True)
plt.show()

# 4. Segment near objects
near_mask = (depth_map > 0) & (depth_map < 2.5)
cv.imwrite("near_objects.png", np.uint8(near_mask) * 255)

# 5. 3D distance between two points
pt1 = (150, 150)
pt2 = (300, 300)
z1 = depth_map[pt1[1], pt1[0]]
z2 = depth_map[pt2[1], pt2[0]]
if not np.isnan(z1) and not np.isnan(z2):
    cx = left.shape[1] / 2
    cy = left.shape[0] / 2
    x1 = (pt1[0] - cx) * z1 / focal_length
    y1 = (pt1[1] - cy) * z1 / focal_length
    x2 = (pt2[0] - cx) * z2 / focal_length
    y2 = (pt2[1] - cy) * z2 / focal_length
    p1_3d = np.array([x1, y1, z1])
    p2_3d = np.array([x2, y2, z2])
    distance = np.linalg.norm(p1_3d - p2_3d)
    print(f"\n📏 Real-world distance between {pt1} and {pt2}: {distance:.2f} meters")
else:
    print("\n❌ Invalid depth at one or both selected points.")

# 6. Export depth data
pd.DataFrame(depth_map).to_csv("depth_values.csv", index=False)
print("✅ Depth values saved to depth_values.csv")

# 7. Point cloud export
h, w = depth_map.shape
cx = w / 2
cy = h / 2
i, j = np.meshgrid(np.arange(w), np.arange(h))
Z = depth_map
X = (i - cx) * Z / focal_length
Y = (j - cy) * Z / focal_length

mask = ~np.isnan(Z)
points = np.stack((X[mask], Y[mask], Z[mask]), axis=-1)
colors = color[mask] / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("pointcloud.ply", pcd)
print("✅ 3D point cloud saved to pointcloud.ply")

# Optional visualization
o3d.visualization.draw_geometries([pcd])

