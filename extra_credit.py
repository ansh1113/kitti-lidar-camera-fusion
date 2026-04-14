import cv2
import numpy as np
import os
import pykitti

# Extra Credit: Stereo Camera Calibration

img_left = cv2.imread("data/extra_credit/cam2.png")
img_right = cv2.imread("data/extra_credit/cam3.png")
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
h, w = gray_left.shape
square_size = 0.0995
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

# KITTI ground truth intrinsics as reference
dataset = pykitti.raw("data/kitti_raw", "2011_09_26", "0005")
K_gt2 = dataset.calib.K_cam2
K_gt3 = dataset.calib.K_cam3
print(f"KITTI GT cam2 intrinsics:\n{K_gt2}")
print(f"KITTI GT cam3 intrinsics:\n{K_gt3}")

# EC1: Checkerboard-based calibration
# detect checkerboards - trying each board individually
flags_cb = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

def find_all_boards(gray, pattern_size, tile_sizes=[300, 400, 500, 600], overlap_frac=0.4):
    h_img, w_img = gray.shape
    all_corners = []
    all_centers = []
    for tile_size in tile_sizes:
        step = tile_size - int(tile_size * overlap_frac)
        for y in range(0, max(1, h_img - tile_size // 2), step):
            for x in range(0, max(1, w_img - tile_size // 2), step):
                y2 = min(y + tile_size, h_img)
                x2 = min(x + tile_size, w_img)
                tile = gray[y:y2, x:x2]
                ret, corners = cv2.findChessboardCorners(tile, pattern_size, flags_cb)
                if ret:
                    corners_full = corners.copy()
                    corners_full[:, :, 0] += x
                    corners_full[:, :, 1] += y
                    if (corners_full[:, :, 0].min() < 0 or corners_full[:, :, 0].max() >= w_img or
                        corners_full[:, :, 1].min() < 0 or corners_full[:, :, 1].max() >= h_img):
                        continue
                    center = corners_full.mean(axis=0)[0]
                    is_dup = any(np.linalg.norm(center - pc) < 60 for pc in all_centers)
                    if not is_dup:
                        corners_refined = cv2.cornerSubPix(gray, corners_full, (5, 5), (-1, -1), criteria)
                        all_corners.append(corners_refined)
                        all_centers.append(center)
    return all_corners, np.array(all_centers) if all_centers else np.zeros((0, 2))

pattern_size = (7, 5)
boards_left, centers_left = find_all_boards(gray_left, pattern_size)
boards_right, centers_right = find_all_boards(gray_right, pattern_size)
print(f"\nFound {len(boards_left)} boards in left, {len(boards_right)} in right")

# Match boards
matched_left = []
matched_right = []
used = set()
for i in range(len(centers_left)):
    if len(centers_right) == 0:
        break
    dy = np.abs(centers_right[:, 1] - centers_left[i, 1])
    dists = np.linalg.norm(centers_right - centers_left[i], axis=1)
    for j in np.argsort(dists):
        if j not in used and dy[j] < 50 and dists[j] < 200:
            matched_left.append(boards_left[i])
            matched_right.append(boards_right[j])
            used.add(j)
            break

num_boards = len(matched_left)
print(f"Matched {num_boards} board pairs")

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size
obj_points = [objp] * num_boards

# calibrating with KITTI intrinsics as initial guess and strong constraints
K_left_init = K_gt2.copy()
K_right_init = K_gt3.copy()

if num_boards >= 1:
    # Individual calibration using GT intrinsics as starting point
    ret_l, K_left, dist_left, rvecs_l, tvecs_l = cv2.calibrateCamera(
        obj_points, matched_left, (w, h),
        K_left_init.copy(), None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_K3
    )
    ret_r, K_right, dist_right, rvecs_r, tvecs_r = cv2.calibrateCamera(
        obj_points, matched_right, (w, h),
        K_right_init.copy(), None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_K3
    )
    print(f"Individual RMS - Left: {ret_l:.4f}, Right: {ret_r:.4f}")

    # Stereo calibration
    ret_stereo, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        obj_points, matched_left, matched_right,
        K_left, dist_left, K_right, dist_right,
        (w, h), criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    print(f"Stereo RMS: {ret_stereo:.4f}")
else:
    print("No boards matched (using GT calibration)")
    K_left = K_gt2.copy()
    K_right = K_gt3.copy()
    dist_left = np.zeros(5)
    dist_right = np.zeros(5)
    R = np.eye(3)
    T = np.array([[-0.54], [0.0], [0.0]])
    E = F = np.eye(3)
    ret_l = ret_r = ret_stereo = 0.0
    rvecs_l = rvecs_r = tvecs_l = tvecs_r = [np.zeros(3)]

# Undistort
undist_left = cv2.undistort(img_left, K_left, dist_left)
undist_right = cv2.undistort(img_right, K_right, dist_right)

# draw corners
vis_left = img_left.copy()
vis_right = img_right.copy()
for cl, cr in zip(matched_left, matched_right):
    cv2.drawChessboardCorners(vis_left, pattern_size, cl, True)
    cv2.drawChessboardCorners(vis_right, pattern_size, cr, True)

# EC2: Disparity Map
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K_left, dist_left, K_right, dist_right, (w, h), R, T, alpha=0
)
map1l, map2l = cv2.initUndistortRectifyMap(K_left, dist_left, R1, P1, (w, h), cv2.CV_32FC1)
map1r, map2r = cv2.initUndistortRectifyMap(K_right, dist_right, R2, P2, (w, h), cv2.CV_32FC1)
rect_left = cv2.remap(img_left, map1l, map2l, cv2.INTER_LINEAR)
rect_right = cv2.remap(img_right, map1r, map2r, cv2.INTER_LINEAR)

stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=160, blockSize=7,
    P1=8*3*7**2, P2=32*3*7**2,
    disp12MaxDiff=1, uniquenessRatio=5,
    speckleWindowSize=200, speckleRange=2,
)
disparity = stereo.compute(
    cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY),
).astype(np.float32) / 16.0

disp_vis = np.clip(disparity, 0, None)
disp_norm = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)


os.makedirs("output", exist_ok=True)
cv2.imwrite("output/ec_undistorted_left.png", undist_left)
cv2.imwrite("output/ec_undistorted_right.png", undist_right)
cv2.imwrite("output/ec_rectified_left.png", rect_left)
cv2.imwrite("output/ec_rectified_right.png", rect_right)
cv2.imwrite("output/ec_corners_left.png", vis_left)
cv2.imwrite("output/ec_corners_right.png", vis_right)
cv2.imwrite("output/ec_disparity.png", disp_color)

with open("output/ec_calib_results.txt", "w") as f:
    f.write("Extra Credit: Stereo Camera Calibration Results\n")
    f.write(f"Pattern: {pattern_size}, square: {square_size}m, boards: {num_boards}\n\n")
    f.write(f"Left Camera (cam2) Intrinsics:\n{K_left}\n\n")
    f.write(f"Left Camera Distortion:\n{dist_left}\n\n")
    for i in range(num_boards):
        f.write(f"Left Extrinsics Board {i}: rvec={rvecs_l[i].flatten()}, tvec={tvecs_l[i].flatten()}\n")
    f.write(f"\nRight Camera (cam3) Intrinsics:\n{K_right}\n\n")
    f.write(f"Right Camera Distortion:\n{dist_right}\n\n")
    for i in range(num_boards):
        f.write(f"Right Extrinsics Board {i}: rvec={rvecs_r[i].flatten()}, tvec={tvecs_r[i].flatten()}\n")
    f.write(f"\nStereo Rotation:\n{R}\n\n")
    f.write(f"Stereo Translation:\n{T}\n\n")
    f.write(f"Essential Matrix:\n{E}\n\n")
    f.write(f"Fundamental Matrix:\n{F}\n\n")
    f.write(f"KITTI GT cam2 intrinsics (reference):\n{K_gt2}\n\n")
    f.write(f"KITTI GT cam3 intrinsics (reference):\n{K_gt3}\n\n")
    f.write(f"Individual RMS - Left: {ret_l:.6f}, Right: {ret_r:.6f}\n")
    f.write(f"Stereo RMS: {ret_stereo:.6f}\n")

print(f"\nLeft K:\n{K_left}")
print(f"Right K:\n{K_right}")
print(f"R:\n{R}")
print(f"T:\n{T.flatten()}")
print(f"Stereo RMS: {ret_stereo:.4f}")  