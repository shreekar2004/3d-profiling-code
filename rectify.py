import cv2 as cv
import numpy as np

# Load images
imgL = cv.imread("calibresult left.png")
imgR = cv.imread("calibresult right.png")

# Convert to grayscale
grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

# Load known intrinsics and distortion
K = np.array([[3154.3040007934105, 0, 2217.5405339442036],
              [0, 3149.7128617658154, 1589.753703668311],
              [0, 0, 1]])
D = np.array([[-0.003350358297753392, -0.04183989139398258,
               0.0011247929508742514, 0.00243557360495318, 0.17971911555313308]])

# Image size (width, height)
img_size = (grayL.shape[1], grayL.shape[0])

# Manually defined rotation (no rotation) and translation (baseline)
R = np.eye(3)  # identity matrix
T = np.array([[0.06], [0], [0]])  # 6 cm baseline to the right

# Stereo Rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
    K, D, K, D, img_size, R, T,
    flags=cv.CALIB_ZERO_DISPARITY, alpha=1.0
)

# Compute undistortion and rectification maps
map1x, map1y = cv.initUndistortRectifyMap(K, D, R1, P1, img_size, cv.CV_32FC1)
map2x, map2y = cv.initUndistortRectifyMap(K, D, R2, P2, img_size, cv.CV_32FC1)

# Remap images
rectifiedL = cv.remap(imgL, map1x, map1y, cv.INTER_LINEAR)
rectifiedR = cv.remap(imgR, map2x, map2y, cv.INTER_LINEAR)

# Save and show
cv.imwrite("rectified_left.jpg", rectifiedL)
cv.imwrite("rectified_right.jpg", rectifiedR)
np.save("Q_from_rectify.npy", Q)

cv.imshow("Rectified Left", rectifiedL)
cv.imshow("Rectified Right", rectifiedR)
cv.waitKey(0)
cv.destroyAllWindows()

