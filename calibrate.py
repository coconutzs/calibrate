import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--depth_dir', help='folder of depth images', default='chessboard_depth')
args = parser.parse_args()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
depth_img_num = len(os.listdir(args.depth_dir))
img_name = args.depth_dir + '/left%d.bmp'

#chessboard corner number
w = 7
h = 9

# corner positions in world coordinate
objp = np.zeros((63,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)

# list of objp and imgp for each picture
objpoints0 = [] # points3D in world
imgpoints = [] # points2D in img

for i in range(depth_img_num):
    # pdb.set_trace()
    fname = img_name % i
    img = cv2.imread(fname, 0)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = img.copy()

    # find corners
    ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
    if i == depth_img_num - 1:
        objp = objp[::-1]
    # if find
    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objpoints0.append(objp)
        imgpoints.append(corners)
        # show corners on ori picture
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        # plt.imshow(img)
        # plt.show()
        # cv2.imshow('findCorners',img)
        # cv2.waitKey(10)
# cv2.destroyAllWindows()
# calibrate
ret0, mtx0, dist0, rvecs0, tvecs0 = cv2.calibrateCamera(objpoints0, 
                                                        imgpoints, 
                                                        gray.shape[::-1], 
                                                        None, 
                                                        None)

objp_w = np.loadtxt('./coordinate_cal_3').reshape(-1, 3).astype(np.float32)
objpoints = [] # points3D in world

for i in range(depth_img_num):

    if i == depth_img_num - 1:
        objp_w = objp_w[::-1]

    objpoints.append(objp_w)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                   imgpoints, 
                                                   gray.shape[::-1], 
                                                   mtx0, 
                                                   None,
                                                   flags=cv2.CALIB_USE_INTRINSIC_GUESS)
'''
ret: retval re-projection error
mtx: camera matrix
dist: distort coefficients
rvecs: extrinsic rotation parameters
tvecs: extrinsic translation parameters
'''

# distort
# img2 = cv2.imread('chessboard_rgb/right1.bmp')
# h, w = img2.shape[:2]
# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h)) 
# dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)

cv2.imwrite('drawChessBoard.png', img)
# cv2.imwrite('calibresult.png', dist)

# re-project error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error

rmat, _ = cv2.Rodrigues(rvecs[-1])
tvec = tvecs[-1]
np.savez("came_para.npz", mtx = mtx, tvec = tvecs[-1], rmat = rmat)

A = np.mat(rmat).I
B = -A * tvec
C = np.mat(np.zeros((1, 3)))
D = np.mat(np.ones((1, 1)))

final_cam2arm = np.vstack((np.hstack((A, B)),
                           np.hstack((C, D))
                         ))

arm2cam = np.vstack((np.hstack((np.mat(rmat), tvec)),
                     np.hstack((C, D))
                   ))

another_cam2arm = np.mat(arm2cam).I
np.savez("came_para2.npz", mtx = mtx, tvec = tvecs[-1], rmat = rmat, c_to_d_mtx = final_cam2arm)

print("total error: ", total_error/len(objpoints))
print()
print('mtx:\n', mtx)
print()
print('dist:', dist)
print()
print('rotation vec:\n', rvecs[-1])
print('\ntranslation vec:\n', tvecs[-1])
print('\nrotation matrix:\n', rmat)
print('\ncamera to dobot matrix:\n', final_cam2arm)
