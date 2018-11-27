import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

import pickle
depth = np.load('0.pkl')
depth = np.fliplr(depth)

plt.imshow(depth, cmap='gray')
# plt.show()

cb = cv2.imread('chessboard_depth/left21.bmp', 0)
plt.imshow(cb, cmap='gray')
# plt.show()

gray = cb.copy()
w, h = 7, 9
ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
corners_3d = np.concatenate([np.squeeze(corners).transpose(), [np.ones(63)]]).transpose()

# corner_cloud =
cv2.drawChessboardCorners(gray, (w,h), corners, ret)
plt.imshow(gray)
# plt.show()

pw = np.loadtxt('coordinate_cal_3').reshape((-1, 3))
print(pw)
# rvec = np.array([[-0.06496874],
#  [ 3.12195727],
#  [-0.11710924]])
data = np.load('came_para2.npz')
mtx = data['mtx']
mtx_i = np.mat(mtx).I
rmat = data['rmat']
tvec = data['tvec']
# corners_cloud_c = np.matmul(mtx_i, corners_3d.transpose()).transpose()
corners_cloud_d = np.matmul(rmat, pw.transpose())
corners_project_d = np.matmul(mtx, corners_cloud_d).transpose()

d_matrix = np.concatenate([[np.ones(63)],
                           np.concatenate([[np.ones(63)], [corners_project_d[:,2]]])]).transpose()
corners_project_d = corners_project_d / d_matrix

import pdb
pdb.set_trace()

C = np.mat(np.zeros((1, 3)))
D = np.mat(np.ones((1, 1)))
arm2cam = np.vstack((np.hstack((np.mat(rmat), tvec)),
                     np.hstack((C, D))
                   ))

pc_cal = arm2cam * np.mat(np.concatenate((pw, np.ones((63, 1))), -1)).T
zc_cal = pc_cal[2, :]

zc = []
for corner in corners.astype(np.int):
    zc.append(depth[corner[0][1], corner[0][0]])
zc = np.array(zc)
print(zc)
print(zc_cal)
print(np.array(zc_cal - zc).reshape((9, 7)))