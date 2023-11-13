import os
import numpy as np
import glob
import cv2

input_path = glob.glob('data path')


# 왼쪽 10pixel 비우고 첫번째 픽셀로 땡기기
# for i in range(len(input_path)):
#     output_path = input_path[i].split('\\')[0]+'_left/'+input_path[i].split('\\')[1]+'/'+input_path[i].split('\\')[2]+'/'
#     name = ((input_path[i].split('\\'))[-1].split('.')[0])+'_left.png'
#     src = cv2.imread(input_path[i], cv2.IMREAD_UNCHANGED)
#     roi = src[0:224, 0:1]
#     h, w = src.shape[:2]
#     M = np.float32([[1, 0, 10], [0, 1, 0]])
#     src2 = cv2.warpAffine(src, M, (w, h))
#     roi2 = cv2.hconcat([roi, roi, roi, roi, roi, roi, roi, roi, roi, roi])
#     width, height, channel = roi2.shape
#     src2[0:width, 0:height] = roi2
#     cv2.imwrite(os.path.join(output_path, name), src2)

##########################################################################################

# # 오른쪽 10pixel 비우고 첫번째 픽셀로 땡기기
# for i in range(len(input_path)):
#     output_path = input_path[i].split('\\')[0] + '_right/' + input_path[i].split('\\')[1] + '/' + input_path[i].split('\\')[2] + '/'
#     name = ((input_path[i].split('\\'))[-1].split('.')[0]) + '_right.png'
#     src = cv2.imread(input_path[i], cv2.IMREAD_UNCHANGED)
#     roi = src[0:224, 223:224]
#     h, w = src.shape[:2]
#     M = np.float32([[1, 0, -10], [0, 1, 0]])
#     src2 = cv2.warpAffine(src, M, (w, h))
#     roi2 = cv2.hconcat([roi, roi, roi, roi, roi, roi, roi, roi, roi, roi])
#     width, height, channel = roi2.shape
#     src2[0:width, 224-height:224] = roi2
#     cv2.imwrite(os.path.join(output_path, name), src2)


#########################################################################################

# 위쪽 10pixel 비우고 첫번째 픽셀로 땡기기
for i in range(len(input_path)):
    output_path = input_path[i].split('\\')[0] + '_up/' + input_path[i].split('\\')[1] + '/' + input_path[i].split('\\')[2] + '/'
    name = ((input_path[i].split('\\'))[-1].split('.')[0])+'_up.png'
    src = cv2.imread(input_path[i], cv2.IMREAD_UNCHANGED)
    roi = src[0:1, 0:224]
    h, w = src.shape[:2]
    M = np.float32([[1, 0, 0], [0, 1, 10]])
    src2 = cv2.warpAffine(src, M, (w, h))
    roi2 = cv2.vconcat([roi, roi, roi, roi, roi, roi, roi, roi, roi, roi])
    width, height, channel = roi2.shape
    src2[0:width, 0:height] = roi2
    cv2.imwrite(os.path.join(output_path, name), src2)

#########################################################################################

#아래쪽 10pixel 비우고 첫번째 픽셀로 땡기기
for i in range(len(input_path)):
    output_path = input_path[i].split('\\')[0] + '_down/' + input_path[i].split('\\')[1] + '/' + input_path[i].split('\\')[2] + '/'
    name = ((input_path[i].split('\\'))[-1].split('.')[0])+'_down.png'
    src = cv2.imread(input_path[i], cv2.IMREAD_UNCHANGED)
    roi = src[223:224, 0:224]
    h, w = src.shape[:2]
    M = np.float32([[1, 0, 0], [0, 1, -10]])
    src2 = cv2.warpAffine(src, M, (w, h))
    roi2 = cv2.vconcat([roi, roi, roi, roi, roi, roi, roi, roi, roi, roi])
    width, height, channel = roi2.shape
    src2[224-width:224, 0:hight] = roi2

    cv2.imwrite(os.path.join(output_path, name), src2)
#
# cv2.imwrite(os.path.join(output_path, name), src2)
