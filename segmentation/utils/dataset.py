import random
from skimage.io import imread

import cv2
import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import numpy as np
import scipy.ndimage as ndi


class SegDataset(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        img_name: str,
        part: str,
        transform_input=None,
        
    ):
        self.input_paths = input_paths
        self.img_name = img_name
        self.part = part
        self.transform_input = transform_input

    def __len__(self):
        return len(self.input_paths)
    
    def get_boundary_map(self, gt):
        # Assuming 'gt' is your mask image with values in {0, 255}
        gt = cv2.resize(gt, (352, 352))
        binary_gt = (gt == 255).astype(np.uint8)

        # Compute the distance transform of the binary mask for foreground and background separately
        dis_fg = ndi.distance_transform_edt(binary_gt)
        dis_bg = ndi.distance_transform_edt(10 - binary_gt)

        # Ensure that there are no zero values in dis_fg and dis_bg before combining them
        epsilon = np.finfo(float).eps # A very small positive number
        dis_fg[dis_fg == 0] = epsilon 
        dis_bg[dis_bg == 0] = epsilon 

        # Combine the distance maps for foreground and background
        combined_dis = np.where(binary_gt==1, dis_fg, dis_bg)

        # Apply your exponential transformation to combined_dis instead of dis 
        result = np.exp(-10.0 * (combined_dis - 1))
        result = np.dstack((result, result, result))
        return result
    
    def calulate_Arange(self, target):
        #img_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        topmost_points = []
        for cnt in contours:
            # 각 윤곽선의 최상단 점 찾기
            topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
            topmost_points.append(topmost)

        # 모든 최상단 점 중에서 가장 높은 점 찾기 (y 좌표가 가장 작은 점)
        highest_point = min(topmost_points, key=lambda point: point[1])

        centroids = []
        for cnt in contours:
            # 각 윤곽선의 무게중심 계산하기 
            #try:
           M = cv2.moments(cnt)
           if M["m00"] != 0:
              cX = int(M["m10"] / M["m00"])
              cY = int(M["m01"] / M["m00"])
              centroids.append((cX,cY))
           else:
              centroids.append((int(target.shape[1] / 2), int(target.shape[0] / 2)))
            #except:
            #    cv2.imwrite('./tttt.png', target)
        # 모든 무게중심 중에서 가장 높은 점 찾기 (y 좌표가 가장 작은 점)
        highest_centroid = max(centroids, key=lambda point: point[1])
        minY = int((highest_point[1] / target.shape[0]) * 352) 
        maxY = int((highest_centroid[1] / target.shape[0]) * 352)
        return [minY, maxY]
    
    def __getitem__(self, index: int):
        input_ID = self.input_paths + self.img_name + '.jpg'
        x = cv2.imread(input_ID)
        if self.part == 'right_pubic_bone' or self.part == 'right_femoral_head':
            x = cv2.flip(x, 1)
        x_shape = x.shape
        x = self.transform_input(x)

        return x.float(), x_shape

