import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import math 
import albumentations as A
import torch
from scipy import signal
from torchvision import models
from torch import nn
from albumentations.pytorch import transforms
import torch.nn.functional as F
from scipy.spatial import distance


def erasingNoise(img_mask, img_boundary):
    gray_boundary = cv2.cvtColor(img_boundary, cv2.COLOR_BGR2GRAY)
    h, w = img_boundary.shape[:2]

    img_mask_resized = cv2.resize(img_mask, dsize=(w, h))
    gray_mask = cv2.cvtColor(img_mask_resized, cv2.COLOR_BGR2GRAY)
    gray_canny = cv2.Canny(gray_mask, 50, 200)

    img_erased_temp = cv2.bitwise_and(gray_mask, gray_boundary)
    img_erased_gray = cv2.bitwise_or(gray_canny, img_erased_temp)
    img_erased = cv2.cvtColor(img_erased_gray, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(img_erased_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    min_contour_area = 150 
    small_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < min_contour_area]
    
    img_large_contours = cv2.drawContours(img_erased.copy(), small_contours, -1, (0,0,0), 3)
    
    
    return img_large_contours

def countContours(img):
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return len(contours), contours

def countContours_TREE(img):
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    return len(contours), contours

def findBoundingBox(img, contour):
    img_copy = img.copy()
    x, y, w, h = cv2.boundingRect(contour)
    img_drawed = cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 1)    
    
    return [x, y, w, h], img_drawed

def findMinContourBBox(img, contour):
    cnt = np.array(contour).squeeze()
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)   
    box = np.intp(box)
    img_drawed = cv2.drawContours(img,[box],0,(0,0,255),2)

    return box, img_drawed

def calContourArea(contours):
    areas = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        print(f'Area of contour {i}: {area}')
        areas.append(area)
    return areas    

def findMaxContour(contours):
    contours = list(contours)
    contours.sort(key=len)
    
    return contours[-1]

def calculate_distance_between_line(x1, y1, x2, y2, x0, y0):
    if x2 - x1 == 0: 
        return np.abs(x1 - x0)
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m*x1

        distance = np.abs(m*x0 - y0 + b) / np.sqrt(m**2 + 1)
        return distance
    
def find_nearest_point_to_right_bbox(bbox, max_cnt):
    x, y, w, h = bbox
    
    min_distance_point = []
    min_distance = 99999
    idx = -1
    for i, point in enumerate(np.array(max_cnt).squeeze()):
        current_distance = calculate_distance_between_line(x + w, y, x + w, y + h, point[0], point[1])
        if current_distance < min_distance:
            min_distance = current_distance
            min_distance_point = point
            idx = i
    
    return min_distance_point, idx


def point_distance(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]

    c = math.sqrt((a * a) + (b * b))

    return c

def find_nearest_point_to_point(main_point, max_cnt):    
    min_distance_point = []
    min_distance = 99999
    idx = -1
    for i, point in enumerate(np.array(max_cnt).squeeze()):
        current_distance = point_distance(main_point, point)
        if current_distance < min_distance:
            min_distance = current_distance
            min_distance_point = point
            idx = i
    
    return min_distance_point, idx

def findCentroid(contour):
    M = cv2.moments(contour)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    
    return [cX, cY]

def dist(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2)

def innerProduct(v1, v2):
    distA = dist(v1)
    distB = dist(v2)

    ip = v1[0] * v2[0] + v1[1] * v2[1]
    ip2 = distA * distB
    try:
        cost = ip / ip2
    except:
        return -1
    if cost > 1.0:
        cost = 1.0
    x = math.acos(cost)

    degX = math.degrees(x)

    return degX

def rotate_points(degree, points, cenx, ceny):
    rotated_point = []
    for k, point in enumerate(points):
        rotated_x = (point[0] - cenx) * math.cos(-np.radians(degree)) - (point[1] - ceny) * math.sin(
            -np.radians(degree)) + cenx
        rotated_y = (point[0] - cenx) * math.sin(-np.radians(degree)) + (point[1] - ceny) * math.cos(
            -np.radians(degree)) + ceny
        rotated_point.append([int(rotated_x), int(rotated_y)])
        #rotated_point.append([rotated_x, rotated_y])
    return rotated_point

def rotate_points_float(degree, points, cenx, ceny):
    rotated_point = []
    for k, point in enumerate(points):
        rotated_x = (point[0] - cenx) * math.cos(-np.radians(degree)) - (point[1] - ceny) * math.sin(
            -np.radians(degree)) + cenx
        rotated_y = (point[0] - cenx) * math.sin(-np.radians(degree)) + (point[1] - ceny) * math.cos(
            -np.radians(degree)) + ceny
        #rotated_point.append([int(rotated_x), int(rotated_y)])
        rotated_point.append([rotated_x, rotated_y])
    return rotated_point

def findBoundingBox(img, contour):
    img_copy = img.copy()
    x, y, w, h = cv2.boundingRect(contour)
    img_drawed = cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)    
    
    return [x, y, w, h], img_drawed

def findOnlyBoundingBox(contour):
    x, y, w, h = cv2.boundingRect(contour)
    
    return [x, y, w, h]

class densenet201_femoral(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.densenet201 = models.densenet201(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.densenet201(x)
        x = self.classifier(x)
        return x
    
test_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        transforms.ToTensorV2()])


def hasFemoralHead(img, img_retouched_bd):
    len_, contours = countContours(img_retouched_bd)
    
    ckpt_path = '/home/moon/child_proj/all_system/algorithm/weights/best_densenet201.pth'
    model = densenet201_femoral(num_classes=2)
    model.load_state_dict(torch.load(ckpt_path))
    model = model.cpu()
    model.eval()
    input = test_transforms(image=img)['image'].float()
    input = input.unsqueeze(0)
    outputs = model(input) 
    
    prob = torch.sigmoid(outputs).detach().numpy()[0]
    
    topk = (1,1)
    maxk = max(topk)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    pred_np = pred.cpu().numpy()[0][0]
    
    retouched_pred_label = -1
    print(prob)
    if pred_np == 0:
        if prob[pred_np] < 0.9:
            if len_ >= 2:
                contours_cp = list(contours)
                contours_cp.sort(key=len)
                isHead = False
                areas = calContourArea(contours_cp[:-1])
                for area in areas:
                    if area > 300:
                        isHead = True
                if isHead:
                    retouched_pred_label = 1
                else:
                    retouched_pred_label = 0
            else:
                retouched_pred_label = 0
        else:
            retouched_pred_label = 0
    else:
        retouched_pred_label = pred_np
    return retouched_pred_label

def pelvis_neck(contour):
    
    contours = np.array(contour).squeeze()

    hull_index = cv2.convexHull(contours, returnPoints=False)

    defects = cv2.convexityDefects(contours, hull_index)

    sorted_defects = defects.squeeze()[:, -1]
    defect = defects[np.argsort(-sorted_defects)]
    defect = defect.squeeze()[:2]

    neck_points = []
    for i in range(len(defect)):
        sp, ep, fp, dist = defect[i]
        far = tuple(contours[fp])
        neck_points.append(far)
    
    return neck_points

def pelvis_neck_in_convergence(contour):
    
    contours = np.array(contour).squeeze()

    hull_index = cv2.convexHull(contours, returnPoints=False)

    defects = cv2.convexityDefects(contours, hull_index)

    sorted_defects = defects.squeeze()[:, -1]
    defect = defects[np.argsort(-sorted_defects)]
    defect = defect.squeeze()[:1]

    neck_points = []
    for i in range(len(defect)):
        sp, ep, fp, dist = defect[i]
        far = tuple(contours[fp])
        neck_points.append(far)
    
    return neck_points

def curvature(contour):
    curvature = []

    for i in range(len(contour)):
        prev_point = contour[(i-1) % len(contour)]
        curr_point = contour[i % len(contour)]
        next_point = contour[(i+1) % len(contour)]

        prev_vector = np.subtract(prev_point, curr_point)
        next_vector = np.subtract(next_point, curr_point)

        angle = np.arctan2(prev_vector[1], prev_vector[0]) - np.arctan2(next_vector[1], next_vector[0])

        if angle < 0:
            angle += 2 * np.pi

        curvature.append(angle)

    return curvature

def findFemoralPoint(img, contour, head_size):
    cnt = np.array(contour).squeeze()
    
    black_img =  np.zeros(img.shape)
    only_femoral_img = cv2.polylines(black_img, [contour], True, (255, 255, 255), 2)
    only_femoral_img_cp = only_femoral_img.copy()
    
    head_centroid = findCentroid(contour)
    only_femoral_bbox_points, bd_femoral = findBoundingBox(only_femoral_img, contour)
    print(only_femoral_bbox_points)
    
    bbox_height_thres = only_femoral_bbox_points[-1] * 0.15
    
    subset_cnt = np.array([point for point in cnt if point[1] < head_centroid[1] - bbox_height_thres])
    
    for point in subset_cnt:
        cv2.circle(only_femoral_img, point, 3, (0, 255, 0), -1)
    
    sub_set_bbox, bd_femoral_upper = findBoundingBox(only_femoral_img, np.array(subset_cnt))
        
    plt.imshow(bd_femoral)
    plt.title('bbox femoral')
    plt.show()
        
    plt.imshow(bd_femoral_upper)
    plt.title('bd_femoral_upper')
    plt.show()
        
    min_x_point = subset_cnt[np.argmin(subset_cnt[:, 0])]
    max_x_point = subset_cnt[np.argmax(subset_cnt[:, 0])]
        
    min_x_idx_in_cnt = -1
    max_x_idx_in_cnt = -1
        
    for i, point in enumerate(cnt):
        if point[0] == min_x_point[0] and point[1] == min_x_point[1]:
            min_x_idx_in_cnt = i
        if point[0] == max_x_point[0] and point[1] == max_x_point[1]:
            max_x_idx_in_cnt = i
    
    print(min_x_idx_in_cnt)
    print(max_x_idx_in_cnt)
    #print(cnt.shape)
    #cv2.circle(only_femoral_img_cp, min_x_point, 3, (0, 0, 255), -1)
    #cv2.circle(only_femoral_img_cp, max_x_point, 3, (0, 0, 255), -1)
        
    if max_x_idx_in_cnt > min_x_idx_in_cnt:
        real_subset_cnt = np.concatenate((cnt[:min_x_idx_in_cnt],cnt[max_x_idx_in_cnt:]))
    else:    
        real_subset_cnt = np.concatenate((cnt[:max_x_idx_in_cnt],cnt[min_x_idx_in_cnt:]))
    
    if head_size > 11000:
        pelvis_neck_point = pelvis_neck(subset_cnt)[0]
        pelvis_neck_idx = -1
        for i, point in enumerate(cnt):
            if point[0] == pelvis_neck_point[0] and point[1] == pelvis_neck_point[1]:
                pelvis_neck_idx = i
        
        cv2.circle(only_femoral_img_cp, pelvis_neck_point, 3, (0, 0, 255), -1)
    
        plt.imshow(only_femoral_img_cp)
        plt.title('pelvis neck')
        plt.show()
        
        
        if max_x_idx_in_cnt > pelvis_neck_idx:
            real_subset_cnt = np.concatenate((cnt[:pelvis_neck_idx],cnt[max_x_idx_in_cnt:]))
        else:    
            real_subset_cnt = np.concatenate((cnt[:max_x_idx_in_cnt],cnt[pelvis_neck_idx:]))
    
        
    for point in real_subset_cnt:
        cv2.circle(only_femoral_img_cp, point, 3, (0, 255, 0), -1)
    
    real_real_subset_cnt = np.array([point for point in real_subset_cnt if point[0] < 0.8 * max_x_point[0]])
    new_max_x_point = real_real_subset_cnt[np.argmax(real_real_subset_cnt[:, 0])]
    for point in real_real_subset_cnt:
        cv2.circle(only_femoral_img_cp, point, 3, (0, 0, 255), -1)
    
    h, w, c = img.shape
    cenx = int(w / 2)
    ceny = int(h / 2)
        
    std_vec = [w, 0]
    #current_vec = [max_x_point[0] - min_x_point[0], max_x_point[1] - min_x_point[1]]
    current_vec = [new_max_x_point[0] - min_x_point[0], new_max_x_point[1] - min_x_point[1]]
    angle = innerProduct(std_vec, current_vec)
    print('start_angle: ', angle)
    if (new_max_x_point[1] - min_x_point[1]) < 0: 
        angle = 180 - angle
    else:
        angle = (180 - angle) + angle
    
    rotate_subset = np.array(rotate_points_float(angle, real_real_subset_cnt, cenx, ceny))
    max_y_rotate_subset = rotate_subset[np.argmax(rotate_subset[:, 1])]
    max_y_rotate_subset_in_subset = real_real_subset_cnt[np.argmax(rotate_subset[:, 1])]
        
    for point in rotate_subset:
        cv2.circle(only_femoral_img_cp, [int(point[0]), int(point[1])], 3, (0, 255, 0), -1)
        
        
    peaks, properties = signal.find_peaks(rotate_subset[:, 1], rel_height=0.5)
    promin = signal.peak_prominences(rotate_subset[:, 1], peaks)[0]
            
    cv2.circle(only_femoral_img_cp, [int(max_y_rotate_subset[0]), int(max_y_rotate_subset[1])], 3, (0, 0, 255), -1)
    
    
    plt.imshow(only_femoral_img_cp)
    plt.title('rotation')
    plt.show()
            
    max_y_rotate_subset_in_subset_idx = -1
    for i, point in enumerate(cnt):
        if point[0] == max_y_rotate_subset_in_subset[0] and point[1] == max_y_rotate_subset_in_subset[1]:
            max_y_rotate_subset_in_subset_idx = i
            break
        
    if max_x_idx_in_cnt > max_y_rotate_subset_in_subset_idx:
        real_end_subset = np.concatenate((cnt[:max_y_rotate_subset_in_subset_idx],cnt[max_x_idx_in_cnt:]))
    else:    
        real_end_subset = np.concatenate((cnt[:max_x_idx_in_cnt],cnt[max_y_rotate_subset_in_subset_idx:]))
        
    #std_vec = [w, 0]
    current_end_vec = [max_x_point[0] - max_y_rotate_subset_in_subset[0], max_x_point[1] - max_y_rotate_subset_in_subset[1]]
    angle = innerProduct(std_vec, current_end_vec)
    print('end_angle: ', angle)
    if head_size < 11000:
        angle = 260 - angle
    else:
        angle = 285 - angle    
    rotate_end_subset = np.array(rotate_points_float(angle, real_end_subset, cenx, ceny))
    for point in rotate_end_subset:
        cv2.circle(only_femoral_img_cp, [int(point[0]), int(point[1])], 3, (255, 0, 100), -1)        
        
    end_in_subset = real_end_subset[np.argmax(rotate_end_subset[:, 1])]
    cv2.circle(only_femoral_img_cp, end_in_subset, 3, (0, 0, 255), -1)        
    plt.imshow(only_femoral_img_cp)
    plt.title('rotation2')
    plt.show()
    
    if head_size > 11000:
        std_point = max_y_rotate_subset_in_subset
    else:
        if promin != []:
            std_point = real_subset_cnt[np.argmax(promin)]
        else:
            std_point = max_y_rotate_subset_in_subset
    std_y_point = end_in_subset
        
    return std_point, std_y_point
    

def findFemoralPoint_convergence(img, contour, head_size):
    cnt = np.array(contour).squeeze()
    
    black_img =  np.zeros(img.shape)
    black_img_cp = black_img.copy()
    only_femoral_img = cv2.polylines(black_img, [contour], True, (255, 255, 255), 2)
    only_femoral_img_new = only_femoral_img.copy()
    
    only_femoral_img_cp = only_femoral_img.copy()
    
    head_centroid = findCentroid(contour)
    
    only_femoral_bbox_points, bd_femoral = findBoundingBox(only_femoral_img, contour)
    print(only_femoral_bbox_points)
    
    bbox_height_thres = only_femoral_bbox_points[-1] * 0.15
    cv2.circle(bd_femoral, head_centroid, 3, (0, 255, 0), -1)
    subset_cnt = np.array([point for point in cnt if point[1] < head_centroid[1] - bbox_height_thres])
    
    for point in subset_cnt:
        cv2.circle(bd_femoral, point, 3, (0, 255, 0), -1)
    
    plt.imshow(bd_femoral)
    plt.title('only femoral')
    plt.show()
    
    pelvis_neck_point = pelvis_neck_in_convergence(subset_cnt)[0]
    
    cv2.circle(only_femoral_img, pelvis_neck_point, 3, (0, 0, 255), -1)
    pelvis_neck_point_idx = -1
    for i, point in enumerate(cnt):
        if point[0] == pelvis_neck_point[0] and point[1] == pelvis_neck_point[1]:
            pelvis_neck_point_idx = i
    
    max_x_point = subset_cnt[np.argmax(subset_cnt[:, 0])]
    max_x_idx_in_cnt = -1
    
    for i, point in enumerate(cnt):
        if point[0] == max_x_point[0] and point[1] == max_x_point[1]:
            max_x_idx_in_cnt = i
    
    if max_x_idx_in_cnt > pelvis_neck_point_idx:
        real_subset_cnt = cnt[pelvis_neck_point_idx: max_x_idx_in_cnt]
    
    else:    
        real_subset_cnt = np.concatenate((cnt[:max_x_idx_in_cnt],cnt[pelvis_neck_point_idx:]))
    
    
    real_real_subset_cnt = np.array([point for point in real_subset_cnt if point[0] < 0.8 * max_x_point[0]])
    new_max_x_point = real_real_subset_cnt[np.argmax(real_real_subset_cnt[:, 0])]
    
    for point in real_subset_cnt:
        cv2.circle(only_femoral_img, point, 3, (0, 0, 255), -1)
    
    for point in real_real_subset_cnt:
        cv2.circle(only_femoral_img, point, 3, (0, 255, 0), -1)
        
    
    real_real_subset_cnt_cp = list(real_real_subset_cnt)
    real_real_subset_cnt_cp.sort(key=lambda x: x[0])
    real_real_subset_cnt_cp = np.array(real_real_subset_cnt_cp)
    
    
    epsilon = cv2.arcLength(real_real_subset_cnt, False) * 0.005
    approx_poly = cv2.approxPolyDP(real_real_subset_cnt, epsilon, True)
    
    approx_poly_cp = np.array(approx_poly).squeeze()
    print(approx_poly_cp.shape)
    approx_poly_cp = list(approx_poly_cp)
    approx_poly_cp.sort(key=lambda x: x[0])
    approx_poly_cp = np.array(approx_poly_cp)
    
    for i, approx in enumerate(approx_poly_cp):
        cv2.circle(only_femoral_img, tuple(approx), 5, (255, 0, 0), -1)
    
    plt.imshow(only_femoral_img)
    plt.title(f'approx_poly_cp')
    plt.show()
    
    h, w, c = img.shape
    cenx = int(w / 2)
    ceny = int(h / 2)
        
    std_vec = [w, 0]
    current_vec = [new_max_x_point[0] - pelvis_neck_point[0], new_max_x_point[1] - pelvis_neck_point[1]]
    angle = innerProduct(std_vec, current_vec)
    final_std_point = []
    print('temp_angle: ', angle)

    if (new_max_x_point[1] - pelvis_neck_point[1]) < 0: 
        #angle = (180 - angle) - angle
        angle = 180 - angle
    else:
        angle = (180 - angle) + angle
         
    
    rotate_subset = np.array(rotate_points_float(angle, real_real_subset_cnt, cenx, ceny))
    max_y_rotate_subset_in_subset = real_real_subset_cnt[np.argmax(rotate_subset[:, 1])]
        
    for point in rotate_subset:
        cv2.circle(only_femoral_img, [int(point[0]), int(point[1])], 3, (0, 255, 0), -1)
    max_y_rotate_subset = rotate_subset[np.argmax(rotate_subset[:, 1])]
    cv2.circle(only_femoral_img, [int(max_y_rotate_subset[0]), int(max_y_rotate_subset[1])], 3, (0, 255, 255), -1)
    cv2.circle(only_femoral_img, [int(max_y_rotate_subset_in_subset[0]), int(max_y_rotate_subset_in_subset[1])], 3, (0, 255, 0), -1)
    
    plt.imshow(only_femoral_img)
    plt.title('rotated')
    plt.show()
    
        
    cv2.line(only_femoral_img_new, max_y_rotate_subset_in_subset, [max_y_rotate_subset_in_subset[0], h], (255, 255, 255), 3)
    only_femoral_img_new = only_femoral_img_new.astype(np.int8)
    only_femoral_img_new = cv2.convertScaleAbs(only_femoral_img_new)
    only_femoral_img_new_gray = cv2.cvtColor(only_femoral_img_new, cv2.COLOR_BGR2GRAY)
    divided_cnt, _ = cv2.findContours(only_femoral_img_new_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    divided_cnt = list(divided_cnt)
    divided_cnt.sort(key=len)
    divided_cnt = divided_cnt[:-1]
    
    right_x_cnt = []
    breaker = False
    min_distance_in_divide_cnt_1 = 99999
    min_distance_in_divide_cnt_2 = 99999
    for i, cnt_temp in enumerate(divided_cnt):
        if i == len(divided_cnt) - 1:
            for point in np.array(cnt_temp).squeeze():
                dist_1 = point_distance(point, max_x_point)
                if dist_1 < min_distance_in_divide_cnt_1:
                    min_distance_in_divide_cnt_1 = dist_1
        if i == len(divided_cnt) - 2:
            for point in np.array(cnt_temp).squeeze():
                dist_2 = point_distance(point, max_x_point)
                if dist_2 < min_distance_in_divide_cnt_2:
                    min_distance_in_divide_cnt_2 = dist_2
                    
    if min_distance_in_divide_cnt_1 < min_distance_in_divide_cnt_2:
        right_x_cnt = np.array(divided_cnt[-1]).squeeze()
    else:
        right_x_cnt = np.array(divided_cnt[-2]).squeeze()
    
    epsilon2 = cv2.arcLength(right_x_cnt, False) * 0.005
    approx_poly2 = cv2.approxPolyDP(right_x_cnt, epsilon2, True)
    approx_poly2_np = np.array(approx_poly2).squeeze()
    for point in approx_poly2_np:
        cv2.circle(only_femoral_img_cp, point, 5, (0, 0, 255), -1)
    plt.imshow(only_femoral_img_cp)
    plt.title('approx_poly2_np')
    plt.show() 
    
    approx_poly_top = approx_poly2_np[np.argmin(approx_poly2_np[:, 1])]
    approx_poly_bottom = approx_poly2_np[np.argmax(approx_poly2_np[:, 1])]
    
    std_for_right_x_cnt_point = [max_y_rotate_subset_in_subset[0], h]
    
    current_vec = [approx_poly_bottom[0] - approx_poly_top[0], approx_poly_bottom[1] - approx_poly_top[1]]
    angle = innerProduct(std_vec, current_vec)
    print('2222 angle: ', angle)
    #angle = 160 + angle
    angle = 180 - 2 * abs(90 - angle) + angle
    
    rotate_end_subset = np.array(rotate_points_float(angle, right_x_cnt, cenx, ceny))
    

    for point in rotate_end_subset:
        cv2.circle(only_femoral_img_cp, [int(point[0]), int(point[1])], 3, (100, 255, 100), -1)
    
    end_point_in_rotate_end_subset = rotate_end_subset[np.argmax(rotate_end_subset[:, 1])]
    end_point_in_right_cnt_subset = right_x_cnt[np.argmax(rotate_end_subset[:, 1])]
    
    cv2.circle(only_femoral_img_cp, [int(end_point_in_rotate_end_subset[0]), int(end_point_in_rotate_end_subset[1])], 3, (0, 0, 255), -1)
    cv2.circle(only_femoral_img_cp, [int(end_point_in_right_cnt_subset[0]), int(end_point_in_right_cnt_subset[1])], 3, (100, 255, 0), -1)
    
    plt.imshow(only_femoral_img_cp)
    plt.title('rotated2')
    plt.show() 
    
    
    
    
    std_point = max_y_rotate_subset_in_subset
    std_y_point = end_point_in_right_cnt_subset
    return std_point, std_y_point

def calLongestDiameter(img ,contours):
    contours_cp = list(contours)
    contours_cp.sort(key=len)
    
    femoral_cnt = np.array(contours_cp[1]).squeeze()
    head_cnt = np.array(contours_cp[0]).squeeze()
    head_size = cv2.contourArea(head_cnt)
    print('head_size:', cv2.contourArea(head_cnt))
    std_point, std_y_point = findFemoralPoint(img, femoral_cnt, head_size)

    head_centroid = findCentroid(contours_cp[0])
    
    sample_head = np.array([point for point in head_cnt if point[0] > head_centroid[0]])
    min_top_point = sample_head[np.argmin(sample_head[:, 1])]
    min_bottom_point = sample_head[np.argmax(sample_head[:, 1])]
    min_top_distance_idx = -1
    min_bottom_distance_idx = -1

    for i, point in enumerate(head_cnt):
        if point[0] == min_top_point[0] and point[1] == min_top_point[1]:
            min_top_distance_idx = i
        
        if point[0] == min_bottom_point[0] and point[1] == min_bottom_point[1]:
            min_bottom_distance_idx = i
    
    cv2.circle(img, head_cnt[min_top_distance_idx], 3, (0, 255, 0), -1)
    cv2.circle(img, head_cnt[min_bottom_distance_idx], 3, (0, 255, 0), -1)
    
    plt.imshow(img)
    plt.title('min_top, head min_bottom')
    plt.show()
    
    print('min_top_idx:', min_top_distance_idx)
    print('min_bottom_idx:', min_bottom_distance_idx)

    
    if min_top_distance_idx < min_bottom_distance_idx:
        right_max_x_cnt = head_cnt[min_bottom_distance_idx:]
    else:
        right_max_x_cnt = head_cnt[min_bottom_distance_idx: min_top_distance_idx]

    
    for point in right_max_x_cnt:
        cv2.circle(img, point, 3, (0, 255, 0), -1)
    print(len(right_max_x_cnt))
    
    h, w, c = img.shape
    cenx = int(w / 2)
    ceny = int(h / 2)
    
    std_vec = [w, 0]
    
    current_vec = [min_bottom_point[0] - min_top_point[0], min_bottom_point[1] - min_top_point[1]]
    print(min_top_point)
    print(min_bottom_point)
    angle = innerProduct(std_vec, current_vec)
    print('angle in head', angle)

    
    if 90 <= angle <= 120:
        angle = angle - abs(angle - 90)
    else:
        angle = angle + abs(90 - angle)
    
    epsilon = cv2.arcLength(right_max_x_cnt, False) * 0.01
    approx_poly = cv2.approxPolyDP(right_max_x_cnt, epsilon, True)
    approx_poly_np = np.array(approx_poly).squeeze()
    for point in approx_poly_np:
        cv2.circle(img, point, 3, (0, 0, 255), -1)
    
    
    rotated_right_max_x_cnt = np.array(rotate_points_float(angle, right_max_x_cnt, cenx, ceny))
    
    right_max_point = right_max_x_cnt[np.argmin(rotated_right_max_x_cnt[:, 1])]
    
    for point in rotated_right_max_x_cnt:
        cv2.circle(img, [int(point[0]), int(point[1])], 3, (50, 50, 100), -1)
    cv2.circle(img, right_max_point, 3, (100, 0, 100), -1)

    m = (std_point[1] - std_y_point[1]) / (std_point[0] - std_y_point[0])
    b = std_y_point[1] - m * std_y_point[0]
    shift = right_max_point[1] - (m * right_max_point[0] + b)
    b = b + shift
    
    start_point = (0, int(b))
    end_point = (img.shape[1], int(m * (img.shape[1]) + b))

    cv2.line(img, start_point, end_point, (0,255,255), 1)
    
    distances = np.abs(head_cnt[:, 1] - m * head_cnt[:, 0] - b) / np.sqrt(1 + m**2)
 
    distances = np.abs(head_cnt[:, 1] - m * head_cnt[:, 0] - b) / np.sqrt(1 + m**2)
    min_indices = [[dist, i] for i, dist in enumerate(distances)]
    min_indices.sort(key=lambda x:x[0])

    min_distance_1 = min_indices[0][1]
    min_distance_2 = -1

    for min_idx in min_indices:
        if abs(min_distance_1 - min_idx[1]) < 5:
            continue
        else:
            min_distance_2 = min_idx[1]
            break
    
    left_point_in_head_dist = -9999
    left_point_in_head = []
    new_min_indice_left_candidates = []
    for data in min_indices:
        if data[0] < 1:
            print('data: ', data)
            new_min_indice_left_candidates.append(data)
            cv2.circle(img, head_cnt[data[1]], 3, (255, 0, 0), -1)

    plt.imshow(img)
    plt.title('min_idx')
    plt.show()
    
    for min_idx2 in new_min_indice_left_candidates:
        current_distance_in_head = point_distance(head_cnt[min_idx2[1]], right_max_point)
        if current_distance_in_head > left_point_in_head_dist:
            left_point_in_head_dist = current_distance_in_head
            left_point_in_head = head_cnt[min_idx2[1]]
    
    
    cv2.circle(img, left_point_in_head, 3, (255, 0, 0), -1)
    cv2.circle(img, right_max_point, 3, (255, 0, 0), -1)

 
    
    max_dist = point_distance(left_point_in_head, right_max_point)
            
    return max_dist, left_point_in_head, right_max_point, std_point, std_y_point, head_cnt

def calLongestDiameterInConnectedCnt(img, contours):
    contours_cp = list(contours)
    contours_cp.sort(key=len)
    contours_area = []
    
    for cnt in contours_cp:
        area = cv2.contourArea(cnt)
        contours_area.append([area, cnt])
        
    contours_area.sort(key=lambda x:x[0])
    femoral_cnt = np.array(contours_area[-2][1]).squeeze()
    
    new_contours_area = []
    contours_cp = contours_cp[:-1]
    areas = []
    for contour in contours_cp:
        area = cv2.contourArea(contour)
        if area < 5000:
            continue
        areas.append(area)
    average = np.mean(np.array(areas))
    
    for contour1 in contours_cp:
        area = cv2.contourArea(contour1)
        if area >= average:
            continue
        else:
            new_contours_area.append(contour1)
            
    new_contours_area.sort(key=len)

    head_cnt = np.array(new_contours_area[-1]).squeeze()
    head_size = cv2.contourArea(head_cnt)
    print('head_size:', head_size)
    std_point, std_y_point = findFemoralPoint_convergence(img, femoral_cnt, head_size)

    head_centroid = findCentroid(new_contours_area[-1])
    
    top_head = [point for point in head_cnt if point[1] < head_centroid[1]]
    
    min_top_distance = 99999
    min_top_point = []
    min_top_distance_idx = -1
    
    for point in top_head:
        current_distance_top = point_distance(head_centroid, point)
        if current_distance_top < min_top_distance:
            min_top_distance = current_distance_top
            min_top_point = point
            
    min_bottom_distance = 99999
    min_bottom_point = []
    min_bottom_distance_idx = -1
    
    bottom_head = [point for point in head_cnt if point[1] > head_centroid[1]]
    
    for point in bottom_head:
        current_distance_bottom = point_distance(head_centroid, point)
        if current_distance_bottom < min_bottom_distance:
            min_bottom_distance = current_distance_bottom
            min_bottom_point = point
            
        
    for i, point in enumerate(head_cnt):
        if point[0] == min_top_point[0] and point[1] == min_top_point[1]:
            min_top_distance_idx = i
        
        if point[0] == min_bottom_point[0] and point[1] == min_bottom_point[1]:
            min_bottom_distance_idx = i
    
    cv2.circle(img, head_cnt[min_top_distance_idx], 3, (0, 255, 0), -1)
    cv2.circle(img, head_cnt[min_bottom_distance_idx], 3, (0, 0, 255), -1)
    print('head_cnt[min_top_distance_idx]: ', head_cnt[min_top_distance_idx])
    print('head_cnt[min_bottom_distance_idx]: ', head_cnt[min_bottom_distance_idx])
    
    plt.imshow(img)
    plt.title('min_top, head min_bottom')
    plt.show()
        

    print('min_top_idx:', min_top_distance_idx)
    print('min_bottom_idx:', min_bottom_distance_idx)
    if min_top_distance_idx < min_bottom_distance_idx:
        #right_max_x_cnt = head_cnt[min_bottom_distance_idx:]
        right_max_x_cnt = head_cnt[min_top_distance_idx:min_bottom_distance_idx]
    else:
        right_max_x_cnt = head_cnt[min_bottom_distance_idx: min_top_distance_idx]
    for point in right_max_x_cnt:
        cv2.circle(img, point, 3, (0, 255, 0), -1)
    
    h, w, c = img.shape
    cenx = int(w / 2)
    ceny = int(h / 2)
    
    std_vec = [w, 0]
    current_vec = [min_bottom_point[0] - min_top_point[0], min_bottom_point[1] - min_top_point[1]]
    
    angle = innerProduct(std_vec, current_vec)
    print('angle in head', angle)
    angle = angle - (90 - angle)
    
    if 90 <= angle <= 120:
        angle = angle - abs(angle - 90)
    elif angle > 120:
        angle = angle - abs(angle - 90) + (angle - 120) / 1.2
    else:
        angle = angle + abs(90 - angle) 
    
    rotated_right_max_x_cnt = np.array(rotate_points(angle, right_max_x_cnt, cenx, ceny))
    
    right_max_point = right_max_x_cnt[np.argmin(rotated_right_max_x_cnt[:, 1])]
    
    for point in rotated_right_max_x_cnt:
        cv2.circle(img, point, 3, (50, 50, 100), -1)
    cv2.circle(img, right_max_point, 3, (100, 0, 100), -1)
    
    plt.imshow(img)
    plt.title('111111111111111')
    plt.show()    
    
    m = (std_point[1] - std_y_point[1]) / (std_point[0] - std_y_point[0])
    b = std_y_point[1] - m * std_y_point[0]
    #shift = head_centroid[1] - (m * head_centroid[0] + b)
    shift = right_max_point[1] - (m * right_max_point[0] + b)
    b = b + shift
    
    start_point = (0, int(b))
    end_point = (img.shape[1], int(m * (img.shape[1]) + b))

    cv2.line(img, start_point, end_point, (0,255,255), 1)
    
    distances = np.abs(head_cnt[:, 1] - m * head_cnt[:, 0] - b) / np.sqrt(1 + m**2)
    min_indices = [[dist, i] for i, dist in enumerate(distances)]
    min_indices.sort(key=lambda x:x[0])

    left_point_in_head_dist = -9999
    left_point_in_head = []
    
    for min_idx2 in min_indices[:20]:
        current_distance_in_head = point_distance(head_cnt[min_idx2[1]], right_max_point)
        if current_distance_in_head > left_point_in_head_dist:
            left_point_in_head_dist = current_distance_in_head
            left_point_in_head = head_cnt[min_idx2[1]]
    
    cv2.circle(img, left_point_in_head, 3, (255, 0, 0), -1)
    cv2.circle(img, right_max_point, 3, (255, 0, 0), -1)
    
    plt.imshow(img)
    plt.title('fasdjfasdhfklhasdjkf')
    plt.show()
    
    max_dist = point_distance(left_point_in_head, right_max_point)
            
    return max_dist, left_point_in_head, right_max_point, std_point, std_y_point, head_cnt


def calLongestDiameter_px_before(img_retouched_bd, contours, tag):
    contours_cp = list(contours)
    contours_cp.sort(key=len)
    
    
    if len(contours_cp) > 2:
        head_cnt = np.array(contours_cp[-1]).squeeze()
    else:
        head_cnt = np.array(contours_cp[0]).squeeze()

    print('debug point')
    print('head cnt:', len(head_cnt))
    for cnts in contours_cp:
        print('len cnts', len(cnts))
    
    
    dists = distance.cdist(head_cnt, head_cnt, 'euclidean')
    max_dist = np.max(dists)
    idx = np.unravel_index(np.argmax(dists), dists.shape)
    
    cv2.circle(img_retouched_bd, head_cnt[np.argmax(head_cnt[:, 0])], 3, (0, 0, 255), -1)
    cv2.circle(img_retouched_bd, head_cnt[np.argmin(head_cnt[:, 0])], 3, (0, 0, 255), -1)
    left_head_points = []
    right_head_points = []
    
    for point in head_cnt:
        if point[0] == head_cnt[np.argmin(head_cnt[:, 0])][0]:
            cv2.circle(img_retouched_bd, point, 5, (0, 0, 255), -1)
            left_head_points.append(point)
    
    for point in head_cnt:
        if point[0] == head_cnt[np.argmax(head_cnt[:, 0])][0]:
            cv2.circle(img_retouched_bd, point, 5, (0, 0, 255), -1)
            right_head_points.append(point)
    

    
    head_start_point = [head_cnt[np.argmin(head_cnt[:, 0])][0], int(np.median(np.array(left_head_points)[:, 1]))]
    head_end_point = [head_cnt[np.argmax(head_cnt[:, 0])][0], right_head_points[np.argmax(np.array(right_head_points)[:, 1])][1]]
    
    head_align_vec = [head_start_point[0] - head_end_point[0], head_start_point[1] - head_end_point[1]]
    cv2.line(img_retouched_bd, head_start_point, head_end_point, (0, 0, 255), 5)
    std_vec = [-1, 0]
    
    align_angle = innerProduct(head_align_vec, std_vec)
    h, w, c = img_retouched_bd.shape
    cenx = int(w / 2)
    ceny = int(h / 2)
    
    if align_angle <= 10:
        align_angle = 1.5 * align_angle
    else:
        align_angle = align_angle / 2
        
    
    rotated_head_cnt = np.array(rotate_points(align_angle, head_cnt, cenx, ceny))
    for point in rotated_head_cnt:
        cv2.circle(img_retouched_bd, point, 3, (0, 0, 255), -1)
    
    rotated_left_head_points = []
    rotated_right_head_points = []

    for point in rotated_head_cnt:
        if point[0] == rotated_head_cnt[np.argmin(rotated_head_cnt[:, 0])][0]:
            cv2.circle(img_retouched_bd, point, 5, (255, 0, 255), -1)
            rotated_left_head_points.append(point)
    
    for point in rotated_head_cnt:
        if point[0] == rotated_head_cnt[np.argmax(rotated_head_cnt[:, 0])][0]:
            cv2.circle(img_retouched_bd, point, 5, (255, 0, 255), -1)
            rotated_right_head_points.append(point)
        
    rotated_head_start_point = [rotated_head_cnt[np.argmin(rotated_head_cnt[:, 0])][0], rotated_left_head_points[np.argmin(np.array(rotated_left_head_points)[:, 1])][1]]
    rotated_head_end_point = [rotated_head_cnt[np.argmax(rotated_head_cnt[:, 0])][0], rotated_right_head_points[np.argmax(np.array(rotated_right_head_points)[:, 1])][1]]
    cv2.circle(img_retouched_bd, rotated_head_start_point, 3, (0, 255, 255), -1)
    cv2.circle(img_retouched_bd, rotated_head_end_point, 3, (0, 255, 255), -1)
    
    rotated_head_start_idx = -1
    rotated_head_end_idx = -1
    
    for i, point in enumerate(rotated_head_cnt):
        if point[0] == rotated_head_start_point[0] and point[1] == rotated_head_start_point[1]:
            rotated_head_start_idx = i
        elif point[0] == rotated_head_end_point[0] and point[1] == rotated_head_end_point[1]:
            rotated_head_end_idx = i
    
    
    final_head_start_point = head_cnt[rotated_head_start_idx]
    final_head_end_point = head_cnt[rotated_head_end_idx]
    
    cv2.circle(img_retouched_bd, final_head_start_point, 7, (0, 255, 255), -1)
    cv2.circle(img_retouched_bd, final_head_end_point, 7, (0, 255, 255), -1)
    cv2.imwrite(f'./ttdafd_{tag}.png', img_retouched_bd)    
    
    head_diameter = point_distance(final_head_start_point, final_head_end_point)

    return head_diameter, final_head_start_point, final_head_end_point, head_cnt


def calLongestDiameter_px(img_retouched_bd, contours, tag):
    contours_cp = list(contours)
    contours_cp.sort(key=len)
    
    if len(contours_cp) >= 2:
        head_cnt = np.array(contours_cp[-1]).squeeze()
    else:
        head_cnt = np.array(contours_cp[0]).squeeze()

    print('debug point')
    print('head cnt:', len(head_cnt))
    for cnts in contours_cp:
        print('len cnts', len(cnts))
    
    h, w, c = img_retouched_bd.shape
    cenx = int(w / 2)
    ceny = int(h / 2)
    
    dists = distance.cdist(head_cnt, head_cnt, 'euclidean')
    max_dist = np.max(dists)
    idx = np.unravel_index(np.argmax(dists), dists.shape)
    
    head_centroid = findCentroid(head_cnt)
    
    left_head_cnt = np.array([point for point in head_cnt if point[0] < head_centroid[0]])
    right_head_cnt = np.array([point for point in head_cnt if point[0] > head_centroid[0]])    
     
  
    
    left_head_top_point = left_head_cnt[np.argmin(left_head_cnt[:, 1])]
    std_vec1 = [1, 0]
    
    left_head_vec = [left_head_top_point[0] - head_centroid[0], left_head_top_point[1] - head_centroid[1]]
    rotation_angle = innerProduct(std_vec1, left_head_vec)
    print('rotation_angle: ', rotation_angle)
    left_head_cnt_rotated = np.array(rotate_points(-rotation_angle, left_head_cnt, cenx, ceny))
    
    temp_img = img_retouched_bd.copy()
    for point in left_head_cnt_rotated:
        cv2.circle(temp_img, point, 5, (0, 255, 0) , -1)
    cv2.imwrite(f'./tttttads_{tag}.png', temp_img)
    
    head_start_point = left_head_cnt[np.argmin(left_head_cnt_rotated[:, 1])]
    
    
    head_end_point = right_head_cnt[np.argmax(right_head_cnt[:, 1])]
    
    cv2.circle(img_retouched_bd, head_start_point, 5, (0, 255, 0) , -1)
    cv2.circle(img_retouched_bd, head_end_point, 5, (0, 0, 255) , -1)
    cv2.imwrite(f'./ttdfaas_{tag}.png', img_retouched_bd)
    
    std_vec = [1, 0]
    dist_between = point_distance(head_start_point, head_end_point)
    dist_between = point_distance(head_start_point, head_end_point)
    if dist_between <= 10:
        head_align_angle = 5
    else:
        head_align_vec = [head_end_point[0] - head_start_point[0], head_end_point[1] - head_start_point[1]]
        head_align_angle = innerProduct(std_vec, head_align_vec)
    print(f'{tag} head_align_angle: ', head_align_angle)
    
    
    left_align_angle = 0
    right_align_angle = 0
    
    left_align_angle = head_align_angle
    right_align_angle = head_align_angle 
    
   
    if 25 <= head_align_angle <= 35:
        left_align_angle = head_align_angle * 1.2
        right_align_angle = head_align_angle / 1.5
    elif head_align_angle > 35:
        left_align_angle = head_align_angle * 1.3
        right_align_angle = head_align_angle / 2
    elif 16 <= head_align_angle < 20:
        left_align_angle = head_align_angle / 1.5
        right_align_angle = head_align_angle / 2
    elif 20 <= head_align_angle < 25:
        left_align_angle = head_align_angle * 1.2
        right_align_angle = head_align_angle / 5
    elif head_align_angle <= 15:
        left_align_angle = head_align_angle * 2
        right_align_angle = head_align_angle / 4

    
    left_img = img_retouched_bd.copy()
    right_img = img_retouched_bd.copy()
    
    left_rotated_head_cnt = np.array(rotate_points(left_align_angle, head_cnt, cenx, ceny))
    right_rotated_head_cnt = np.array(rotate_points(right_align_angle, head_cnt, cenx, ceny))
    
    for point in left_rotated_head_cnt:
        cv2.circle(left_img, point, 3, (0, 255, 0), -1)
    
    for point in right_rotated_head_cnt:
        cv2.circle(right_img, point, 3, (0, 255, 0), -1)
    
    
    rotated_left_max_point = left_rotated_head_cnt[np.argmin(left_rotated_head_cnt[:, 0])]
    rotated_right_max_point = right_rotated_head_cnt[np.argmax(right_rotated_head_cnt[:, 0])]
    
    left_max_point = head_cnt[np.argmin(left_rotated_head_cnt[:, 0])]
    right_max_point = head_cnt[np.argmax(right_rotated_head_cnt[:, 0])]
    
    cv2.circle(img_retouched_bd, rotated_left_max_point, 3, (0, 0, 255) , 2)
    cv2.circle(img_retouched_bd, rotated_right_max_point, 3, (0, 0, 255) , 2)
    
    cv2.circle(img_retouched_bd, left_max_point, 3, (100, 0, 255) , 2)
    cv2.circle(img_retouched_bd, right_max_point, 3, (100, 0, 255) , 2)
    
    cv2.imwrite(f'./ttdfaas_{tag}.png', img_retouched_bd)
    

    final_head_start_point = left_max_point
    final_head_end_point = right_max_point
    
    head_diameter = point_distance(final_head_start_point, final_head_end_point)
    
    return head_diameter, final_head_start_point, final_head_end_point, head_cnt

def calLongestDiameter_px_temp(img_retouched_bd, contours):
    contours_cp = list(contours)
    contours_cp.sort(key=len)
    

    if len(contours_cp) > 2:
        head_cnt = np.array(contours_cp[-1]).squeeze()
    else:
        head_cnt = np.array(contours_cp[0]).squeeze()
    
    rect = cv2.minAreaRect(head_cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    print(box)
    cv2.drawContours(img_retouched_bd, [box], 0, (0, 255, 0), 2)
    
    head_centroid = findCentroid(head_cnt)
    cv2.circle(img_retouched_bd, head_centroid, 3, (255, 0, 0) , -1)
    
    left_head_cnt = [point for point in head_cnt if point[0] < head_centroid[0]]
    right_head_cnt = [point for point in head_cnt if point[0] > head_centroid[0]]
    
    left_temp_point = -1
    right_temp_point = -1
    max_left_dist = -9999
    
    std_vec = [1, 0]
    bbox_vector = [box[2][0] - box[1][0], box[2][1] - box[1][1]]
    print('bbox vector:', innerProduct(bbox_vector, std_vec))
    if innerProduct(bbox_vector, std_vec) > 30:
        bbox_vector = [box[1][0] - box[0][0], box[1][1] - box[0][1]]
    
    for point in left_head_cnt:
        current_dist = point_distance(point, head_centroid)
        current_vec = [head_centroid[0] - point[0], head_centroid[1] - point[1]]
        current_angle = innerProduct(bbox_vector, current_vec)
        if current_dist > max_left_dist and current_angle <= 30:
            max_left_dist = current_dist
            left_temp_point = point
    
    min_angle = 180
    
    for point in right_head_cnt:
        current_vec = [point[0] - head_centroid[0], point[1] - head_centroid[1]]
        current_angle = innerProduct(bbox_vector, current_vec)
        #print('current_head_angle:', current_angle)
        if current_angle < min_angle:
            min_angle = current_angle
            right_temp_point = point
    print(min_angle)
    cv2.circle(img_retouched_bd, left_temp_point, 5, (0, 255, 0) , -1)
    cv2.circle(img_retouched_bd, right_temp_point, 5, (0, 0, 255) , -1)
    
    final_head_start_point = left_temp_point
    final_head_end_point = right_temp_point
    
    #cv2.imwrite('./ttttt.png', img_retouched_bd)
    head_diameter = point_distance(final_head_start_point, final_head_end_point)
    return head_diameter, final_head_start_point, final_head_end_point, head_cnt

def calLongestDiameterInConnectedCnt_px(img_retouched_bd, contours):
    contours_cp = list(contours)
    contours_cp_cp = contours_cp.copy()
    contours_cp.sort(key=len)
    contours_cp = contours_cp[:-1]
    contours_area = []
    areas = []
    for contour in contours_cp:
        area = cv2.contourArea(contour)
        if area < 5000:
            continue
        areas.append(area)
    average = np.mean(np.array(areas))
    
    for contour1 in contours_cp:
        area = cv2.contourArea(contour1)
        if area >= average:
            continue
        else:
            contours_area.append(contour1)
            
    contours_area.sort(key=len)
    
    head_cnt = np.array(contours_area[-1]).squeeze()
    dists = distance.cdist(head_cnt, head_cnt, 'euclidean')
    max_dist = np.max(dists)
    idx = np.unravel_index(np.argmax(dists), dists.shape)

    cv2.circle(img_retouched_bd, head_cnt[np.argmax(head_cnt[:, 0])], 3, (0, 0, 255), -1)
    cv2.circle(img_retouched_bd, head_cnt[np.argmin(head_cnt[:, 0])], 3, (0, 0, 255), -1)
    
    left_head_points = []
    right_head_points = []
    
    for point in head_cnt:
        if point[0] == head_cnt[np.argmin(head_cnt[:, 0])][0]:
            cv2.circle(img_retouched_bd, point, 5, (0, 0, 255), -1)
            left_head_points.append(point)
    
    for point in head_cnt:
        if point[0] == head_cnt[np.argmax(head_cnt[:, 0])][0]:
            cv2.circle(img_retouched_bd, point, 5, (0, 0, 255), -1)
            right_head_points.append(point)
    
    plt.imshow(img_retouched_bd)
    plt.title('left right')
    plt.show()
    
    head_start_point = [head_cnt[np.argmin(head_cnt[:, 0])][0], int(np.median(np.array(left_head_points)[:, 1]))]
    head_end_point = [head_cnt[np.argmax(head_cnt[:, 0])][0], right_head_points[np.argmax(np.array(right_head_points)[:, 1])][1]]

    head_align_vec = [head_start_point[0] - head_end_point[0], head_start_point[1] - head_end_point[1]]
    std_vec = [-1, 0]
    
    align_angle = innerProduct(head_align_vec, std_vec)
    h, w, c = img_retouched_bd.shape
    cenx = int(w / 2)
    ceny = int(h / 2)
    print('align_angle:', align_angle)
    
    if align_angle <= 10:
        align_angle = 1.5 * align_angle
    else:
        align_angle = align_angle / 1.5
    
    rotated_head_cnt = np.array(rotate_points(1.5 * align_angle, head_cnt, cenx, ceny))
    for point in rotated_head_cnt:
        cv2.circle(img_retouched_bd, point, 3, (0, 0, 255), -1)
    
    rotated_left_head_points = []
    rotated_right_head_points = []
    
    for point in rotated_head_cnt:
        if point[0] == rotated_head_cnt[np.argmin(rotated_head_cnt[:, 0])][0]:
            cv2.circle(img_retouched_bd, point, 5, (255, 0, 255), -1)
            rotated_left_head_points.append(point)
    
    for point in rotated_head_cnt:
        if point[0] == rotated_head_cnt[np.argmax(rotated_head_cnt[:, 0])][0]:
            cv2.circle(img_retouched_bd, point, 5, (255, 0, 255), -1)
            rotated_right_head_points.append(point)
    
    
    rotated_head_start_point = [rotated_head_cnt[np.argmin(rotated_head_cnt[:, 0])][0], rotated_left_head_points[np.argmin(np.array(rotated_left_head_points)[:, 1])][1]]
    rotated_head_end_point = [rotated_head_cnt[np.argmax(rotated_head_cnt[:, 0])][0], rotated_right_head_points[np.argmax(np.array(rotated_right_head_points)[:, 1])][1]]
    cv2.circle(img_retouched_bd, rotated_head_start_point, 3, (0, 255, 255), -1)
    cv2.circle(img_retouched_bd, rotated_head_end_point, 3, (0, 255, 255), -1)
    
    rotated_head_start_idx = -1
    rotated_head_end_idx = -1
    
    for i, point in enumerate(rotated_head_cnt):
        if point[0] == rotated_head_start_point[0] and point[1] == rotated_head_start_point[1]:
            rotated_head_start_idx = i
        elif point[0] == rotated_head_end_point[0] and point[1] == rotated_head_end_point[1]:
            rotated_head_end_idx = i
    
    final_head_start_point = head_cnt[rotated_head_start_idx]
    final_head_end_point = head_cnt[rotated_head_end_idx]
    
    cv2.circle(img_retouched_bd, final_head_start_point, 7, (0, 255, 255), -1)
    cv2.circle(img_retouched_bd, final_head_end_point, 7, (0, 255, 255), -1)
    
    
    plt.imshow(img_retouched_bd)
    plt.title('rotated_head')
    plt.show()

    head_diameter = point_distance(final_head_start_point, final_head_end_point)
    #return max_dist, head_cnt[idx[0]], head_cnt[idx[1]], head_cnt
    #return head_diameter, head_start_point, head_end_point, head_cnt
    return head_diameter, final_head_start_point, final_head_end_point, head_cnt

def findTopContour(contours):
    bbox = findOnlyBoundingBox(contours)
    
    return bbox[1]

def findMaxHeightInContour(contours):
    max_h_point_candidates = []
    
    for contour in contours:
        cnt = np.array(contour).squeeze()
        if cnt.ndim != 2:
            max_h_point_candidates.append(cnt)
        else:
            y_s = cnt[:, 1]
            argmax_y_s = np.argmin(y_s)
            max_h_point_candidates.append(cnt[argmax_y_s])
    
    max_h_point_candidates = np.array(max_h_point_candidates)
    max_h_point = max_h_point_candidates[np.argmin(max_h_point_candidates[:, 1])]
    
    return max_h_point    

def findMaxHeightInOneContour(contour):
    
    cnt = np.array(contour).squeeze()   
    y_s = cnt[:, 1]
    argmax_y_s = np.argmin(y_s)
    
    max_h_point = cnt[argmax_y_s]
    
    return max_h_point    

def comparebetweenContours(img, contours):
    contours_top_point = findMaxHeightInContour(contours)
    return contours_top_point

def extractInterseactionArea(contour1, contour2, shape):
    h, w, c = shape
    
    mask_img1 = np.zeros(shape)
    mask_img1 = cv2.polylines(mask_img1, [contour1], True, (255, 255, 255), 1)
    mask_img2 = np.zeros(shape)
    mask_img2 = cv2.polylines(mask_img2, [contour2], True, (255, 255, 255), 10)
    subtract = mask_img1 - mask_img2
    
    subtract = subtract.astype(np.uint8)
    _, subtract = cv2.threshold(subtract, 127, 255, cv2.THRESH_BINARY)    
    subtract_gray = cv2.cvtColor(subtract, cv2.COLOR_BGR2GRAY)
    subset_contours, _ = cv2.findContours(subtract_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    subset_contours = list(subset_contours)
    subset_contours.sort(key=len)
    min_y = 9999
    min_y_contours = []
    
    for cnt in subset_contours:
        cnt_np = np.array(cnt).squeeze()
        cnt_length = cv2.arcLength(cnt, closed=False)
        print('cnt_length: ', cv2.arcLength(cnt, closed=False))
        #if cv2.contourArea(cnt) < 10:
        #    continue
        if cnt_np.shape[0] == 2 or cnt_length < 10:
            #cnt_np = cnt_np[1]
            continue
        else:    
            cnt_np = cnt_np[:, 1]
        current_y = np.min(cnt_np)
        if current_y < min_y:
            min_y = current_y
            min_y_contours = cnt
    
    cv2.drawContours(subtract, min_y_contours, -1, (255, 0, 0), 4)
    print('subset_ :', len(subset_contours))
    plt.imshow(subtract)
    plt.show()
    return min_y_contours
    
    
def doEdgeConnection(img, contours):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((11, 11), np.uint8)
    
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    close123 = close.copy()
    close_cp = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(close_cp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)            
    if len(contours) == 1:
        print('first in')
        new_kernel = np.ones((7, 7), np.uint8)
        close = cv2.filter2D(close, -1, new_kernel)
        close_cp = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(close_cp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)    
        contour_cp = list(contours)
        contour_cp.sort(key=len)
        contour_cp = contour_cp[:-1]
    else:
        print('second in')
        contour_cp = list(contours)
        contour_cp.sort(key=len)

        sample_contours = []

        for i, sub_cnt in enumerate(contour_cp):
            dist = cv2.pointPolygonTest(contour_cp[-2], [int(sub_cnt[0][0][0]), int(sub_cnt[0][0][1])], True)
            if dist < 0:
                continue
            if i != len(contour_cp) - 2:
                sample_contours.append(sub_cnt)
        print('lens in second: ', len(contour_cp))
    
    sample_contours.sort(key=len)
    
    
    for k, compare_main_cnt in enumerate(sample_contours[:-1]):
        print(f'{k}번째 cnt')
        compare_main_cnt = np.array(compare_main_cnt).squeeze()
        print(compare_main_cnt.shape)
        if len(compare_main_cnt) == 2:
            continue
        else:
            max_x_in_compare_main = compare_main_cnt[np.argmax(compare_main_cnt[:, 0])]
            min_x_in_compare_main = compare_main_cnt[np.argmin(compare_main_cnt[:, 0])]
    
        
        min_x_dist_points = []
        max_x_dist_points = []
        
        dist_min_x_in_compare_main = 99999
        point_min_x_with_min_angles = []
        
        dist_max_x_in_compare_main = 99999
        point_max_x_with_min_angles = []
        
        final_connect_min_x_point = []
        final_connect_max_x_point = []

        cv2.circle(close, max_x_in_compare_main, 5, (255, 100, 0), -1)
        cv2.circle(close, min_x_in_compare_main, 5, (255, 0, 0), -1)
        
        plt.imshow(close)
        plt.title(f'max_x_in_compare_main, max_x_in_compare_main_{k}')
        plt.show()
        
        for kk, compare_target_cnt in enumerate(sample_contours):
            compare_target_cnt = np.array(compare_target_cnt).squeeze()
            similarity = cv2.matchShapes(compare_main_cnt, compare_target_cnt, cv2.CONTOURS_MATCH_I1, 0)
            if similarity == 0.0:
                continue
            
            epsilon = cv2.arcLength(compare_target_cnt, False) * 0.001
            approx_poly = cv2.approxPolyDP(compare_target_cnt, epsilon, True)
            approx_poly_cp = np.array(approx_poly).squeeze()
            
            min_x_dist = 99999
            max_x_dists = 99999
            min_x_dist_point = []
            max_x_dist_point = []    
            
            temp_angles = []
            
            for kkk, point in enumerate(approx_poly_cp):
                #cv2.circle(close, point, 7, (0, 0, 255), -1)
                
                current_dist_min_x = point_distance(min_x_in_compare_main, point)
                if current_dist_min_x < min_x_dist:
                    min_x_dist = current_dist_min_x
                    min_x_dist_point = point
                
                general_std_vec = [-img.shape[0], 0]
                std_for_min_x = [min_x_in_compare_main[0] - img.shape[0], min_x_in_compare_main[1]]
                general_std_angle = innerProduct(general_std_vec, std_for_min_x)
                #print('general_std_angle: ', general_std_angle)
                
                current_min_x_vec = [point[0] - min_x_in_compare_main[0], point[1] - min_x_in_compare_main[1]]
                angle_for_min_x = innerProduct(std_for_min_x, current_min_x_vec)
                
                if general_std_angle <= 30:
                    if angle_for_min_x <= 20:
                        #cv2.circle(close, point, 5, (0, 255, 0), -1)
                        point_min_x_with_min_angles.append(point)
                elif 30 < general_std_angle <= 45:
                    if angle_for_min_x <= 90:
                    
                        #cv2.circle(close, point, 5, (0, 255, 0), -1)
                        point_min_x_with_min_angles.append(point)
                elif 45 < general_std_angle:
                    if angle_for_min_x <= 110:
                        #cv2.circle(close, point, 5, (0, 255, 0), -1)
                        point_min_x_with_min_angles.append(point)
                
                current_dist_max_x = point_distance(max_x_in_compare_main, point)
                if current_dist_max_x < max_x_dists:
                    max_x_dists = current_dist_max_x
                    max_x_dist_point = point
                
                general_std_vec_for_max = [img.shape[0], 0]
                std_for_max_x = [max_x_in_compare_main[0], max_x_in_compare_main[1]]
                general_std_angle_max_x = innerProduct(general_std_vec_for_max, std_for_max_x)
                #print('general_std_angle_max_x: ', general_std_angle_max_x)
                current_max_x_vec = [point[0] - max_x_in_compare_main[0], point[1] - max_x_in_compare_main[1]]
                angle_for_max_x = innerProduct(std_for_max_x, current_max_x_vec)
                
                if general_std_angle_max_x <= 15:
                    if angle_for_max_x <= 20:
                        #cv2.circle(close, point, 5, (0, 255, 0), -1)
                        point_max_x_with_min_angles.append(point)
                elif 15 < general_std_angle_max_x <= 45:
                    if angle_for_max_x <= 90:
                    
                        cv2.circle(close, point, 5, (0, 255, 0), -1)
                        point_max_x_with_min_angles.append(point)
                elif 45 < general_std_angle_max_x:
                    if angle_for_max_x <= 110:
                        #cv2.circle(close, point, 5, (0, 255, 0), -1)
                        point_max_x_with_min_angles.append(point)
                
                
            min_x_dist_points.append(min_x_dist_point)
            max_x_dist_points.append(max_x_dist_point)
            #print('temp_angle: ', temp_angles)
        
        distance_compare_min_xs = []
        distance_compare_max_xs = []
        
        plt.imshow(close)
        plt.title(f'angle_for_max_x_{k}')
        plt.show()
            
        for point in point_min_x_with_min_angles:
            current_dist = point_distance(point, min_x_in_compare_main)
            if dist_min_x_in_compare_main > current_dist:
                dist_min_x_in_compare_main = current_dist
                final_connect_min_x_point = point
        
        dist_temp_min_x = 99999
            
        for point in min_x_dist_points:
            current_dist = point_distance(point, min_x_in_compare_main)
            if dist_temp_min_x > current_dist:
                dist_temp_min_x = current_dist
                #final_connect_min_x_point = point
                distance_compare_min_xs = point
            
        for point in point_max_x_with_min_angles:
            current_dist = point_distance(point, max_x_in_compare_main)
            if dist_max_x_in_compare_main > current_dist:
                dist_max_x_in_compare_main = current_dist
                final_connect_max_x_point = point
        
        dist_temp_max_x = 99999
            
        for point in max_x_dist_points:
            current_dist = point_distance(point, max_x_in_compare_main)
            #cv2.circle(close123, point, 5, (255, 255, 0), -1)

            if dist_temp_max_x > current_dist:
                dist_temp_max_x = current_dist    
                #final_connect_max_x_point = point
                distance_compare_max_xs = point
    
        
        if final_connect_min_x_point == []:
            print("final_connect_min_x_point is null")
            print(f'{k}, distance_minxs with distance: {point_distance(min_x_in_compare_main, distance_compare_min_xs)}')
        else:
            print(f'{k}, distance_minxs with angle: {point_distance(min_x_in_compare_main, final_connect_min_x_point)}')
            print(f'{k}, distance_minxs with distance: {point_distance(min_x_in_compare_main, distance_compare_min_xs)}')
        if final_connect_max_x_point == []:
            #print(f'{k}, distance_maxxs with angle: {point_distance(max_x_in_compare_main, final_connect_max_x_point)}')
            print("final_connect_max_x_point is null")
            print(f'{k}, distance_maxxs with distance: {point_distance(max_x_in_compare_main, distance_compare_max_xs)}')
        else:
            print(f'{k}, distance_maxxs with angle: {point_distance(max_x_in_compare_main, final_connect_max_x_point)}')
            print(f'{k}, distance_maxxs with distance: {point_distance(max_x_in_compare_main, distance_compare_max_xs)}')
        
        if final_connect_min_x_point == []:
            final_connect_min_x_point = distance_compare_min_xs
        else:
            if point_distance(min_x_in_compare_main, final_connect_min_x_point) >= 50:
                final_connect_min_x_point = distance_compare_min_xs
        
        if final_connect_max_x_point == []:
            final_connect_max_x_point = distance_compare_max_xs
        else:
            if point_distance(max_x_in_compare_main, final_connect_max_x_point) >= 50:
                final_connect_max_x_point = distance_compare_max_xs
        
        print(f'{k} step')
        #cv2.circle(close123, final_connect_min_x_point, 5, (0, 255, 0), -1)
        #cv2.circle(close123, final_connect_max_x_point, 5, (0, 0, 255), -1)
        
        center_connection_min = [int((min_x_in_compare_main[0] + final_connect_min_x_point[0]) / 2),
                     int((min_x_in_compare_main[1] + final_connect_min_x_point[1]) / 2)]
        
        a_line_pelvis = min_x_in_compare_main[0] - final_connect_min_x_point[0]
        b_line_pelvis = min_x_in_compare_main[1] - final_connect_min_x_point[1]
        c_line_pelvis = math.sqrt((a_line_pelvis * a_line_pelvis) + (b_line_pelvis * b_line_pelvis)) / 2
        line_std_vec = [close.shape[1], 0]
        line_target_vec = [final_connect_min_x_point[0] - min_x_in_compare_main[0], final_connect_min_x_point[1] - min_x_in_compare_main[1]]
        line_deg_min = innerProduct(line_std_vec, line_target_vec)
        
        if 120 >= line_deg_min > 90 and min_x_in_compare_main[1] < final_connect_min_x_point[1]:
            cv2.ellipse(close123, center_connection_min, (int(c_line_pelvis), int(c_line_pelvis * 0.15)), line_deg_min , 0, 180, (255, 255, 255), 5)
        elif 120 >= line_deg_min > 90 and min_x_in_compare_main[1] > final_connect_min_x_point[1]:
            cv2.ellipse(close123, center_connection_min, (int(c_line_pelvis), int(c_line_pelvis * 0.15)), -line_deg_min , 0, 180, (255, 255, 255), 5)
        elif line_deg_min > 120 and final_connect_min_x_point[1] > min_x_in_compare_main[1]:
            cv2.ellipse(close123, center_connection_min, (int(c_line_pelvis), int(c_line_pelvis * 0.15)), line_deg_min , 0, 180, (255, 255, 255), 5)
        elif line_deg_min > 120 and final_connect_min_x_point[1] < min_x_in_compare_main[1]:
            cv2.ellipse(close123, center_connection_min, (int(c_line_pelvis), int(c_line_pelvis * 0.15)), 180-line_deg_min , 0, 180, (255, 255, 255), 5)
        else:
            cv2.ellipse(close123, center_connection_min, (int(c_line_pelvis), int(c_line_pelvis * 0.15)), line_deg_min + 180 , 0, 180, (255, 255, 255), 5)
        print(f'min_line_deg_{k}: ', line_deg_min)
        
        center_connection_max = [int((max_x_in_compare_main[0] + final_connect_max_x_point[0]) / 2),
                     int((max_x_in_compare_main[1] + final_connect_max_x_point[1]) / 2)]
        
        a_line_pelvis = max_x_in_compare_main[0] - final_connect_max_x_point[0]
        b_line_pelvis = max_x_in_compare_main[1] - final_connect_max_x_point[1]
        c_line_pelvis = math.sqrt((a_line_pelvis * a_line_pelvis) + (b_line_pelvis * b_line_pelvis)) / 2
        line_std_vec = [close.shape[1], 0]
        line_target_vec = [final_connect_max_x_point[0] - max_x_in_compare_main[0], final_connect_max_x_point[1] - max_x_in_compare_main[1]]
        line_deg_max = innerProduct(line_std_vec, line_target_vec)
        
        if 120 >= line_deg_max > 90 and final_connect_max_x_point[1] > max_x_in_compare_main[1]:
            cv2.ellipse(close123, center_connection_max, (int(c_line_pelvis), int(c_line_pelvis * 0.15)), line_deg_max , 0, 180, (255, 255, 255), 5)
        elif 120 >= line_deg_max > 90 and final_connect_max_x_point[1] < max_x_in_compare_main[1]:
            cv2.ellipse(close123, center_connection_max, (int(c_line_pelvis), int(c_line_pelvis * 0.15)), -line_deg_max , 0, 180, (255, 255, 255), 5)
        elif line_deg_min > 120 and final_connect_max_x_point[1] > max_x_in_compare_main[1]:
            cv2.ellipse(close123, center_connection_max, (int(c_line_pelvis), int(c_line_pelvis * 0.15)), line_deg_max , 0, 180, (255, 255, 255), 5)
        elif line_deg_min > 120 and final_connect_max_x_point[1] < max_x_in_compare_main[1]:
            cv2.ellipse(close123, center_connection_max, (int(c_line_pelvis), int(c_line_pelvis * 0.15)), 180-line_deg_max , 0, 180, (255, 255, 255), 5)
        else:
            cv2.ellipse(close123, center_connection_max, (int(c_line_pelvis), int(c_line_pelvis * 0.15)), line_deg_max + 180 , 0, 180, (255, 255, 255), 5)
        print(f'max_line_deg_{k}: ', line_deg_max)
        
        plt.imshow(close123)
        plt.title(f'final_connect_min_x_point_{k}')
        plt.show()
    
    #close_cp2 = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
    close_cp2 = cv2.cvtColor(close123, cv2.COLOR_BGR2GRAY)
    new_contours, _ = cv2.findContours(close_cp2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    new_contours_cp = list(new_contours)
    new_contours_cp.sort(key=len)
    print('new_contours_cp: ', len(new_contours_cp))
    
    max_centroid = findCentroid(new_contours_cp[-1])
  
    cnt_external = np.array(new_contours_cp[-1]).squeeze()
    cnt_detail = np.array(new_contours_cp[-2]).squeeze()
    
    intersection_area = extractInterseactionArea(cnt_external, cnt_detail, close.shape)
    mask_top_point = findMaxHeightInOneContour(cnt_external)
    new1 = cv2.drawContours(close, intersection_area, -1, (255, 0, 0), 4)
    plt.imshow(new1)
    plt.title('asdfhjklasfjklsda')
    plt.show()
    
    cnt_intersection_area = np.array(intersection_area).squeeze()
    isConnected = False
    for point in cnt_intersection_area:
        if point[1] < max_centroid[1]:
            if mask_top_point[1] == point[1]:
                isConnected = True
                break
            else:
                continue
        else:
            isConnected = True
            break
    print('isConnected: ', isConnected)
    if not isConnected:
        
        if abs(cnt_intersection_area[np.argmax(cnt_intersection_area[:, 1])][0] - cnt_intersection_area[np.argmax(cnt_intersection_area[:, 0])][0]) < 10:
            new_max_x_point = cnt_intersection_area[np.argmax(cnt_intersection_area[:, 1])]
        else:
            new_max_x_point = cnt_intersection_area[np.argmax(cnt_intersection_area[:, 0])]    
        
        cv2.circle(close, new_max_x_point, 3, (255, 100, 0), 1)
        #new_max_x_point = pelvis_neck_in_convergence(intersection_area)[0]
        
        new_min_x_point = cnt_intersection_area[np.argmin(cnt_intersection_area[:, 0])]
        
        max_angle = -1000
        max_angle_new_min_x = -1000
        new_min_point = []
        min_dist_max_x = 99999
        min_dist_min_x = 99999
        new_min_point_new_min_x = []
        #std_vec = [new_min_x_point[0] - new_max_x_point[0], new_min_x_point[1] - new_max_x_point[1]]
        #std_vec = [0 - new_max_x_point[0], 0 - new_max_x_point[1]]
        std_vec = [new_max_x_point[0], new_max_x_point[1]]
        #std_vec = [new_max_x_point[0], 0]
        #std_vec = [0, new_max_x_point[1]]
        subset_cnt_detail = []
        for point in cnt_external:
            current_vec = [point[0] - new_max_x_point[0], point[1] - new_max_x_point[1]]
            angle = innerProduct(std_vec, current_vec)
            
            #if angle < 60:
            if angle < 55:
                subset_cnt_detail.append(point)
                cv2.circle(close, point, 3, (255, 100, 0), 1)
        
        plt.imshow(close)
        plt.title('angle cal')
        plt.show()
        want_angle = []
        new_std_vec = [img.shape[1], 0]
        want_angle_idxs = []
        #for point in cnt_detail:
        for i, point in enumerate(subset_cnt_detail):
            current_vec = [new_max_x_point[0] - point[0], new_max_x_point[1] - point[1]]
            #angle = innerProduct(std_vec, current_vec)
            current_distance_max_x = point_distance(new_max_x_point, point)
            current_want_vec = [point[0] - new_max_x_point[0], point[1] - new_max_x_point[1]]
            want_angle.append(innerProduct(new_std_vec, current_want_vec))
            want_angle_idxs.append(i)
            if new_max_x_point[0] < point[0] and min_dist_max_x > current_distance_max_x: # and max_angle < angle:
                #max_angle = angle
                #print('temp_angle: ', max_angle)
                min_dist_max_x = current_distance_max_x
                new_min_point = point
        
        print(want_angle)
        print(np.max(np.array(want_angle)))
        print(np.min(np.array(want_angle)))
        want_vec = [new_min_point[0] - new_max_x_point[0], new_min_point[1] - new_max_x_point[1]]
        print('angle_want: ', innerProduct(new_std_vec, want_vec))
        
        new_std_vec2 = [-img.shape[1], 0]
        general_vec = [new_max_x_point[0] - img.shape[1], new_max_x_point[1]]
        general_angle2 = innerProduct(general_vec, new_std_vec2)
        print('general_angle2: ', general_angle2)
        if general_angle2 >= 45:
            new_min_point = subset_cnt_detail[want_angle_idxs[np.argmax(np.array(want_angle))]]
        
        
        subset_cnt_detail_min_x = []
        std_vec_min_x = [new_min_x_point[0], 0]
    
        for point in cnt_external:
            current_vec = [new_min_x_point[0] - point[0], new_min_x_point[1] - point[1]]
            angle = innerProduct(std_vec_min_x, current_vec)
            #print(angle)
            if angle < 30:
                subset_cnt_detail_min_x.append(point)
                #cv2.circle(close, point, 4, (255, 100, 0), -1)
        
        plt.imshow(close)
        plt.title('min_distance1')
        plt.show()
        print('subset_cnt_detail_min_x: ', len(subset_cnt_detail_min_x))
        
        for point in subset_cnt_detail_min_x:
            current_distance_min_x = point_distance(new_min_x_point, point)
            #if new_min_x_point[0] > point[0] and current_distance_min_x < min_dist_min_x:
            if new_min_x_point[0] > point[0] and min_dist_min_x > current_distance_min_x:
            #if min_dist_min_x > current_distance_min_x:
                min_dist_min_x = current_distance_min_x
                new_min_point_new_min_x = point
        
        close1 = cv2.circle(close, new_min_x_point, 5, (0, 0 ,255), -1)
        
        #print(new_min_point)
        plt.imshow(close1)
        plt.title('min_distance2')
        plt.show()
        
        if new_min_point_new_min_x == []:
            cv2.line(close, new_min_x_point, new_min_point_new_min_x, (255, 255, 255), 7)
        else:
            cv2.line(close, new_min_x_point, new_min_point_new_min_x, (255, 255, 255), 7)
        
        center_connection = [int((new_max_x_point[0] + new_min_point[0]) / 2),
                     int((new_max_x_point[1] + new_min_point[1]) / 2)]
        
        a_line_pelvis = new_max_x_point[0] - new_min_point[0]
        b_line_pelvis = new_max_x_point[1] - new_min_point[1]
        c_line_pelvis = math.sqrt((a_line_pelvis * a_line_pelvis) + (b_line_pelvis * b_line_pelvis)) / 2
        line_std_vec = [close.shape[1], 0]
        line_target_vec = [new_min_point[0] - new_max_x_point[0], new_min_point[1] - new_max_x_point[1]]
        line_deg = innerProduct(line_std_vec, line_target_vec)
        
        cv2.ellipse(close, center_connection, (int(c_line_pelvis), int(c_line_pelvis * 0.2)), line_deg + 180, 0, 180, (255, 255, 255), 10)
        
        
        close_cp2 = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
        new_contours, _ = cv2.findContours(close_cp2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    for i, cnt in enumerate(new_contours):
        _, img_drawed = findBoundingBox(close, cnt)
        
        cv2.drawContours(img_drawed, cnt, -1, (255, 0, 0), 4)
        plt.imshow(img_drawed)
        plt.title(f'{i}')
        plt.show()
    
    return new_contours

def line_from_two_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def position_of_point(m, b, point):
    x, y = point
    line_y = m * x + b

    if y > line_y: 
        return 1
    elif y < line_y: 
        return -1
    else: 
        return 1

def find_perpendicular_and_on_line_points(points, line):
    A = np.array(line[0])
    B = np.array(line[1])
    P = np.array(points)
    
    t = np.dot(P - A, B - A) / np.dot(B - A, B - A)
    projection = A + t[:, None] * (B - A)

    distances = np.sqrt(np.sum((P - projection) ** 2, axis=1))
    idx_perpendicular = np.argmax(distances)
    AP = P - A
    AB = B - A

    return points[idx_perpendicular]

def cross_product(p1, p2):
    return p1[0] * p2[1] - p2[0] * p1[1]



def find_intersection_point(line, point):
    A = np.array(line[0])
    B = np.array(line[1])
    P = np.array(point)
    
    AB = B - A

    t = np.dot(P - A, AB) / np.dot(AB, AB)
    D = A + t * AB

    return D

def calFemoralHeadHeight(img, head_start_point, head_end_point, head_cnt):
       
    cv2.circle(img, head_start_point, 7, (255, 0, 255), -1)
    cv2.circle(img, head_end_point, 7, (255, 0, 255), -1)
    
    plt.imshow(img)
    plt.title('find head start end')
    plt.show()
    
    long_diameter_m, long_diameter_y_value = line_from_two_points(head_start_point, head_end_point)
    
    temp_point = []
    
    for point in head_cnt:
        current_dist = calculate_distance_between_line(head_start_point[0], head_start_point[1], head_end_point[0], head_end_point[1], point[0], point[1])
        if current_dist <= 0.5:
            temp_point.append(point)
    
    area = cv2.contourArea(head_cnt)
    print('area: ', area)
    print('temp_point:', len(temp_point))

    line_top_points = []
    line_bottom_points = []
    
    for point in head_cnt:
        std_value = position_of_point(long_diameter_m, long_diameter_y_value, point)
        if std_value == 1:
            line_bottom_points.append(point)
        else:
            line_top_points.append(point)
    
    for point in line_top_points:
        cv2.circle(img, point, 3, (0, 255, 0), -1)
        
    
    plt.imshow(img)
    plt.show()
    
    y_90_point = []
    
    #middle_point_x = (head_start_point[0] + head_end_point[0]) / 2
    #line_bottom_points = [point for point in line_bottom_points if point[0] > middle_point_x - 50 and point[0] < middle_point_x + 50] 
    
    top_height_point = find_perpendicular_and_on_line_points(line_top_points, [head_start_point, head_end_point])
    bottom_height_point = find_perpendicular_and_on_line_points(line_bottom_points, [head_start_point, head_end_point])

    on_line_points_top = find_intersection_point([head_start_point, head_end_point], top_height_point)
    on_line_points_bottom = find_intersection_point([head_start_point, head_end_point], bottom_height_point)
    
    print(on_line_points_top)
    print(on_line_points_bottom)
    print(top_height_point)
    print(bottom_height_point)
    on_line_points_top = [int(on_line_points_top[0]), int(on_line_points_top[1])]
    on_line_points_bottom = [int(on_line_points_bottom[0]), int(on_line_points_bottom[1])]
    cv2.circle(img, top_height_point, 7, (0, 255, 255), -1)
    cv2.circle(img, bottom_height_point, 7, (0, 255, 255), -1)
    cv2.circle(img, [int(on_line_points_top[0]), int(on_line_points_top[1])], 7, (0, 255, 255), -1)
    cv2.circle(img, [int(on_line_points_bottom[0]), int(on_line_points_bottom[1])], 7, (0, 255, 255), -1)
    cv2.line(img, [int(on_line_points_top[0]), int(on_line_points_top[1])], top_height_point, (0, 0, 255), 3)
    cv2.line(img, [int(on_line_points_bottom[0]), int(on_line_points_bottom[1])], bottom_height_point, (0, 0, 255), 3)
    cv2.line(img, head_start_point, head_end_point, (0, 0, 255), 3)

    top_vector = [int(on_line_points_top[0]) - top_height_point[0], int(on_line_points_top[1]) - top_height_point[1]]
    bottom_vector = [int(on_line_points_bottom[0]) - bottom_height_point[0], int(on_line_points_bottom[1]) - bottom_height_point[1]]
    line_vector = [head_end_point[0] - head_start_point[0], head_end_point[1] - head_start_point[1]]
    
    top_angle = innerProduct(top_vector, line_vector)
    bottom_angle = innerProduct(bottom_vector, line_vector)
    
    print('top_angle:', top_angle)
    print('bottom_angle:', bottom_angle)
    


    plt.imshow(img)
    plt.title('line on image')
    plt.show()
    '''for point_1 in line_bottom_points:
        #cv2.circle(img, point, 3, (0, 0, 255), -1)
        for point_2 in line_top_points:
            current_vec = [point_2[0] - point_1[0], point_2[1] - point_1[1]]
            long_diameter_vec = [head_end_point[0] -  head_start_point[0], head_end_point[1] -  head_start_point[1]]
            
            angle = innerProduct(current_vec, long_diameter_vec)
            
            if 90 <= angle <= 91:
                y_90_point.append([point_1, point_2])
        
    max_dist = -99999
    final_points = []
    print('y_90_point: ', len(y_90_point))
    for points in y_90_point:
        #cv2.line(img, points[0], points[1], (0, 0, 255), 3)
        current_dist = point_distance(points[0], points[1])
        if current_dist > max_dist:
            max_dist = current_dist
            final_points = points
            
    if final_points == []:
        print('None of final points')
        final_points = [head_cnt[0], head_cnt[-1]]
    
    cv2.line(img, final_points[0], final_points[1], (0, 0, 255), 3)'''
    
    plt.imshow(img)
    plt.title('test1')
    plt.show()
    
    #return final_points[0], final_points[1]
    return top_height_point, bottom_height_point, on_line_points_top, on_line_points_bottom, top_height_point, bottom_height_point
    

def calHeadSize(img_retouched_bd, tag):
    new_mask_img = img_retouched_bd.copy()
    len_, contours = countContours(img_retouched_bd)
    print('first_len: ', len_)
    print(calContourArea(contours))
    max_dist, head_px_1, head_px_2, head_cnt = calLongestDiameter_px_before(img_retouched_bd, contours, tag)
    head_height_start_point, head_height_end_point, on_line_points_top, on_line_points_bottom, top_height_point, bottom_height_point = calFemoralHeadHeight(new_mask_img, head_px_1, head_px_2, head_cnt)

    return max_dist, head_px_1, head_px_2, head_height_start_point, head_height_end_point, on_line_points_top, on_line_points_bottom, top_height_point, bottom_height_point
    
        

def find_femoral_head_diameter(img_femoral, img_pelvis, img_femoral_bd, img_femoral_head, img_femoral_head_bd, tag):
    img_erased_noise = erasingNoise(img_pelvis, img_femoral_bd)
    
    #img_erased_noise_head = erasingNoise(img_femoral_head, img_femoral_bd)
    contour_head, _ = cv2.findContours(cv2.cvtColor(img_femoral_head, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour_head = list(contour_head)
    contour_head.sort(key=len)
    print('find_femoral_head_diameter contour:', len(contour_head))
    if len(contour_head) == 1:
        print(cv2.contourArea(contour_head[0]))
        if cv2.contourArea(contour_head[0]) < 150:
            img_erased_noise_head = img_femoral_head
            if cv2.contourArea(contour_head[0]) < 10:
                return None
        else:
            img_erased_noise_head = erasingNoise(img_femoral_head, img_femoral_bd)
            #img_erased_noise_head = erasingNoise(img_femoral_head_bd, img_femoral_bd)
            #cv2.imwrite('./ttdafd.png', img_erased_noise_head)
    elif len(contour_head) == 0:
        #return None
        img_erased_noise_head = erasingNoise(img_pelvis, img_femoral_bd)
        contour_head_erased_0, _ = cv2.findContours(cv2.cvtColor(img_erased_noise_head, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contour_head_erased_0) == 1:
            return None
    else:
        print('here is head size: ', cv2.contourArea(contour_head[-1]))
        if cv2.contourArea(contour_head[-1]) < 100:
            img_erased_noise_head = img_femoral_head
        else:
            img_erased_noise_head = erasingNoise(img_femoral_head, img_femoral_bd)
            contour_head_erased, _ = cv2.findContours(cv2.cvtColor(img_erased_noise_head, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contour_head_erased) > 1:
                #img_erased_noise_head = erasingNoise(img_pelvis, img_femoral_bd)
                img_erased_noise_head = erasingNoise(img_femoral_head, img_erased_noise_head)
                contour_head_erased_new, _ = cv2.findContours(cv2.cvtColor(img_erased_noise_head, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour_head_erased_new = list(contour_head_erased_new)
                contour_head_erased_new.sort(key=len)
                empty = np.zeros(img_femoral_bd.shape)
                empty = empty.astype(dtype=np.uint8)
                new_head_contours = np.array(contour_head_erased_new[-1]).squeeze()
                img_erased_noise_head = cv2.fillConvexPoly(empty, new_head_contours, (255, 255, 255))
                #cv2.imwrite('./ttdafd.png', img_erased_noise_head)
        
    checkHead = hasFemoralHead(img_femoral, img_erased_noise)
    femoral_head_param = []
    if checkHead:
        max_dist, head_px_1, head_px_2, head_height_start_point, head_height_end_point, on_line_points_top, on_line_points_bottom, top_height_point, bottom_height_point = calHeadSize(img_erased_noise_head, tag) # px-px 간 거리 최대 기준
        femoral_head_param = [head_px_1, head_px_2, max_dist, head_height_start_point, head_height_end_point, on_line_points_top, on_line_points_bottom, top_height_point, bottom_height_point]
        return femoral_head_param
    else:
        return None

def draw_femoral_head_diameter(output_path, img_name, img_origin, left_femoral_head_param, right_femoral_head_param, left_pelvis_cropped_point, right_pelvis_cropped_point, right_femoral_shape):
    txt_fontFace = 0
    txt_fontScale = 1
    txt_color = (100, 0, 255)
    txt_thickness = 3
    #print(left_femoral_head_param)
    #print(right_femoral_head_param)
    if left_femoral_head_param != None:
        left_head_start_point_x = left_femoral_head_param[0][0] + left_pelvis_cropped_point[0]
        left_head_start_point_y = left_femoral_head_param[0][1] + left_pelvis_cropped_point[1]
    
        left_head_end_point_x = left_femoral_head_param[1][0] + left_pelvis_cropped_point[0]
        left_head_end_point_y = left_femoral_head_param[1][1] + left_pelvis_cropped_point[1]
    
        cv2.line(img_origin, [left_head_start_point_x, left_head_start_point_y], [left_head_end_point_x, left_head_end_point_y], (0 ,100, 255), 3)
        cv2.putText(img_origin, str(round(left_femoral_head_param[2] * 0.14, 2))+'mm', (left_head_start_point_x-180, left_head_start_point_y), fontFace=txt_fontFace, fontScale=txt_fontScale, color=txt_color, thickness=txt_thickness)
        with open(output_path + img_name + '/' + 'left_femoral_param.txt', 'w') as f:
            f.write(str(left_femoral_head_param[2] * 0.14) + '\n') 
            
    if right_femoral_head_param != None:
        r_h, r_w, _ = right_femoral_shape    
    
        right_head_start_point_x = r_w - right_femoral_head_param[0][0] + right_pelvis_cropped_point[0]
        right_head_start_point_y = right_femoral_head_param[0][1] + right_pelvis_cropped_point[1]
    
        right_head_end_point_x = r_w - right_femoral_head_param[1][0] + right_pelvis_cropped_point[0]
        right_head_end_point_y = right_femoral_head_param[1][1] + right_pelvis_cropped_point[1]
    
        cv2.line(img_origin, [right_head_start_point_x, right_head_start_point_y], [right_head_end_point_x, right_head_end_point_y], (0 ,100, 255), 3)
        cv2.putText(img_origin, str(round(right_femoral_head_param[2] * 0.14, 2))+'mm', (right_head_start_point_x + 50, right_head_start_point_y), fontFace=txt_fontFace, fontScale=txt_fontScale, color=txt_color, thickness=txt_thickness)
        with open(output_path + img_name + '/' + 'right_femoral_param.txt', 'w') as f:
            f.write(str(right_femoral_head_param[2] * 0.14) + '\n')
    cv2.imwrite(f'{output_path}/{img_name}/{img_name}_femoral_head_param.jpg', img_origin)
    
    
        