import os
import cv2
import numpy as np


def small_object_remover(img_path, part):
    # segmentation 후 생성된 mask에서 자잘하게 생긴 부분들을 지우기 위한 코드
    new = cv2.imread(img_path)
    new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    new[new < new.mean()] = 0
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(new, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = max(sizes)
    image = np.zeros((output.shape))
    for i in range(0, nb_components):
       if sizes[i] >= min_size:
           image[output == i + 1] = 255
    #image_gray = cv2.cvtColor(image.astype(dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(image.astype(dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        for cnt in contours:
            cnt = np.array(cnt).squeeze()
            if cnt.shape[0] < 30:
                cv2.circle(image, cnt[0], 5, (255,255,255),-1)
    
    if part == 'right_pubic_bone':
        image = cv2.flip(image, 1)
    cv2.imwrite(img_path[:-4]+'_removal.png',image) 
    
def small_object_remover_pelvis(img_path, part):
    # segmentation 후 생성된 mask에서 자잘하게 생긴 부분들을 지우기 위한 코드
    new = cv2.imread(img_path)
    new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    new[new < new.mean()] = 0
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(new, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = max(sizes)
    image = np.zeros((output.shape))
    for i in range(0, nb_components):
        image[output == i + 1] = 255
           
    if part == 'right_femoral_head_only':
        image = cv2.flip(image, 1)
        
    cv2.imwrite(img_path[:-4]+'_removal.png',image)
    
def read_coordinate(coord_path):
    coord_list = []
    with open(coord_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            left_x, left_y, right_x, right_y = line.split(',')
            coord_list.append(int(left_x))
            coord_list.append(int(left_y))
            coord_list.append(int(right_x))
            coord_list.append(int(right_y))
                     
    return coord_list