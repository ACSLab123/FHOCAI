import ultralytics
import os
import sys
import argparse
import cv2
sys.path.append('/home/moon/child_proj/all_system')
from utils.dcm2jpg import *
from utils.tools import *
from detection.predict import *
from segmentation.predict import *
from algorithm.pubic_bone_index import *
from algorithm.shenton_line import *
from algorithm.pelvic_tilt_index import *
from algorithm.acetabular_index import *
from algorithm.rotational_index import *
from algorithm.femoral_head_diameter import *
from landmark_detection.landmark_detection import *
from tqdm import tqdm
import traceback

# flow 순서
# inputs dcm or jpg branch 제작 (230206)

exception_list = []

def main(opt):
    img_png, ps_x_list, ps_y_list, col, row = Dcm2jpg(opt.input_path)

    jpg_img_list = os.listdir(opt.input_path + 'jpg')
    
    
    for jpg_img_name in tqdm(jpg_img_list):
        try:
            output_dir = opt.output_path + jpg_img_name[:-4]
            print('img_name: ', jpg_img_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
            img_origin = cv2.imread(opt.input_path + 'jpg/' + jpg_img_name)
            h, w, c = img_origin.shape
            
            detection(opt.input_path + 'jpg/' + jpg_img_name, img_origin, opt.output_path, jpg_img_name[:-4])
            segmentation(opt.output_path, jpg_img_name[:-4])
            
            # ===========================================================================================================================================================================
        
            part_list = ['left_pubic_bone', 'right_pubic_bone']
            img_inner_list = []

            for part in part_list:
                img_pubic_bone = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], part, jpg_img_name[:-4] + '_mask_removal.png'))
                if part == 'right_pubic_bone':
                    img_pubic_bone = cv2.flip(img_pubic_bone, 1)
                
                img_pubic_bone_cp = img_pubic_bone.copy()
        
            # ===========================================================================================================================================================================
        
        
            img_origin = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/' + jpg_img_name[:-4] + '/' + jpg_img_name[:-4] + '_origin.jpg')

            img_left_pelvis = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], 'left_femoral_head', jpg_img_name[:-4] + '_mask_removal.png'))
            img_left_femoral = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], 'left_femoral_head', jpg_img_name[:-4] + '.jpg'))
            img_left_femoral_bd = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], 'left_femoral_head', jpg_img_name[:-4] + '_boundary.png'))
            img_left_femoral_head = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], 'left_femoral_head', jpg_img_name[:-4] + '_head_mask.png'))
            img_left_femoral_head_bd = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], 'left_femoral_head', jpg_img_name[:-4] + '_head_boundary.png'))
            img_left_pubic = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], 'left_pubic_bone', jpg_img_name[:-4] + '_mask_removal.png'))
            
            img_left_pubic_cp = img_left_pubic.copy()
        
            img_right_pelvis = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], 'right_femoral_head', jpg_img_name[:-4] + '_mask_removal.png'))
            img_right_femoral = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], 'right_femoral_head', jpg_img_name[:-4] + '.jpg'))
            img_right_femoral_bd = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], 'right_femoral_head', jpg_img_name[:-4] + '_boundary.png'))
            img_right_femoral_head = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], 'right_femoral_head', jpg_img_name[:-4] + '_head_mask.png'))
            img_right_femoral_head_bd = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], 'right_femoral_head', jpg_img_name[:-4] + '_head_boundary.png'))
            img_right_pubic = cv2.imread('/home/moon/child_proj/all_system/inference/outputs/{}/{}/{}'.format(jpg_img_name[:-4], 'right_pubic_bone', jpg_img_name[:-4] + '_mask_removal.png'))   
        
            left_pelvis_cropped_point = read_coordinate('/home/moon/child_proj/all_system/inference/outputs/{}/{}'.format(jpg_img_name[:-4], 'left_femoral_head/roi_coordinate.txt'))
            left_pubic_bone_cropped_point = read_coordinate('/home/moon/child_proj/all_system/inference/outputs/{}/{}'.format(jpg_img_name[:-4], 'left_pubic_bone/roi_coordinate.txt'))
        
            right_pelvis_cropped_point = read_coordinate('/home/moon/child_proj/all_system/inference/outputs/{}/{}'.format(jpg_img_name[:-4], 'right_femoral_head/roi_coordinate.txt'))
            right_pubic_bone_cropped_point = read_coordinate('/home/moon/child_proj/all_system/inference/outputs/{}/{}'.format(jpg_img_name[:-4], 'right_pubic_bone/roi_coordinate.txt'))

            total_cropped_points = []
        
            #===================== big size change ======================
        
            big_left_pelvis = np.zeros(img_origin.shape)
            big_right_pelvis = np.zeros(img_origin.shape)
            big_left_pubic_bone = np.zeros(img_origin.shape)
            big_right_pubic_bone = np.zeros(img_origin.shape)
            big_left_pubic_bone_inner = np.zeros(img_origin.shape)
            big_right_pubic_bone_inner = np.zeros(img_origin.shape)
            
            img_inner_final_list = []
            for img_inner in img_inner_list:
                img_inner_three_dims = cv2.cvtColor(img_inner, cv2.COLOR_GRAY2BGR)
                img_inner_final_list.append(img_inner_three_dims)
            
            big_left_pelvis[left_pelvis_cropped_point[1]: left_pelvis_cropped_point[1] + img_left_pelvis.shape[0], left_pelvis_cropped_point[0]: left_pelvis_cropped_point[0] + img_left_pelvis.shape[1]] = img_left_pelvis
            big_right_pelvis[right_pelvis_cropped_point[1]: right_pelvis_cropped_point[1] + img_right_pelvis.shape[0], right_pelvis_cropped_point[0]: right_pelvis_cropped_point[0] + img_right_pelvis.shape[1]] = img_right_pelvis
            big_left_pubic_bone[left_pubic_bone_cropped_point[1]: left_pubic_bone_cropped_point[1] + img_left_pubic.shape[0], left_pubic_bone_cropped_point[0]: left_pubic_bone_cropped_point[0] + img_left_pubic.shape[1]] = img_left_pubic
            big_right_pubic_bone[right_pubic_bone_cropped_point[1]: right_pubic_bone_cropped_point[1] + img_right_pubic.shape[0], right_pubic_bone_cropped_point[0]: right_pubic_bone_cropped_point[0] + img_right_pubic.shape[1]] = img_right_pubic
            big_left_pubic_bone_inner[left_pubic_bone_cropped_point[1]: left_pubic_bone_cropped_point[1] + img_inner_list[0].shape[0], left_pubic_bone_cropped_point[0]: left_pubic_bone_cropped_point[0] + img_inner_list[0].shape[1]] = img_inner_final_list[0]
            big_right_pubic_bone_inner[right_pubic_bone_cropped_point[1]: right_pubic_bone_cropped_point[1] + img_inner_list[1].shape[0], right_pubic_bone_cropped_point[0]: right_pubic_bone_cropped_point[0] + img_inner_list[1].shape[1]] = img_inner_final_list[1]
            
            big_left_pelvis = big_left_pelvis.astype(dtype=np.uint8)
            big_right_pelvis = big_right_pelvis.astype(dtype=np.uint8)
            big_left_pubic_bone = big_left_pubic_bone.astype(dtype=np.uint8)
            big_right_pubic_bone = big_right_pubic_bone.astype(dtype=np.uint8)
        
            big_right_pelvis = cv2.flip(big_right_pelvis, 1)
            big_right_pubic_bone = cv2.flip(big_right_pubic_bone, 1)
            big_right_pubic_bone_inner = cv2.flip(big_right_pubic_bone_inner, 1)
            
            left_femoral_head_param = find_femoral_head_diameter(img_left_femoral, img_left_pelvis, img_left_femoral_bd, img_left_femoral_head, img_left_femoral_head_bd)
            right_femoral_head_param = find_femoral_head_diameter(cv2.flip(img_right_femoral, 1), cv2.flip(img_right_pelvis, 1), img_right_femoral_bd, img_right_femoral_head, img_right_femoral_head_bd)
            
            draw_femoral_head_diameter(opt.output_path, jpg_img_name[:-4], img_origin, left_femoral_head_param, right_femoral_head_param, left_pelvis_cropped_point, right_pelvis_cropped_point, img_right_pelvis.shape)            
        except:
            exception_list.append(jpg_img_name)
            traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='', help='inputs the image path for testing')
    parser.add_argument('--output_path', type=str, default='', help='inputs the output path')

    opt = parser.parse_args()

    main(opt)
