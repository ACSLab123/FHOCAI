import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from scipy import misc
import sys
sys.path.append('/home/moon/child_proj/abdomen_system/segmentation')
from segmentation.models.xnet import XNet
from utils.dataloader import get_dataloaders
import imageio
from skimage import img_as_ubyte
import cv2
import argparse
from tqdm import tqdm
sys.path.append('/home/moon/child_proj/abdomen_system/utils')
import time
from utils.tools import *
from torchvision import transforms

def segmentation(output_path, img_name):
   print('Segmentation start')
   start_time = time.time()
   parts_list = ['left_femoral_head_only', 'right_femoral_head_only']

   for idx, part in enumerate(parts_list):

      if part == 'left_femoral_head_only' or part == 'right_femoral_head_only':
         pth_path = "/home/moon/child_proj/abdomen_system/segmentation/weights/femoral/best_model.pt"

      
      if part == 'left_femoral_head_only' or part == 'right_femoral_head_only':
         img_path = '/home/moon/child_proj/abdomen_system/inference/outputs/' + img_name + '/' + part[:-5] + '/'
      else:
         img_path = '/home/moon/child_proj/abdomen_system/inference/outputs/' + img_name + '/' + part + '/'
      

      if part == 'left_femoral_head_only' or part == 'right_femoral_head_only':
         model = XNet()
   
      if part == 'left_femoral_head_only' or part == 'right_femoral_head_only':
         save_path = output_path + img_name + '/' + part[:-5] + '/'
      else:
         save_path = output_path + img_name + '/' + part + '/'   

      state_dict = torch.load(pth_path)
      model.load_state_dict(state_dict["model_state_dict"])
      model.cuda()
      model.eval()
        
      transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((448, 448), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
         )
        
      if part == 'right_femoral_head_only':
         data = cv2.imread(img_path + img_name + '.jpg')
         data = cv2.flip(data, 1)
      else:
         data = cv2.imread(img_path + img_name + '.jpg')
      

      h, w, c = data.shape
      data = transform_input4test(data).unsqueeze(dim=0)
        
      data = data.cuda()
      with torch.no_grad():
         output_all, output_head, reg_all, reg_head = model(data, data)

      predicted_map_all = np.array(output_all.detach().cpu())
      predicted_reg_map_all = np.array(reg_all.detach().cpu())
      
      predicted_map_head = np.array(output_head.detach().cpu())
      predicted_reg_map_head = np.array(reg_head.detach().cpu())
         
      predicted_map_all = np.squeeze(predicted_map_all)
      predicted_reg_map_all = np.squeeze(predicted_reg_map_all)
      
      predicted_map_head = np.squeeze(predicted_map_head)
      predicted_reg_map_head = np.squeeze(predicted_reg_map_head)
            
      predicted_map_all = cv2.resize(predicted_map_all, (w, h))
      predicted_reg_map_all = cv2.resize(predicted_reg_map_all, (w, h))
      
      predicted_map_head = cv2.resize(predicted_map_head, (w, h))
      predicted_reg_map_head = cv2.resize(predicted_reg_map_head, (w, h))
      #predicted_map = np.where(predicted_map > 0, 255, 0)
      #predicted_reg_map = np.where(predicted_reg_map > 0, 255, 0)
            
      predicted_map_all = predicted_map_all > 0
      predicted_reg_map_all = predicted_reg_map_all > 0
      
      predicted_map_head = predicted_map_head > 0
      predicted_reg_map_head = predicted_reg_map_head > 0
      
      predicted_map_all = predicted_map_all * 255
      predicted_reg_map_all = predicted_reg_map_all * 255
      
      predicted_map_head = predicted_map_head * 255
      predicted_reg_map_head = predicted_reg_map_head * 255
      
 
            
      cv2.imwrite(save_path + img_name + '_mask.png', predicted_map_all)
      cv2.imwrite(save_path + img_name + '_boundary.png', predicted_reg_map_all)
      
      cv2.imwrite(save_path + img_name + '_head_mask.png', predicted_map_head)
      cv2.imwrite(save_path + img_name + '_head_boundary.png', predicted_reg_map_head)
            
      small_object_remover_pelvis(save_path + img_name + '_mask.png', part)

            
   end_time = time.time()
    
   print('Segmentation finish, {} sec'.format(end_time - start_time))