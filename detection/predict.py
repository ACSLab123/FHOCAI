from ultralytics import YOLO
import os
import cv2
import numpy as np

def cal_iou(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def compare_conf(boxes):
    conf_list = []
    
    for box_condi in boxes:
        conf_list.append(box_condi[-1])
    
    final_box_index = np.argmax(np.array(conf_list))    
    cls_num = boxes[0][0]
    
    return final_box_index, cls_num
    
def check_class_null(cls_list, box_temp):
    pass

def check_final(boxes):
    left_x_list = [[],[],[],[]]
    
    for box in boxes:
        if box[0][0] == 0:
            left_x_list[0].append(box[1][0])
        elif box[0][0] == 1:
            left_x_list[1].append(box[1][0])
        elif box[0][0] == 2:
            left_x_list[2].append(box[1][0])
        elif box[0][0] == 3:
            left_x_list[3].append(box[1][0])
    
    if left_x_list[0][0] > left_x_list[1][0]:
        boxes[0][0] = np.array([1])
        boxes[1][0] = np.array([0])
            
    if left_x_list[2][0] > left_x_list[3][0]:
        boxes[2][0] = np.array([3])
        boxes[3][0] = np.array([2])
    
    return boxes


def draw_bbox(img, boxes, img_name):
    
    color_list = [(0, 0, 255),
                  (0, 255, 0),
                  (255, 0, 0),
                  (100, 100, 100)]
    
    for box in boxes:
        left_x = int(box[1][0])
        left_y = int(box[1][1])
        right_x = int(box[1][2])
        
        right_y = int(box[1][3])
        cv2.rectangle(img, (left_x, left_y), (right_x, right_y), color_list[int(box[0][0])],2)
    
    cv2.imwrite('./data/outputs/' + img_name, img)
    
    
def bbox_refinement(left_y, right_y, left_x, right_x, ori_h, ori_w):
    w = abs(right_x - left_x)
    h = abs(right_y - left_y)
    
    if w < h:
        diff = abs(h - w)
        left_x = int(left_x - (diff / 2))
        right_x = int(right_x + (diff / 2))
    else:
        diff = abs(w - h)
        left_y = int(left_y - (diff / 2))
        right_y = int(right_y + (diff / 2))
    
    left_x = left_x - 50
    left_y = left_y
    
    right_x = right_x + 50
    right_y = right_y + 100
    
    if left_x < 0:
        left_x = 0
        
    if left_y < 0:
        left_y = 0
    
    if right_x > ori_w:
        right_x = ori_w
        
    if right_y > ori_h:
        right_y = ori_h
    
    w = abs(right_x - left_x)
    h = abs(right_y - left_y)

    return left_y, right_y, left_x, right_x
    
def detection(input_image_path, img_origin, output_path, img_name):
    # Load a model
    # model = YOLO("yolov8n.pt")  # load an official model
    model = YOLO("../detection/weights/best.pt")  # load a custom model 
    
    # Predict with the model
    #results = model.predict(source=input_image_path, imgsz=640)
    results = model.predict(img_origin, imgsz=640)
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        box_cls_list = []
        boxes_temp = [[], [], [], []]
        step2_boxes = []
        #print(boxes)        
        for box in boxes:
            box_xyxy = box.xyxy.cpu().numpy()[0]
            box_conf = box.conf.cpu().numpy()
            box_cls = box.cls.cpu().numpy()
            if box_cls[0] == 0:
                boxes_temp[0].append([box_cls, box_xyxy, box_conf])
            elif box_cls[0] == 1:
                boxes_temp[1].append([box_cls, box_xyxy, box_conf])
            elif box_cls[0] == 2:
                boxes_temp[2].append([box_cls, box_xyxy, box_conf])
            elif box_cls[0] == 3:
                boxes_temp[3].append([box_cls, box_xyxy, box_conf])
        
        """ print(boxes_temp[0])
        print(boxes_temp[1])
        print(boxes_temp[2])
        print(boxes_temp[3])
         """
        #print('temp!!!!!!!')
        #print(boxes_temp)
        for boxes_tmp in boxes_temp:
            if len(boxes_tmp) > 1:
                idx, cls_num = compare_conf(boxes_tmp)
                step2_boxes.append(boxes_temp[int(cls_num[0])][idx])
            else:
                step2_boxes.append(boxes_tmp[0])
        
        """ print("====="*5)
        print(step2_boxes[0])
        print(step2_boxes[1])
        print(step2_boxes[2])
        print(step2_boxes[3])
         """
        final_box_list = check_final(step2_boxes)
        
        ori_h, ori_w, _ = img_origin.shape
        
        for box in final_box_list:
            box_xyxy = box[1]
            box_conf = box[-1]
            box_cls = box[0]
                        
            if box_cls[0] == 0:
               left_x = int(box_xyxy[0])
               left_y = int(box_xyxy[1])
               right_x = int(box_xyxy[2])
               right_y = int(box_xyxy[3])
               
               if not os.path.exists(output_path + img_name + '/left_femoral_head'):
                   os.makedirs(output_path + img_name + '/left_femoral_head')
               
               left_y, right_y, left_x, right_x = bbox_refinement(left_y, right_y, left_x, right_x, ori_h, ori_w)
               
               img_crop = img_origin[left_y: right_y, left_x: right_x]
               
               with open(output_path + img_name + '/left_femoral_head/roi_coordinate.txt', 'w') as f1:
                    f1.write(str(int(left_x)))
                    f1.write(",")
                    f1.write(str(int(left_y)))
                    f1.write(",")
                    f1.write(str(int(right_x)))
                    f1.write(",")
                    f1.write(str(int(right_y)))
               
               cv2.imwrite(output_path + img_name + '/left_femoral_head/' + img_name + '.jpg', img_crop)
            
            
            elif box_cls[0] == 1:
               left_x = int(box_xyxy[0])
               left_y = int(box_xyxy[1])
               right_x = int(box_xyxy[2])
               right_y = int(box_xyxy[3])
               
               if not os.path.exists(output_path + img_name + '/right_femoral_head'):
                   os.makedirs(output_path + img_name + '/right_femoral_head')
            
               left_y, right_y, left_x, right_x = bbox_refinement(left_y, right_y, left_x, right_x, ori_h, ori_w) 
               #print(right_y - left_y)
               #print(right_x - left_x)
               #print(left_y, right_y, left_x, right_x)
               img_crop = img_origin[left_y: right_y, left_x: right_x]
               #print(img_crop.shape)
               #print(left_y, right_y, left_x, right_x)
               with open(output_path + img_name + '/right_femoral_head/roi_coordinate.txt', 'w') as f1:
                   f1.write(str(int(left_x)))
                   f1.write(",")
                   f1.write(str(int(left_y)))
                   f1.write(",")
                   f1.write(str(int(right_x)))
                   f1.write(",")
                   f1.write(str(int(right_y)))
               
               cv2.imwrite(output_path + img_name + '/right_femoral_head/' + img_name + '.jpg', img_crop)
            
            elif box_cls[0] == 2:
               left_x = int(box_xyxy[0])
               left_y = int(box_xyxy[1])
               right_x = int(box_xyxy[2])
               right_y = int(box_xyxy[3])
               
               if not os.path.exists(output_path + img_name + '/left_pubic_bone'):
                   os.makedirs(output_path + img_name + '/left_pubic_bone')
                   
               left_y, right_y, left_x, right_x = bbox_refinement(left_y, right_y, left_x, right_x, ori_h, ori_w) 
               img_crop = img_origin[left_y: right_y, left_x: right_x]
               
               with open(output_path + img_name + '/left_pubic_bone/roi_coordinate.txt', 'w') as f1:
                   f1.write(str(int(left_x)))
                   f1.write(",")
                   f1.write(str(int(left_y)))
                   f1.write(",")
                   f1.write(str(int(right_x)))
                   f1.write(",")
                   f1.write(str(int(right_y)))
               
               cv2.imwrite(output_path + img_name + '/left_pubic_bone/' + img_name + '.jpg', img_crop)
            
            elif box_cls[0] == 3:
               left_x = int(box_xyxy[0])
               left_y = int(box_xyxy[1])
               right_x = int(box_xyxy[2])
               right_y = int(box_xyxy[3])
               
               if not os.path.exists(output_path + img_name + '/right_pubic_bone'):
                   os.makedirs(output_path + img_name + '/right_pubic_bone')
               
               left_y, right_y, left_x, right_x = bbox_refinement(left_y, right_y, left_x, right_x, ori_h, ori_w)
               img_crop = img_origin[left_y: right_y, left_x: right_x]
               
               with open(output_path + img_name + '/right_pubic_bone/roi_coordinate.txt', 'w') as f1:
                   f1.write(str(int(left_x)))
                   f1.write(",")
                   f1.write(str(int(left_y)))
                   f1.write(",")
                   f1.write(str(int(right_x)))
                   f1.write(",")
                   f1.write(str(int(right_y)))
               
               cv2.imwrite(output_path + img_name + '/right_pubic_bone/' + img_name + '.jpg', img_crop)
               
            cv2.imwrite(output_path + img_name + '/' + img_name + '_origin.jpg', img_origin)