import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pickle

import matplotlib.pyplot as plt 

def read_pickle_files(data_path):
    with open(data_path, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


def image_visualization(tensor):
    CAM_FRONT_LEFT = (tensor[0].permute(1,2,0) * 0.5 + 0.5).cpu().numpy()
    CAM_FRONT = (tensor[1].permute(1,2,0) * 0.5 + 0.5).cpu().numpy()
    CAM_FRONT_RIGHT = (tensor[2].permute(1,2,0) * 0.5 + 0.5).cpu().numpy()
    CAM_BACK_RIGHT = (tensor[3].permute(1,2,0) * 0.5 + 0.5).cpu().numpy()
    CAM_BACK = (tensor[4].permute(1,2,0) * 0.5 + 0.5).cpu().numpy()
    CAM_BACK_LEFT = (tensor[5].permute(1,2,0) * 0.5 + 0.5).cpu().numpy()

    plt.subplot(2,3,1)
    plt.axis("off")
    plt.imshow(CAM_FRONT_LEFT)
    plt.subplot(2,3,2)
    plt.axis("off")
    plt.imshow(CAM_FRONT)
    plt.subplot(2,3,3)
    plt.axis("off")
    plt.imshow(CAM_FRONT_RIGHT)
    plt.subplot(2,3,4)
    plt.axis("off")
    plt.imshow(CAM_BACK_LEFT)
    plt.subplot(2,3,5)
    plt.axis("off")
    plt.imshow(CAM_BACK)
    plt.subplot(2,3,6)
    plt.axis("off")
    plt.imshow(CAM_BACK_RIGHT)
    plt.savefig("image.png")

if __name__=="__main__":

    # Six Views: 
    # id =0 : "CAM_FRONT_LEFT"
    # id =1 : "CAM_FRONT"
    # id =2 : "CAM_FRONT_RIGHT"
    # id =3 : "CAM_BACK_RIGHT"
    # id =4 :  "CAM_BACK"
    # id =5 ： "CAM_BACK_LEFT"

    example_nuscene_data_sample_path = "/home/Zihua/DEV/MagicDrive/playrgound/data_example.pkl"
    # already normalized as -1, 1
    loaded_data_dict_sample = read_pickle_files(example_nuscene_data_sample_path)

    print(loaded_data_dict_sample.keys())
    # dict_keys(['img', 
    #               'gt_bboxes_3d', 
    #               'gt_labels_3d', 
    #               'gt_masks_bev', 
    #               'camera_intrinsics', 
    #                       'lidar2ego', 
    #               'lidar2camera', 
    #               'camera2lidar', 
    #               'lidar2image', 
    #                'img_aug_matrix', 
    #               'metas'])


    '''Image for Six Views'''
    # image_visualization(loaded_data_dict_sample['img'].data) #[6,H,W] for nuScence Six Views

    ''' Loaded bounding box in the LiDAR coordiante. '''
    # here the gt_bboxes_3d is a LiDARInstance3DBoxes List: LiDAR 坐标系下的 3D bounding boxes。
    # tensor is 9 dimensoion , the first 7 is :
    # [x, y, z, w, l, h, yaw]  yaw → 物体的 航向角（yaw rotation）

    # print(loaded_data_dict_sample['gt_bboxes_3d'].data[0].tensor[:,:7]) # the original data
    # print(loaded_data_dict_sample['gt_bboxes_3d'].data[0].center)       # the x,y,z
    # print(loaded_data_dict_sample['gt_bboxes_3d'].data[0].dims) # the w,l,h
    # print(loaded_data_dict_sample['gt_bboxes_3d'].data[0].yaw) # the yaw

    '''Loading the lidar2cam or lidar2img matrix'''

    print(loaded_data_dict_sample['lidar2image'].data.shape) #[6,4,4]

    print(loaded_data_dict_sample['lidar2camera'].data.shape) #[6,4,4]


    