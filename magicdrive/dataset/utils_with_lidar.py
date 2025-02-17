from typing import Tuple, List
from functools import partial
import random

import torch
import numpy as np

from transformers import CLIPTokenizer
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from ..runner.utils import trans_boxes_to_views

import copy
import numpy as np

import copy

def remove_points_inside_bboxes(lidar_points, bounding_boxes):
    
    geometry_lidar_points = lidar_points[:,:3]
    """
    Removes all LiDAR points inside any 3D bounding box.

    Args:
        lidar_points (torch.Tensor): (N1, 3) LiDAR point cloud (x, y, z).
        bounding_boxes (torch.Tensor): (N2, 8, 3) Bounding box corners.

    Returns:
        torch.Tensor: Filtered LiDAR points (only points outside the bounding boxes).
    """
    def is_inside_bbox(points, bbox):
        """
        Check if multiple 3D points are inside a given 3D bounding box.

        Args:
            points (torch.Tensor): (N1, 3) LiDAR points.
            bbox (torch.Tensor): (8, 3) Bounding box corners.

        Returns:
            torch.Tensor: (N1,) Boolean tensor (True if point is inside).
        """
        # Compute the bounding box center
        bbox_center = bbox.mean(dim=0)  # (3,)

        # Get the three main axes of the bounding box
        x_axis = bbox[1] - bbox[0]  # x1 - x0
        y_axis = bbox[3] - bbox[0]  # y1 - y0
        z_axis = bbox[4] - bbox[0]  # z1 - z0

        # Compute the half-size of the bounding box along each axis
        half_sizes = torch.tensor([
            x_axis.norm() / 2,
            y_axis.norm() / 2,
            z_axis.norm() / 2
        ], device=geometry_lidar_points.device)  # (3,)

        # Compute the transformation matrix (box to world)
        axes_matrix = torch.stack([x_axis, y_axis, z_axis], dim=1)  # (3,3)

        # Transform the points to bounding box local coordinates
        local_points = torch.linalg.inv(axes_matrix) @ (points - bbox_center).T  # (3, N1)
        local_points = local_points.T  # (N1, 3)

        # Check if the points are inside the box
        return (local_points.abs() <= half_sizes).all(dim=1)  # (N1,)

    # Create a mask to track points inside any bounding box
    mask = torch.ones(len(geometry_lidar_points), dtype=torch.bool, device=geometry_lidar_points.device)  # All points start as valid

    # Iterate through bounding boxes and remove points inside
    for bbox in bounding_boxes:
        inside_mask = is_inside_bbox(geometry_lidar_points, bbox)  # (N1,)
        mask &= ~inside_mask  # Remove points inside the bounding box

    return lidar_points[mask]  # Return only points outside the bounding boxes

def trans_points_to_view_multi_cam(points, trans_matrix, img_aug_matrix=None, proj=True):
    """
    Transforms LiDAR points to multiple camera views.

    Args:
        points (np.array): (N, 3) LiDAR points.
        trans_matrix (np.array): (6, 4, 4) transformation matrices for 6 cameras.
        img_aug_matrix (np.array, optional): (6, 4, 4) augmentation matrices. Defaults to None.
        proj (bool, optional): Whether to apply perspective projection. Defaults to True.

    Returns:
        list[np.array]: List of transformed points for each camera, each with shape (N, 3).
    """
    if len(points) == 0:
        return None

    num_cameras = trans_matrix.shape[0]  # 6 cameras
    transformed_points_list = []

    # travsal among the six vews
    for cam_id in range(num_cameras):
        # transmissition matirx
        trans = copy.deepcopy(trans_matrix[cam_id]).reshape(4, 4)

        # dara augmentat
        if img_aug_matrix is not None:
            img_aug = copy.deepcopy(img_aug_matrix[cam_id]).reshape(4, 4)
            trans = img_aug @ trans  # 应用增强变换

        # 变换点云
        coords = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)  # 齐次坐标
        coords = coords @ trans.T  # 4x4 变换

        # 透视投影
        if proj:
            z = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)  # 避免 z=0
            coords[:, 0] /= z
            coords[:, 1] /= z
            coords[:, 2] /= np.abs(coords[:, 2])  # 保持 z 轴正负号

        transformed_points_list.append(coords[:, :3])  # 只返回 (N,3)

    return transformed_points_list


# def get_points_inside_bboxes(lidar_points, bounding_boxes, num_samples=100):
#     geometry_lidar_points = lidar_points[:, :3]  # 仅保留 (x, y, z)

#     def is_inside_bbox(points, bbox):
#         bbox_center = bbox.mean(dim=0)  # (3,)
#         x_axis = bbox[1] - bbox[0]  
#         y_axis = bbox[3] - bbox[0]  
#         z_axis = bbox[4] - bbox[0]  

#         half_sizes = torch.tensor([
#             x_axis.norm() / 2,
#             y_axis.norm() / 2,
#             z_axis.norm() / 2
#         ], device=geometry_lidar_points.device)  

#         axes_matrix = torch.stack([x_axis, y_axis, z_axis], dim=1)  

#         local_points = torch.linalg.inv(axes_matrix) @ (points - bbox_center).T  
#         local_points = local_points.T  

#         return (local_points.abs() <= half_sizes).all(dim=1)  

#     selected_points = []

#     for bbox in bounding_boxes:
#         inside_mask = is_inside_bbox(geometry_lidar_points, bbox)
#         inside_points = geometry_lidar_points[inside_mask]  # (M, 3)

#         if len(inside_points) == 0:
#             # 没有点，使用 bbox 中心点
#             bbox_center = bbox.mean(dim=0, keepdim=True)  # (1, 3)
#             sampled_points = bbox_center.repeat(num_samples, 1)  # 复制 num_samples 次
#         elif len(inside_points) >= num_samples:
#             # 随机采样 num_samples 个点
#             indices = torch.randperm(len(inside_points))[:num_samples]
#             sampled_points = inside_points[indices]
#         else:
#             # 点数不够，重复采样到 num_samples 个
#             repeat_factor = (num_samples + len(inside_points) - 1) // len(inside_points)  
#             sampled_points = inside_points.repeat(repeat_factor, 1)[:num_samples]  

#         selected_points.append(sampled_points.unsqueeze(0))  # 变成 (1, num_samples, 3)

#     return torch.cat(selected_points, dim=0) if selected_points else torch.empty((0, num_samples, 3), device=lidar_points.device)

def get_points_inside_bboxes(lidar_points, bounding_boxes, num_samples=100, relative=True):
    geometry_lidar_points = lidar_points[:, :3]  # 仅保留 (x, y, z)

    def is_inside_bbox(points, bbox):
        bbox_center = bbox.mean(dim=0)  # (3,)
        x_axis = bbox[1] - bbox[0]  
        y_axis = bbox[3] - bbox[0]  
        z_axis = bbox[4] - bbox[0]  

        half_sizes = torch.tensor([
            x_axis.norm() / 2,
            y_axis.norm() / 2,
            z_axis.norm() / 2
        ], device=geometry_lidar_points.device)  

        axes_matrix = torch.stack([x_axis, y_axis, z_axis], dim=1)  

        local_points = torch.linalg.inv(axes_matrix) @ (points - bbox_center).T  
        local_points = local_points.T  

        return (local_points.abs() <= half_sizes).all(dim=1), bbox_center

    selected_points = []

    for bbox in bounding_boxes:
        inside_mask, bbox_center = is_inside_bbox(geometry_lidar_points, bbox)
        inside_points = geometry_lidar_points[inside_mask]  # (M, 3)

        if len(inside_points) == 0:
            # 没有点，使用 bbox 中心点
            sampled_points = bbox_center.repeat(num_samples, 1)  # (num_samples, 3)
        elif len(inside_points) >= num_samples:
            # 随机采样 num_samples 个点
            indices = torch.randperm(len(inside_points))[:num_samples]
            sampled_points = inside_points[indices]
        else:
            # 点数不够，重复采样到 num_samples 个
            repeat_factor = (num_samples + len(inside_points) - 1) // len(inside_points)  
            sampled_points = inside_points.repeat(repeat_factor, 1)[:num_samples]  

        # 如果 relative=True，把坐标转换成相对 bbox_center
        if relative:
            sampled_points -= bbox_center

        selected_points.append(sampled_points.unsqueeze(0))  # (1, num_samples, 3)

    return torch.cat(selected_points, dim=0) if selected_points else torch.empty((0, num_samples, 3), device=lidar_points.device)



def ensure_positive_z_for_lidar(coords):
    """
    Returns a mask indicating which points have z > 0.

    Args:
        coords (torch.Tensor): (N, 3) point cloud.

    Returns:
        torch.Tensor: (N,) boolean mask where z > 0.
    """
    return coords[:, 2] > 0  # Mask for each point where z > 0



META_KEY_LIST = [
    "gt_bboxes_3d",
    "gt_labels_3d",
    "camera_intrinsics",
    "camera2ego",
    "lidar2ego",
    "lidar2camera",
    "camera2lidar",
    "lidar2image",
    "img_aug_matrix",
    "metas",
]

def _tokenize_captions(examples, template, tokenizer=None, is_train=True):
    captions = []
    for example in examples:
        caption = template.format(**example["metas"].data)
        captions.append(caption)
    captions.append("")
    if tokenizer is None:
        return None, captions

    # pad in the collate_fn function
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="do_not_pad",
        truncation=True,
    )
    input_ids = inputs.input_ids
    # pad to the longest of current batch (might differ between cards)
    padded_tokens = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    ).input_ids
    return padded_tokens, captions

def ensure_canvas(coords, canvas_size: Tuple[int, int]):
    """Box with any point in range of canvas should be kept.

    Args:
        coords (_type_): _description_
        canvas_size (Tuple[int, int]): _description_

    Returns:
        np.array: mask on first axis.
    """
    (h, w) = canvas_size
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    w_mask = np.any(np.logical_and(
        coords[..., 0] > 0, coords[..., 0] < w), axis=1)
    h_mask = np.any(np.logical_and(
        coords[..., 1] > 0, coords[..., 1] < h), axis=1)
    c_mask = np.logical_and(c_mask, np.logical_and(w_mask, h_mask))
    return c_mask

def ensure_positive_z(coords):
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    return c_mask

def random_0_to_1(mask: np.array, num):
    assert mask.ndim == 1
    inds = np.where(mask == 0)[0].tolist()
    random.shuffle(inds)
    mask = np.copy(mask)
    mask[inds[:num]] = 1
    return mask

def _transform_all(examples, matrix_key, proj):
    """project all bbox to views, return 2d coordinates.

    Args:
        examples (List): collate_fn input.

    Returns:
        2-d list: List[List[np.array]] for B, N_cam. Can be [].
    """
    
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    # lidar2image (np.array): lidar to image view transformation
    
    trans_matrix = np.stack([example[matrix_key].data.numpy()
                            for example in examples], axis=0)
    
    # img_aug_matrix (np.array): augmentation matrix
    img_aug_matrix = np.stack([example['img_aug_matrix'].data.numpy()
                               for example in examples], axis=0)  # 图像增强矩阵（B, N_cam, 4, 4），用于处理数据增强（如缩放、旋转）
    B, N_cam = trans_matrix.shape[:2]

    bboxes_coord = []
    # for each keyframe set
    for idx in range(B):
        # if zero, add empty list
        if len(gt_bboxes_3d[idx]) == 0:
            # keep N_cam dim for convenient
            bboxes_coord.append([None for _ in range(N_cam)])
            continue

        coords_list = trans_boxes_to_views(
            gt_bboxes_3d[idx], trans_matrix[idx], img_aug_matrix[idx], proj)
        bboxes_coord.append(coords_list)
    return bboxes_coord


def _transform_all_with_lidar(examples, matrix_key, proj,gt_lidar_list=None):
    """project all bbox to views, return 2d coordinates.

    Args:
        examples (List): collate_fn input.

    Returns:
        2-d list: List[List[np.array]] for B, N_cam. Can be [].
    """
    
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    # lidar2image (np.array): lidar to image view transformation
    
    if gt_lidar_list is not None:
        gt_lidars = gt_lidar_list
    
    
    trans_matrix = np.stack([example[matrix_key].data.numpy()
                            for example in examples], axis=0)
    
    # img_aug_matrix (np.array): augmentation matrix
    img_aug_matrix = np.stack([example['img_aug_matrix'].data.numpy()
                               for example in examples], axis=0)  # 图像增强矩阵（B, N_cam, 4, 4），用于处理数据增强（如缩放、旋转）
    B, N_cam = trans_matrix.shape[:2]
    


    bboxes_coord = []
    lidar_coord = [] # six view lidars
    
    # for each keyframe set
    for idx in range(B):
        # if zero, add empty list
        if len(gt_bboxes_3d[idx]) == 0:
            # keep N_cam dim for convenient
            bboxes_coord.append([None for _ in range(N_cam)])
            lidar_coord.append([None for _ in range(N_cam)])
            continue

        coords_list = trans_boxes_to_views(
            gt_bboxes_3d[idx], trans_matrix[idx], img_aug_matrix[idx], proj)
        
        lidar_coord_list = trans_points_to_view_multi_cam(
            gt_lidars[idx][:,:3],trans_matrix[idx], img_aug_matrix[idx], proj
        )
        lidar_coord.append(lidar_coord_list)
        bboxes_coord.append(coords_list)

    return bboxes_coord,lidar_coord



def _preprocess_bbox(bbox_mode, canvas_size, examples, is_train=True,
                     view_shared=False, use_3d_filter=True, bbox_add_ratio=0,
                     bbox_add_num=0, bbox_drop_ratio=0):
    """Pre-processing for bbox
    .. code-block:: none

                                       up z
                        front x           ^
                             /            |
                            /             |
              (x1, y0, z1) + -----------  + (x1, y1, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y<-------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)

    Args:
        bbox_mode (str): type of bbox raw data.
            cxyz -> x1y1z1, x1y0z1, x1y1z0, x0y1z1;
            all-xyz -> all 8 corners xyz;
            owhr -> center, l, w, h, z-orientation.
        canvas_size (2-tuple): H, W of input images
        examples: collate_fn input
        view_shared: if enabled, all views share same set of bbox and output
            N_cam=1; otherwise, use projection to keep only visible bboxes.
    Return:
        in form of dict:
            bboxes (Tensor): B, N_cam, max_len, ...
            classes (LongTensor): B, N_cam, max_len
            masks: 1 for data, 0 for padding
    """
    # init data
    bboxes = []
    classes = []
    max_len = 0
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    gt_labels_3d: List[torch.Tensor] = [
        example["gt_labels_3d"].data for example in examples]

    # params
    B = len(gt_bboxes_3d) # here the B is the batch size
    N_cam = len(examples[0]['lidar2image'].data.numpy()) # the number of ligar2
    N_out = 1 if view_shared else N_cam # output is also the 6

    bboxes_coord = None
    # used 3d filter: means ensure the z>0 and make sure the projected bbox inside the canvas_size
    # without using 3d filter: use_3d_filter=False：使用 2D 过滤（确保 bbox 投影在 canvas_size 内）。
    if not view_shared and not use_3d_filter:
        bboxes_coord = _transform_all(examples, 'lidar2image', True) # 将 bbox 投影到图像坐标系（用于 2D 过滤）。True
    elif not view_shared:
        bboxes_coord_3d = _transform_all(examples, 'lidar2camera', False) #将 bbox 投影到相机坐标系（用于 3D 过滤）。

    # for each keyframe set
    for idx in range(B):
        bboxes_kf = gt_bboxes_3d[idx] # LiDARInstance3DBoxes
        classes_kf = gt_labels_3d[idx] # Class

        # if zero, add zero length tensor (for padding).
        if len(bboxes_kf) == 0 or (
                random.random() < bbox_drop_ratio and is_train):
            bboxes.append([None] * N_out)
            classes.append([None] * N_out)
            continue

        # whether share the boxes across views, filtered by 2d projection.
        if not view_shared:
            index_list = []  # each view has a mask
            lidar_index_list = []
            if use_3d_filter:
                coords_list = bboxes_coord_3d[idx]
            
                filter_func = ensure_positive_z
            else:
                # filter bbox according to 2d projection on image canvas
                coords_list = bboxes_coord[idx]
                # judge coord by cancas_size
                filter_func = partial(ensure_canvas, canvas_size=canvas_size)
            # we do not need to handle None since we already filter for len=0
            for coords in coords_list:
                c_mask = filter_func(coords)
                if random.random() < bbox_add_ratio and is_train:
                    c_mask = random_0_to_1(c_mask, bbox_add_num)
                index_list.append(c_mask)
                
                max_len = max(max_len, c_mask.sum())
        else:
            # we use as mask, torch.bool is important
            index_list = [torch.ones(len(bboxes_kf), dtype=torch.bool)]
            max_len = max(max_len, len(bboxes_kf))

        # construct data
        if bbox_mode == 'cxyz':
            # x1y1z1, x1y0z1, x1y1z0, x0y1z1
            bboxes_pt = bboxes_kf.corners[:, [6, 5, 7, 2]]
        elif bbox_mode == 'all-xyz':
            bboxes_pt = bboxes_kf.corners  # n x 8 x 3
        elif bbox_mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {bbox_mode}")
        bboxes.append([bboxes_pt[ind] for ind in index_list])
        classes.append([classes_kf[ind] for ind in index_list])
        bbox_shape = bboxes_pt.shape[1:]

    # there is no (visible) boxes in this batch
    if max_len == 0:
        return None, None

    # pad and construct mask
    # `bbox_shape` should be set correctly
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape)
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.long)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)
    for _b in range(B):
        _bboxes = bboxes[_b]
        _classes = classes[_b]
        for _n in range(N_out):
            if _bboxes[_n] is None:
                continue  # empty for this batch
            this_box_num = len(_bboxes[_n])
            ret_bboxes[_b, _n, :this_box_num] = _bboxes[_n]
            ret_classes[_b, _n, :this_box_num] = _classes[_n]
            ret_masks[_b, _n, :this_box_num] = True

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict, bboxes_coord



def _preprocess_bbox_and_the_lidar_points(bbox_mode, canvas_size, 
                                          examples, is_train=True,view_shared=False, 
                                          use_3d_filter=True, bbox_add_ratio=0,
                     bbox_add_num=0, bbox_drop_ratio=0):
    """Pre-processing for bbox
    .. code-block:: none

                                       up z
                        front x           ^
                             /            |
                            /             |
              (x1, y0, z1) + -----------  + (x1, y1, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y<-------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)

    Args:
        bbox_mode (str): type of bbox raw data.
            cxyz -> x1y1z1, x1y0z1, x1y1z0, x0y1z1;
            all-xyz -> all 8 corners xyz;
            owhr -> center, l, w, h, z-orientation.
        canvas_size (2-tuple): H, W of input images
        examples: collate_fn input
        view_shared: if enabled, all views share same set of bbox and output
            N_cam=1; otherwise, use projection to keep only visible bboxes.
    Return:
        in form of dict:
            bboxes (Tensor): B, N_cam, max_len, ...
            classes (LongTensor): B, N_cam, max_len
            masks: 1 for data, 0 for padding
    """
    # init data
    bboxes = []
    classes = []
    lidares = []
    
    max_len = 0
    max_len_for_lidar = 0
    
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    gt_labels_3d: List[torch.Tensor] = [
        example["gt_labels_3d"].data for example in examples]

    if not gt_bboxes_3d or len(gt_bboxes_3d) == 0:
        print("Warning: gt_bboxes_3d is empty!")
    

    # filered out pints inside the bounding boxes
    gt_lidar_for_each_instance_list = []
    for batch_size_id in range(len(gt_bboxes_3d)):
        # how many bounding box here
        if len(gt_bboxes_3d[batch_size_id].tensor)==0:
            kepted_lidars = None
            gt_lidar_for_each_instance_list.append(kepted_lidars)
        else:
            current_gt_boxes_3d_in_this_scene = gt_bboxes_3d[batch_size_id].corners #(N,8,3)
            current_lidar_in_this_scene = examples[batch_size_id]['points'].data
            kepted_lidars = get_points_inside_bboxes(current_lidar_in_this_scene,current_gt_boxes_3d_in_this_scene,
                                                    relative=True)
            assert kepted_lidars.shape[0] == current_gt_boxes_3d_in_this_scene.shape[0]
            # if kepted_lidars.shape[0]==0:
            #     kepted_lidars = torch.zeros(10,5).type_as(current_gt_boxes_3d_in_this_scene)
            gt_lidar_for_each_instance_list.append(kepted_lidars)
        

    # process all the LiDARs point cloud.
    # gt_instance_lidar_list = [
    #     Sample_LiDAR_With_A_CertainedNum(example['points'].data) for example in examples]

    gt_instance_lidar_list =gt_lidar_for_each_instance_list

    # gt_instance_lidar_list = gt_lidar_for_each_instance_list
    
    # params
    B = len(gt_bboxes_3d) # here the B is the batch size
    N_cam = len(examples[0]['lidar2image'].data.numpy()) # the number of ligar2
    N_out = 1 if view_shared else N_cam # output is also the 6

    bboxes_coord = None
    # used 3d filter: means ensure the z>0 and make sure the projected bbox inside the canvas_size
    # without using 3d filter: use_3d_filter=False：使用 2D 过滤（确保 bbox 投影在 canvas_size 内）。
    if not view_shared and not use_3d_filter:
        bboxes_coord = _transform_all(examples, 'lidar2image', True) # 将 bbox 投影到图像坐标系（用于 2D 过滤）。True
    elif not view_shared:
        bboxes_coord_3d = _transform_all(examples, 'lidar2camera', False)
        #bboxes_coord_3d,lidar_coord_3d = _transform_all_with_lidar(examples, 'lidar2camera', False,gt_instance_lidar_list) #将 bbox 投影到相机坐标系（用于 3D 过滤）。# here is the batch size length list
        # assert len(bboxes_coord_3d) == len(lidar_coord_3d), f"Mismatch: {len(bboxes_coord_3d)} vs {len(lidar_coord_3d)}"
    
    # do the processing here

    # for each keyframe set
    for idx in range(B):
        bboxes_kf = gt_bboxes_3d[idx] # LiDARInstance3DBoxes at the current batch 
        classes_kf = gt_labels_3d[idx] # Class
        
        lidar_points_kf = gt_instance_lidar_list[idx] # LiDAR coordinate--->[N,100,3]

        # assert len(lidar_points_kf) ==len(bboxes_kf)

        # print("points inside objects: ",lidar_points_kf.shape)
        # if zero, add zero length tensor (for padding).
        if len(bboxes_kf) == 0 or (
                random.random() < bbox_drop_ratio and is_train):
            bboxes.append([None] * N_out)
            classes.append([None] * N_out)
            lidares.append([None] * N_out)
            
            # # remove the LiDAR Inside the bounding boxes
            # if len(bboxes_kf)!=0:
            #     lidar_points_kf = remove_points_inside_bboxes(lidar_points_kf, bboxes_kf.corners)
            continue

        # whether share the boxes across views, filtered by 2d projection.
        if not view_shared:
            index_list = []  # each view has a mask

            if use_3d_filter:
                coords_list = bboxes_coord_3d[idx] # current batch six views
                # lidar_coords_list = lidar_coord_3d[idx]
                filter_func = ensure_positive_z
                # filter_func_for_lidar = ensure_positive_z_for_lidar
            else:
                # filter bbox according to 2d projection on image canvas
                coords_list = bboxes_coord[idx]
                # judge coord by cancas_size
                filter_func = partial(ensure_canvas, canvas_size=canvas_size)
            
            # travsaal amopng six views
            # we do not need to handle None since we already filter for len=0
            # assert len(coords_list) == len(lidar_coords_list)

            for sub_idx, coords in enumerate(coords_list):
                c_mask = filter_func(coords)

                # c_mask_for_lidar = filter_func_for_lidar(lidar_coords_list[sub_idx])
                
                # data augmentation
                if random.random() < bbox_add_ratio and is_train:
                    c_mask = random_0_to_1(c_mask, bbox_add_num)
                
                index_list.append(c_mask)
                # index_lidar_list.append(c_mask_for_lidar)
                
                max_len = max(max_len, c_mask.sum())
                # max_len_for_lidar = max(max_len_for_lidar,c_mask_for_lidar.sum())
                
                
        else:
            # we use as mask, torch.bool is important
            index_list = [torch.ones(len(bboxes_kf), dtype=torch.bool)]
            max_len = max(max_len, len(bboxes_kf))

        # construct data
        if bbox_mode == 'cxyz':
            # x1y1z1, x1y0z1, x1y1z0, x0y1z1
            bboxes_pt = bboxes_kf.corners[:, [6, 5, 7, 2]]
        elif bbox_mode == 'all-xyz':
            bboxes_pt = bboxes_kf.corners  # n x 8 x 3
        elif bbox_mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {bbox_mode}")
        
        bboxes.append([bboxes_pt[ind] for ind in index_list])
        classes.append([classes_kf[ind] for ind in index_list])
        lidares.append([lidar_points_kf[ind] for ind in index_list])
        
        bbox_shape = bboxes_pt.shape[1:]
        instance_lidar_shape = lidar_points_kf.shape[1:]

    # there is no (visible) boxes in this batch
    if max_len == 0:
        return None, None , None
    
    # ret_lidares = torch.zeros(B, N_out, max_len_for_lidar,5,dtype=torch.float32)
    # ret_lidar_masks = torch.zeros(B, N_out, max_len_for_lidar, dtype=torch.bool)

    ret_lidares = torch.zeros(B, N_out, max_len,*instance_lidar_shape)
    ret_lidar_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)

    # pad and construct mask
    # `bbox_shape` should be set correctly
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape)
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.long)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)
    
    for _b in range(B):
        _bboxes = bboxes[_b]
        _classes = classes[_b] #(6,N,8,3)        
        _lidars = lidares[_b] #(6,N,200,3)



        for _n in range(N_out):
            if _bboxes[_n] is None:
                continue  # empty for this batch
            
            this_box_num = len(_bboxes[_n])    
            this_lidar_num = len(_lidars[_n])
        
            ret_lidares[_b,_n,:this_lidar_num]= _lidars[_n]
            ret_lidar_masks[_b,_n,:this_lidar_num] = True
            
            ret_bboxes[_b, _n, :this_box_num] = _bboxes[_n]
            ret_classes[_b, _n, :this_box_num] = _classes[_n]
            ret_masks[_b, _n, :this_box_num] = True
            
            
            
    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,
        "classes": ret_classes,
        "masks": ret_masks,
    }
    ret_dict_lidar = {
        "lidars": ret_lidares,
        "masks": ret_lidar_masks
    }
    
    
    return ret_dict, bboxes_coord, ret_dict_lidar




def Sample_LiDAR_With_A_CertainedNum(tensor, num_samples=30000):
    N = tensor.shape[0]
    num_samples = min(N, num_samples)  # 确保不会超过 N
    indices = torch.randperm(N)[:num_samples]  # 生成随机索引
    return tensor[indices]  


def collate_fn_with_lidar(
    examples: Tuple[dict, ...],
    template: str,
    tokenizer: CLIPTokenizer = None,
    is_train: bool = True,
    bbox_mode: str = None,
    bbox_view_shared: bool = False,
    bbox_drop_ratio: float = 0,
    bbox_add_ratio: float = 0,
    bbox_add_num: int = 3,
):
    """
    We need to handle:
    1. make multi-view images (img) into tensor -> [N, 6, 3, H, W]
    2. make masks (gt_masks_bev, gt_aux_bev) into tensor
        -> [N, 25 = 8 map + 10 obj + 7 aux, 200, 200]
    3. make caption (location, desctiption, timeofday) and tokenize, padding
        -> [N, pad_length]
    4. extract camera parameters (camera_intrinsics, camera2lidar)
        camera2lidar: A @ v_camera = v_lidar
        -> [N, 6, 3, 7]
    We keep other meta data as original.
    """
    # here the samples is a list of return for the dataloder

    if "points" in examples[0].keys():
        use_lidar = True
    else:
        use_lidar = False
    

    if bbox_add_ratio > 0 and is_train:
        assert bbox_view_shared == False, "You cannot add any box on view shared."

    # mask: to make sure whether "gt_aux_bev in the examples"
    if "gt_aux_bev" in examples[0] and examples[0]["gt_aux_bev"] is not None:
        keys = ["gt_masks_bev", "gt_aux_bev"] # add keys
        assert bbox_drop_ratio == 0, "map is not affected in bbox_drop"
    else:
        keys = ["gt_masks_bev"]
    
    #  处理 BEV 语义图（Bird’s Eye View Mask）
    # fmt: off
    bev_map_with_aux = torch.stack([torch.from_numpy(np.concatenate([
        example[key] for key in keys  # np array, channel-last
    ], axis=0)).float() for example in examples], dim=0)  # float32
    # fmt: on

    # camera param
    # TODO: camera2lidar should be changed to lidar2camera
    # fmt: off
    # 处理相机参数
    camera_param = torch.stack([torch.cat([
        example["camera_intrinsics"].data[:, :3, :3],  # 3x3 is enough
        example["camera2lidar"].data[:, :3],  # only first 3 rows meaningful
    ], dim=-1) for example in examples], dim=0)
    # fmt: on

    ret_dict = {
        "bev_map_with_aux": bev_map_with_aux,
        "camera_param": camera_param,
        "kwargs": {},
    }
    
    # make sure in images
    if "img" in examples[0]:
        # multi-view images
        pixel_values = torch.stack(
            [example["img"].data for example in examples])
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format).float()
        ret_dict["pixel_values"] = pixel_values  #[N, 6, 3, H, W]
    elif is_train:
        raise RuntimeError("For training, you should provide gt images.")

    # bboxes_3d, convert to tensor
    # here we consider:
    # 1. do we need to filter bboxes for each view? use `view_shared`
    # 2. padding for one batch of data if need (with zero), and output mask.
    # 3. what is the expected output format? dict of kwargs to bbox embedder
    canvas_size = pixel_values.shape[-2:] #(H,w)
    
    
    if bbox_mode is not None:
        # NOTE: both can be None
        # 过滤掉超出画布的 bbox。
        # 进行 bbox 随机丢弃和增强（如果 bbox_drop_ratio 或 bbox_add_ratio > 0）。
        # 进行 bbox 格式转换（cxyz、all-xyz 等）。
        # 进行 视角共享（如果 bbox_view_shared=True，所有视角使用相同 bbox）。
        
        if use_lidar:
            bboxes_3d_input, bbox_view_coord, lidars_3d_input = _preprocess_bbox_and_the_lidar_points(
                bbox_mode, canvas_size, examples, is_train=is_train,
                view_shared=bbox_view_shared, bbox_add_ratio=bbox_add_ratio,
                bbox_add_num=bbox_add_num, bbox_drop_ratio=bbox_drop_ratio)
            
            ret_dict["kwargs"]["bboxes_3d_data"] = bboxes_3d_input
            ret_dict["kwargs"]["lidars_3d_data"] = lidars_3d_input
        else:
            bboxes_3d_input, bbox_view_coord = _preprocess_bbox(
            bbox_mode, canvas_size, examples, is_train=is_train,
            view_shared=bbox_view_shared, bbox_add_ratio=bbox_add_ratio,
            bbox_add_num=bbox_add_num, bbox_drop_ratio=bbox_drop_ratio)
            ret_dict["kwargs"]["bboxes_3d_data"] = bboxes_3d_input
            
    else:
        bbox_view_coord = None
        
    # 处理文本

    # captions: one real caption with one null caption
    input_ids_padded, captions = _tokenize_captions(
        examples, template, tokenizer, is_train)
    ret_dict["captions"] = captions[:-1]  # list of str
    if tokenizer is not None:
        # real captions in head; the last one is null caption
        # we omit "attention_mask": padded_tokens.attention_mask, seems useless
        ret_dict["input_ids"] = input_ids_padded[:-1]
        ret_dict["uncond_ids"] = input_ids_padded[-1:]

    # other meta data
    meta_list_dict = dict()
    for key in META_KEY_LIST:
        try:
            meta_list = [example[key] for example in examples]
            meta_list_dict[key] = meta_list
        except KeyError:
            continue
    ret_dict['meta_data'] = meta_list_dict

    return ret_dict
