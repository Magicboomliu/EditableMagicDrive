import torch
import numpy as np
from mmdet3d.ops.voxel import Voxelization 
import pickle
from mmdet3d.datasets.pipelines import Compose
from mmcv import Config
from mmdet3d.models import build_model
from torchpack.utils.config import configs
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class Voxelization(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_points_per_voxel, max_voxels):
        super(Voxelization, self).__init__()
        self.voxel_size = torch.tensor(voxel_size)
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels

    def forward(self, points):
        """
        :param points: 输入点云，形状为 [B, N, 3] (B 是 batch_size, N 是点数)
        :return: voxels: 体素化的点云，形状为 [B, M, T, 3] (M 是最大体素数, T 是每个体素中的点数)
                 voxel_coords: 体素坐标，形状为 [B, M, 3]
                 voxel_mask: 体素掩码，形状为 [B, M]，表示哪些体素是有效的
        """
        B, N, _ = points.shape
        voxels = []
        voxel_coords = []
        voxel_masks = []
        self.voxel_size = self.voxel_size.to(points.device)
        self.point_cloud_range = self.point_cloud_range.to(points.device)
        
        for b in range(B):
            # 获取当前 batch 的点云
            batch_points = points[b]

            # 计算体素网格大小
            grid_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
            grid_size = torch.round(grid_size).long()

            # 将点云映射到体素坐标
            shifted_coords = batch_points - self.point_cloud_range[:3]
            batch_voxel_coords = torch.floor(shifted_coords / self.voxel_size).long()

            # 过滤超出范围的点
            valid_mask = (batch_voxel_coords >= 0) & (batch_voxel_coords < grid_size)
            valid_mask = valid_mask.all(dim=-1)
            batch_points = batch_points[valid_mask]
            batch_voxel_coords = batch_voxel_coords[valid_mask]

            # 对体素进行唯一化并统计每个体素中的点数
            unique_voxel_coords, inverse_indices, counts = torch.unique(
                batch_voxel_coords, return_inverse=True, return_counts=True, dim=0)

            # 限制体素数量
            if unique_voxel_coords.shape[0] > self.max_voxels:
                unique_voxel_coords = unique_voxel_coords[:self.max_voxels]
                mask = inverse_indices < self.max_voxels
                batch_points = batch_points[mask]
                inverse_indices = inverse_indices[mask]

            # 将点分配到体素中
            batch_voxels = torch.zeros(
                (unique_voxel_coords.shape[0], self.max_points_per_voxel, 3),
                dtype=batch_points.dtype, device=batch_points.device)
            voxel_counts = torch.zeros(
                (unique_voxel_coords.shape[0],), dtype=torch.long, device=batch_points.device)

            for i, idx in enumerate(inverse_indices):
                if voxel_counts[idx] < self.max_points_per_voxel:
                    batch_voxels[idx, voxel_counts[idx]] = batch_points[i]
                    voxel_counts[idx] += 1

            voxels.append(batch_voxels)
            voxel_coords.append(unique_voxel_coords)
            voxel_masks.append(torch.ones(unique_voxel_coords.shape[0], dtype=torch.bool, device=batch_points.device))

        # 找到最大的体素数
        max_M = max([v.shape[0] for v in voxels])

        # 对 voxels 和 voxel_coords 进行填充
        padded_voxels = []
        padded_voxel_coords = []
        padded_voxel_masks = []

        for b in range(B):
            M, T, _ = voxels[b].shape
            if M < max_M:
                # 填充 voxels
                pad_size = max_M - M
                padded_voxel = torch.cat([
                    voxels[b],
                    torch.zeros((pad_size, T, 3), dtype=voxels[b].dtype, device=voxels[b].device)
                ], dim=0)
                padded_voxels.append(padded_voxel)

                # 填充 voxel_coords
                padded_voxel_coord = torch.cat([
                    voxel_coords[b],
                    torch.zeros((pad_size, 3), dtype=voxel_coords[b].dtype, device=voxel_coords[b].device)
                ], dim=0)
                padded_voxel_coords.append(padded_voxel_coord)

                # 填充 voxel_masks
                padded_voxel_mask = torch.cat([
                    voxel_masks[b],
                    torch.zeros((pad_size), dtype=torch.bool, device=voxel_masks[b].device)
                ], dim=0)
                padded_voxel_masks.append(padded_voxel_mask)
            else:
                padded_voxels.append(voxels[b])
                padded_voxel_coords.append(voxel_coords[b])
                padded_voxel_masks.append(voxel_masks[b])

        # 将列表转换为张量
        voxels = torch.stack(padded_voxels, dim=0)  # [B, max_M, T, 3]
        voxel_coords = torch.stack(padded_voxel_coords, dim=0)  # [B, max_M, 3]
        voxel_masks = torch.stack(padded_voxel_masks, dim=0)  # [B, max_M]

        return voxels, voxel_coords, voxel_masks

class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PillarFeatureNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels),
            nn.ReLU()
        )

    def forward(self, voxels, voxel_masks):
        """
        :param voxels: 体素化的点云，形状为 [B, M, T, 3]
        :param voxel_masks: 体素掩码，形状为 [B, M]，表示哪些体素是有效的
        :return: 特征，形状为 [B, M, out_channels]
        """
        B, M, T, _ = voxels.shape

        voxels = voxels.view(B * M, T, -1)  # 展平体素维度

        voxels_tensor = voxels.view(B*M,-1)
        
        features = self.mlp(voxels_tensor)  # 提取特征
        features = features.view(B, M, -1)

        features = features * voxel_masks.unsqueeze(-1)

        return features

class PointPillars(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_points_per_voxel, max_voxels, out_channels):
        super(PointPillars, self).__init__()
        self.voxelization = Voxelization(voxel_size, point_cloud_range, max_points_per_voxel, max_voxels)
        self.pillar_feature_net = PillarFeatureNet(in_channels=300, out_channels=out_channels)

    def forward(self, points):
        """
        :param points: 输入点云，形状为 [B, N, 3]
        :return: 特征，形状为 [B, M, out_channels]
        """
        features_list = []

        B,Num_of_CAMs,N,_ = points.shape
        for cam in range(Num_of_CAMs):
            voxels, voxel_coords, voxel_masks = self.voxelization(points[:,cam,:,:])  # 体素化
            features = self.pillar_feature_net(voxels, voxel_masks)  # 提取特征
            features_list.append(features)
        
        N_min = 1000000000000
        for feat in features_list:
            current_N = feat.shape[1]
            if current_N <=N_min:
                N_min = current_N
        for idx in range(len(features_list)):
            features_list[idx] = features_list[idx][:,:N_min,:].unsqueeze(1)
                 
        features = torch.cat(features_list,dim=1)
        return features


def read_the_pickle_files(filename):
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


def compute_percentile_mean(points, percentile=1):
    """
    计算点云 x, y, z 维度上最大和最小 1% 的均值
    :param points: shape [N, 3] 的点云数据
    :param percentile: 计算最大和最小的百分比 (默认为 1%)
    :return: (min_x, max_x, min_y, max_y, min_z, max_z) 6 个均值
    """
    N = points.shape[0]
    k = max(1, int(N * percentile / 100))  # 确保至少取一个点

    # 计算每个维度的最小 1% 和最大 1%
    min_x = points[:, 0].kthvalue(k, dim=0)[0].mean()
    max_x = points[:, 0].kthvalue(N - k + 1, dim=0)[0].mean()
    
    min_y = points[:, 1].kthvalue(k, dim=0)[0].mean()
    max_y = points[:, 1].kthvalue(N - k + 1, dim=0)[0].mean()
    
    min_z = points[:, 2].kthvalue(k, dim=0)[0].mean()
    max_z = points[:, 2].kthvalue(N - k + 1, dim=0)[0].mean()

    return min_x.item(), max_x.item(), min_y.item(), max_y.item(), min_z.item(), max_z.item()


if __name__=="__main__":
    
    
    LiDARs = read_the_pickle_files("lidar_3d_data.pkl")
    
    lidars =LiDARs['lidars'][...,:3]
    masks = LiDARs['masks']

    lidars = lidars[0:1,:,:,:]
    lidars_check = lidars.reshape(-1,3)
    
    x_min,x_max,y_min,y_max,z_min,z_max = compute_percentile_mean(lidars_check)


    voxel_size = [0.16, 0.16, 4.0]
    point_cloud_range = [-50, -50, -5, 50, 50, 3]
    max_points_per_voxel = 100
    max_voxels = 12000
    out_channels = 768

    # 初始化模型
    model = PointPillars(voxel_size, point_cloud_range, max_points_per_voxel, max_voxels, out_channels)
    model = model.cuda()
    feature = model(lidars.float())
    
    print(feature.shape)

    
    # print(features.shape)
    
    # x_range = (-30, 30)
    # y_range = (-30, 30)
    # grid_size = (200, 200)  # 100x100 的网格
    
    # pillarizer = Pillarization(x_range, y_range, grid_size)
    # output = pillarizer(lidars.float())
    # print(output.shape)
    
    
    # # [Batch_Size,NUMS_OF_CAMERRAS,dimensions, 768]
    
    # cfg_path = "configs/nuscenes/pointpillars_nuscenes.py"  # Modify based on your dataset
    # cfg = Config.fromfile(cfg_path)
    
    # # loaded the pretrained models
    # config_path = "/home/SonyInternship/MagicDrive/third_party/bevfusion/configs/nuscenes/det/default.yaml"
    # configs.load(config_path, recursive=True)
    # cfg = Config(recursive_eval(configs), filename=config_path)
    
    # cfg.model.pretrained = None

    # cfg.model.train_cfg = None
    # model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    # print(model)
    
    

    pass


