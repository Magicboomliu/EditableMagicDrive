import torch
import torch.nn as nn
import torch.nn.functional as F
import os


# create the soft-link
if __name__=="__main__":

    source_nuscenes_data_root_folder = "/data/nuScenes/nuscenes"
    source_nuscenes_dbinfos_train = os.path.join(source_nuscenes_data_root_folder,"nuscenes_dbinfos_train.pkl")
    source_nuscenes_gt_database = os.path.join(source_nuscenes_data_root_folder,"nuscenes_gt_database")
    source_nuscenes_infos_train = os.path.join(source_nuscenes_data_root_folder,"nuscenes_infos_train.pkl")
    source_nuscenes_infos_val = os.path.join(source_nuscenes_data_root_folder,"nuscenes_infos_val.pkl")

    assert os.path.exists(source_nuscenes_dbinfos_train)
    assert os.path.exists(source_nuscenes_gt_database)
    assert os.path.exists(source_nuscenes_infos_train)
    assert os.path.exists(source_nuscenes_infos_val)


    target_my_nuscenes_data_root_folder = "/data/zliu/nuScenes/mmdet3d_backup_2_small"
    target_nuscenes_dbinfos_train = os.path.join(target_my_nuscenes_data_root_folder,"nuscenes_dbinfos_train.pkl")
    target_nuscenes_gt_database = os.path.join(target_my_nuscenes_data_root_folder,"nuscenes_gt_database")
    target_nuscenes_infos_train = os.path.join(target_my_nuscenes_data_root_folder,"nuscenes_infos_train.pkl")
    target_nuscenes_infos_val = os.path.join(target_my_nuscenes_data_root_folder,"nuscenes_infos_val.pkl")

    if not os.path.exists(target_nuscenes_dbinfos_train):
        os.system("ln -s {} {}".format(source_nuscenes_dbinfos_train,target_nuscenes_dbinfos_train))
    if not os.path.exists(target_nuscenes_gt_database):
        os.system("ln -s {} {}".format(source_nuscenes_gt_database,target_nuscenes_gt_database))
    if not os.path.exists(target_nuscenes_infos_train):
        os.system("ln -s {} {}".format(source_nuscenes_infos_train,target_nuscenes_infos_train))
    if not os.path.exists(target_nuscenes_infos_val):
        os.system("ln -s {} {}".format(source_nuscenes_infos_val,target_nuscenes_infos_val))






    pass

