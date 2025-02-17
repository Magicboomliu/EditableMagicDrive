import pickle

# nuscenes_dbinfos_train.pkl  nuscenes_gt_database  nuscenes_infos_train.pkl  nuscenes_infos_val.pkl
with open("/data/zliu/nuScenes/mmdet3d_2/nuscenes_infos_train.pkl", "rb") as f:
    nuscenes_infos = pickle.load(f)

print(type(nuscenes_infos))  # 一般是 list 或 dict
print(nuscenes_infos.keys())
