import os
import sys
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tqdm import tqdm

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

sys.path.append(".")  # noqa
from magicdrive.runner.utils import concat_6_views
from magicdrive.misc.test_utils import (
    prepare_all, run_one_batch
)

transparent_bg = True
target_map_size = 400
# target_map_size = 800


def output_func(x): return concat_6_views(x)
# def output_func(x): return concat_6_views(x, oneline=True)
# def output_func(x): return img_concat_h(*x[:3])

# read the config
@hydra.main(version_base=None, config_path="../configs",
            config_name="test_config")
def main(cfg: DictConfig):

    # ------------------------------------------------------------------------------#

    if cfg.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')

    output_dir = to_absolute_path(cfg.resume_from_checkpoint) # /home/Zihua/DEV/MagicDrive/pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400
    
    # This is read from the user;s inpiy?
    original_overrides = OmegaConf.load(
        os.path.join(output_dir, "hydra/overrides.yaml")) # ['+exp=224x400', 'task_id=224x400']
    
    # print the original overraides
    current_overrides = HydraConfig.get().overrides.task # from user's inpit
    # getting the config name of this job.
    config_name = HydraConfig.get().job.config_name # test conigf
    # concatenating the original overrides with the current overrides
    # ['+exp=224x400', 'task_id=224x400', 'resume_from_checkpoint=/home/Zihua/DEV/MagicDrive/pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400']
    overrides = original_overrides + current_overrides
    # compose a new config from scratch
    cfg = hydra.compose(config_name, overrides=overrides)
    cfg.runner.validation_index = "demo"

    #--------------------------------------------------------------------------------------------------#

    #### setup everything #### Including the MagicDrive Pipeline and the Validation Dataloader
    pipe, val_dataloader, weight_dtype = prepare_all(cfg) # default test batch size = 4
    OmegaConf.save(config=cfg, f=os.path.join(cfg.log_root, "run_config.yaml"))

    # sampled data with 
    # dict_keys(['bev_map_with_aux', 'camera_param', 
                # 'kwargs', 'pixel_values', 
                # 'captions', 
                # 'input_ids', 
                # 'uncond_ids', 
                # 'meta_data'])
    # for data in val_dataloader:
    #     # here why channels is 8, I guess that is because there is 8 calasses
    #     print(data['bev_map_with_aux'].shape) # (Batch_size,8, H,W)
    #     print(data["camera_param"].shape) #(batch_size, 6,3,7) here 7 is first 3 is K, then 4 is the extrins.
    #     # 'kwargs including the 
    #     #   - bboxes_3d_data', where the bboxes3d_data contains: 
    #         # - bboxes
    #         # - claesses
    #         # - masks
    #         # Since they have random uncontrolled numbers of the each bounding boxs, since kinds of the configuration seems reasonable
    #     # what is the masks here
    #     print(data['kwargs']['bboxes_3d_data']['bboxes'].shape) # [batch_size, 6,N(nums of the bounding boxes),8,3]
    #     print(data['kwargs']['bboxes_3d_data']['classes'].shape) # [batch_size, 6, N]
    #     print(data['kwargs']['bboxes_3d_data']['masks'].shape)   # [batch_size, 6,N] ? maybe a binary mask in
    #     print(data['pixel_values'].shape) # images data, which format is [batch_size,6,3,H,W] --> multi-view images
        
    #     # captions is a list.should be aligned with the batch size,
    #     print(len(data['captions']))
    #     # A driving scene image at boston-seaport. Parked trucks, parked cars, avoid parked truck, ped, turn right.
    #     print(data['captions'][0])
    #     print(data['input_ids'].shape) #(batch_size, dimensions_of_token)
    #     print(data['uncond_ids'].shape)  #(batch_size, dimensions_of_token)
    #     print(data['meta_data'].keys()) # raw data
    #     # dict_keys(['gt_bboxes_3d', 'gt_labels_3d', 'camera_intrinsics', 
    #     # 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'metas'])


    #### start ####
    total_num = 0
    progress_bar = tqdm(
        range(len(val_dataloader) * cfg.runner.validation_times),
        desc="Steps",
    )
    for val_input in val_dataloader:

        # do the infernece here to get the final results
        return_tuples = run_one_batch(cfg, pipe, val_input, weight_dtype,
                                      transparent_bg=transparent_bg,
                                      map_size=target_map_size)


        for map_img, ori_imgs, ori_imgs_wb, gen_imgs_list, gen_imgs_wb_list in zip(*return_tuples):

            # save map
            map_img.save(os.path.join(cfg.log_root, f"{total_num}_map.png"))

            # save ori
            if ori_imgs is not None:
                ori_img = output_func(ori_imgs)
                ori_img.save(os.path.join(cfg.log_root, f"{total_num}_ori.png"))
            # save gen
            for ti, gen_imgs in enumerate(gen_imgs_list):
                gen_img = output_func(gen_imgs)
                gen_img.save(os.path.join(
                    cfg.log_root, f"{total_num}_gen{ti}.png"))


            if cfg.show_box:
                # save ori with box
                if ori_imgs_wb is not None:
                    ori_img_with_box = output_func(ori_imgs_wb)
                    ori_img_with_box.save(os.path.join(
                        cfg.log_root, f"{total_num}_ori_box.png"))
                # save gen with box
                for ti, gen_imgs_wb in enumerate(gen_imgs_wb_list):
                    gen_img_with_box = output_func(gen_imgs_wb)
                    gen_img_with_box.save(os.path.join(
                        cfg.log_root, f"{total_num}_gen{ti}_box.png"))

            total_num += 1

        # update bar
        progress_bar.update(cfg.runner.validation_times)


if __name__ == "__main__":
    main()
