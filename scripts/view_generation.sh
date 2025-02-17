# Vanilla MagicDrive
View_Generation_With_Vanilla_MagicDrive(){
cd ..
resume_from_checkpoint="/home/Zihua/DEV/MagicDrive/pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400"
python tools/test.py resume_from_checkpoint=$resume_from_checkpoint
}



# LiDAR-based method
View_Generataion_With_Simple_LiDAR_MagicDrive(){
cd ..
resume_from_checkpoint="/home/Zihua/DEV/MagicDrive/trained_results/with_lidar_version_01/SDv1.5mv-rawbox_2025-02-14_14-49_224x400"
python tools/test_lidar.py resume_from_checkpoint=$resume_from_checkpoint

}


View_Generataion_With_Simple_LiDAR_MagicDrive
# View_Generation_With_Vanilla_MagicDrive