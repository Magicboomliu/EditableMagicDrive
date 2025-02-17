# Vanilla MagicDrive
View_Generation_With_Vanilla_MagicDrive_on_TrainSet(){
cd ..
resume_from_checkpoint="/home/Zihua/DEV/MagicDrive/trained_results/vanilla_magicdrive_small_trainingset/SDv1.5mv-rawbox_2025-02-07_08-16_224x400"

python tools/test_debug.py resume_from_checkpoint=$resume_from_checkpoint

}

# LiDAR MagicDrive
View_Generation_With_Lidar_MagicDrive_on_TrainSet(){
cd ..
resume_from_checkpoint="/home/Zihua/DEV/MagicDrive/trained_results/with_lidar_version_01/SDv1.5mv-rawbox_2025-02-14_14-49_224x400"

python tools/test_debug_lidar.py resume_from_checkpoint=$resume_from_checkpoint

}



View_Generation_With_Lidar_MagicDrive_on_TrainSet

# View_Generation_With_Vanilla_MagicDrive_on_TrainSet