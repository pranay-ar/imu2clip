#!/bin/bash
#SBATCH -c 24  # Number of Cores per Task
#SBATCH --mem=250000  # Requested Memory
#SBATCH -p gypsum-m40  # Partition
#SBATCH --gres=gpu:4  # Number of GPUs
#SBATCH -t 72:00:00  # Job time limit
#SBATCH -o slurm_jobs/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=END

source /home/pranayr_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate imu2clip

cd /home/pranayr_umass_edu/imu2clip/
# bash run.sh /home/pranayr_umass_edu/imu2clip/configs/train_contrastive/ego4d_imu2text_patchrnn.yaml
# bash run.sh /home/pranayr_umass_edu/imu2clip/configs/train_contrastive/ego4d_imu2text_patchtransformer.yaml
# bash run.sh /home/pranayr_umass_edu/imu2clip/configs/train_contrastive/ego4d_imu2text_attentionpooled.yaml
# bash run.sh /home/pranayr_umass_edu/imu2clip/configs/train_contrastive/ego4d_imu2text_mw2.yaml
# bash run.sh /home/pranayr_umass_edu/imu2clip/configs/train_contrastive/ego4d_imu2text_convtransformer.yaml
# bash downstream.sh /home/pranayr_umass_edu/imu2clip/configs/train_downstream/default.yaml /home/pranayr_umass_edu/imu2clip/shane_models/i2c_s_i_t_v_ie_mw2_w_5.0_master_imu_encoder.pt
bash downstream.sh /home/pranayr_umass_edu/imu2clip/configs/train_downstream/ego4d_convtransformer.yaml
bash downstream.sh /home/pranayr_umass_edu/imu2clip/configs/train_downstream/ego4d_patchtransformer.yaml
bash downstream.sh /home/pranayr_umass_edu/imu2clip/configs/train_downstream/ego4d_patchrnn.yaml
bash downstream.sh /home/pranayr_umass_edu/imu2clip/configs/train_downstream/ego4d_attentionpooled.yaml

# cd /home/pranayr_umass_edu/meta_project/imu2clip/
# bash download_videos.sh
# bash download_clips.sh