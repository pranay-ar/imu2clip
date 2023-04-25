#!/bin/bash
#SBATCH -c 24  # Number of Cores per Task
#SBATCH --mem=250000  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH --gres=gpu:3  # Number of GPUs
#SBATCH -t 72:00:00  # Job time limit
#SBATCH -o slurm_jobs/slurm-%j.out  # %j = job ID

source /home/pranayr_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate imu2clip

cd /home/pranayr_umass_edu/imu2clip/
bash run.sh /home/pranayr_umass_edu/imu2clip/configs/train_contrastive/ego4d_imu2text_mw2.yaml

# cd /home/pranayr_umass_edu/meta_project/imu2clip/
# bash download_videos.sh
# bash download_clips.sh