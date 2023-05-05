#!/bin/bash

# get current date
# current_date=$(date +'%Y-%m-%d')
current_date=$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')


# extract config file name from command line argument
config_file_name=$(basename "$1" .yaml)

# set log file name
log_file_name="downstream_${config_file_name}_${current_date}.txt"

# run command and save output to log file
python downstream.py --path_configs "$1" --path_load_pretrained_imu_encoder "$2" --gpus 1 >&1 | tee logs/"$log_file_name"
