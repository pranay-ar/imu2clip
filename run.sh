#!/bin/bash

# get current date
# current_date=$(date +'%Y-%m-%d')
current_date=$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')


# extract config file name from command line argument
config_file_name=$(basename "$1" .yaml)

# set log file name
log_file_name="pretraining_${config_file_name}_${current_date}.txt"

# run command and save output to log file
python pretraining.py --path_configs "$1" --gpus 4 >&1 | tee logs/"$log_file_name"
