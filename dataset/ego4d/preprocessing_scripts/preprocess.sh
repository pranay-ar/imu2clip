# Copyright (c) Meta Platforms, Inc. and affiliates.

#mkdir /checkpoint/andreamad8/ego4d

## Preprocess video
#mkdir /checkpoint/andreamad8/ego4d/full_videos
#mkdir /checkpoint/andreamad8/ego4d/full_videos/processed_video
python extract_video.py -v /home/pranayr_umass_edu/imu2clip/ego4d_data/v1/full_scale -o /work/pi_adrozdov_umass_edu/pranayr_umass_edu/checkpoint/pranay/full_videos/processed_video_multi -f 10 -s 224 
# python extract_video.py -v /home/pranayr_umass_edu/imu2clip/ego4d_data/v1/test_videos -o /work/pi_adrozdov_umass_edu/pranayr_umass_edu/checkpoint/pranay/test -f 10 -s 224
#mkdir /checkpoint/andreamad8/ego4d/clips
#mkdir /checkpoint/andreamad8/ego4d/clips/processed_video
# python extract_video.py -v /datasets01/ego4d_track2/v1/clips/ -o /checkpoint/andreamad8/ego4d/clips/processed_video -f 10 -s 224

## Preprocess IMU
#mkdir /checkpoint/andreamad8/ego4d/clips/processed_imu
#mkdir /checkpoint/andreamad8/ego4d/full_videos/processed_imu

# python extract_imu.py -v /datasets01/ego4d_track2/v1/imu -o /checkpoint/andreamad8/full_videos/processed_imu
# python extract_imu.py -v /datasets01/ego4d_track2/v1/imu -o /checkpoint/andreamad8/clips/processed_imu