import torch
import json
import os
import copy
from argparse import ArgumentParser
from lib.imu_models import MW2StackRNNPooling
from lib.classification_head import Head
import random
import pytorch_lightning as pl
from dataset.ego4d.utils.utils import (
    get_ego4d_metadata,
    get_imu_frames,
    index_narrations,
)
from dataset.ego4d.dataloader import clean_narration_text
import math


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--gpus", default=1)
    parser.add_argument("--model_path", default='shane_models/i2c_s_i_t_t_ie_mw2_w_2.5_master_imu_encoder.pt')
    parser.add_argument("--video_uid", default='0a09f8fc-ff87-4210-b682-d2ae38af33eb')
    args = parser.parse_args()

    device = torch.device("cuda:0")
    loaded_imu_encoder = MW2StackRNNPooling(size_embeddings=512)
    loaded_imu_encoder.load_state_dict(torch.load(args.model_path))

    print("Keys are",loaded_imu_encoder)
    video_uid = args.video_uid
    signal_path = os.path.join('new_data/processed_imu', video_uid + '.npy')
    timestamps_path = os.path.join('new_data/processed_imu', video_uid + '_timestamps.npy')

    # preprocess imu data
    window_sec = float(args.model_path.split('_')[-4])
    print("Window sec:", window_sec)
    narration_dict, _ = index_narrations()
    window_idx = []

    narrations = narration_dict[video_uid]
    random.shuffle(narrations)
    video_duration = get_ego4d_metadata("video")[video_uid]["video_metadata"][
        "video_duration_sec"
    ]
    for (timestamp, text, a_uid, _) in narrations:
        if timestamp <= window_sec * 2:
            w_s = 0.0
            w_e = window_sec * 2

        elif timestamp + window_sec * 2 >= video_duration:
            w_s = video_duration - window_sec * 2
            w_e = video_duration
        else:
            w_s = timestamp - window_sec
            w_e = timestamp + window_sec

        w_s = int(math.floor(w_s))
        w_e = int(math.floor(w_e))

        input_dict = {
            "window_start": w_s,
            "window_end": w_e,
            "video_uid": video_uid,
            "narration_uid": a_uid,
            "text": clean_narration_text(text),
        }
        window_idx.append(input_dict)

    #sort by window start
    window_idx = sorted(window_idx, key=lambda k: k['window_start'])

    d_model = Head(encoder=loaded_imu_encoder, size_embeddings=512, n_classes=4)
    state_dict = torch.load("saved/downstream/downstream_i_ie_mw2_w_2.5_9674.ckpt")['state_dict']
    d_model.load_state_dict(state_dict, strict=False)
    print("Downstream model loaded.")
    print("The number of windows present are", len(window_idx))
    labels = {
            "head movement": 0,
            "stands up": 1,
            "sits down": 2,
            "walking": 3,
        }
    inverted_labels = {v: k for k, v in labels.items()}
    preds = {}
    for idx in window_idx:
        dict_out = copy.deepcopy(idx)
        uid = dict_out["video_uid"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]
        text = dict_out["text"]

        dict_out["imu"] = get_imu_frames(
            uid=uid,
            video_start_sec=w_s,
            video_end_sec=w_e,
        )

        if dict_out["imu"] is None:
            print("IMU is None for this window", w_s, w_e)
            continue

        imu_frames = dict_out["imu"]['signal']
        imu_frames = imu_frames.unsqueeze(0)
        imu_frames = imu_frames.float()

        out = d_model(imu_frames)
        # match with labels
        out = torch.argmax(out, dim=1)
        preds[w_s] = inverted_labels[out.item()]
    

    print("Predictions are", preds)
    #save the dictionary in a json file with the uid as the name in the activity folder
    with open('activity/'+video_uid+'.json', 'w') as fp:
        json.dump(preds, fp, indent=4)

    print("Saved the predictions in activity folder")


    




    





