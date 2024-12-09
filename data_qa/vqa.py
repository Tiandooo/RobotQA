"""

run command: python vqa.py -m FAST-VQA -v D:\Astudy\RobotQA\data\kitchen_40demo\raw_data\3.0.0\exterior_image_1.mp4

"""
import yaml
import decord
from data_qa.fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from data_qa.fastvqa.models import DiViDeAddEvaluator
import torch
import numpy as np
import argparse


def sigmoid_rescale(score, model="FasterVQA"):
    mean, std = mean_stds[model]
    x = (score - mean) / std
    print(f"Inferring with model [{model}]:")
    score = 1 / (1 + np.exp(-x))
    return score

mean_stds = {
    "FasterVQA": (0.14759505, 0.03613452), 
    "FasterVQA-MS": (0.15218826, 0.03230298),
    "FasterVQA-MT": (0.14699507, 0.036453716),
    "FAST-VQA":  (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006), 
}

opts = {
    "FasterVQA": "data_qa/options/fast/f3dvqa-b.yml",
    "FasterVQA-MS": "data_qa/options/fast/fastervqa-ms.yml",
    "FasterVQA-MT": "data_qa/options/fast/fastervqa-mt.yml",
    "FAST-VQA": "data_qa/options/fast/fast-b.yml",
    "FAST-VQA-M": "data_qa/options/fast/fast-m.yml",
}


def get_video_quality(video_path, device='cpu', model="FAST-VQA"):
    # 每个线程内要设置bridge为torch，否则解码出来的视频类型不一致

    decord.bridge.set_bridge("torch")
    print(video_path)
    video_reader = decord.VideoReader(video_path, ctx=decord.cpu(0))

    opt = opts.get(model, opts["FAST-VQA"])
    with open(opt, "r") as f:
        opt = yaml.safe_load(f)

    ### Model Definition
    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(device)
    evaluator.load_state_dict(torch.load(opt["test_load_path"], map_location=device)["state_dict"])

    ### Data Definition
    vsamples = {}
    t_data_opt = opt["data"]["val-kv1k"]["args"]
    s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]
    for sample_type, sample_args in s_data_opt.items():
        ## Sample Temporally
        if t_data_opt.get("t_frag", 1) > 1:
            sampler = FragmentSampleFrames(fsize_t=sample_args["clip_len"] // sample_args.get("t_frag", 1),
                                           fragments_t=sample_args.get("t_frag", 1),
                                           num_clips=sample_args.get("num_clips", 1),
                                           )
        else:
            sampler = SampleFrames(clip_len=sample_args["clip_len"], num_clips=sample_args["num_clips"])

        num_clips = sample_args.get("num_clips", 1)
        frames = sampler(len(video_reader))
        print("Sampled frames are", frames)
        frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}

        imgs = [frame_dict[idx] for idx in frames]

        print("frame_dict", len(frame_dict))
        print("imgs", len(imgs))
        video = torch.stack(imgs, 0)
        video = video.permute(3, 0, 1, 2)

        ## Sample Spatially
        sampled_video = get_spatial_fragments(video, **sample_args)
        mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)

        sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1,
                                              *sampled_video.shape[2:]).transpose(0, 1)
        vsamples[sample_type] = sampled_video.to(device)
        print(sampled_video.shape)
    result = evaluator(vsamples)
    score = sigmoid_rescale(result.mean().item(), model=model)
    return score

